"""Shared agent runtime: AgentID, delegation, MCP/REST/LLM calls through the AI Gateway.

Nothing here holds a backend credential. The agent only ever has:
  - its AgentID client credential (injected by AMP as AMP_AGENTID_*; K8s secret reference)
  - the caller's user/delegated token (per request)
  - gateway URLs (injected by AMP when an MCP proxy / LLM provider is attached)
"""
import json
import re
import logging
import os
import time
import uuid

import httpx
from opentelemetry import propagate, trace

from bt.common import evlog, idp
from bt.common.jwtutil import agent_chain

log = logging.getLogger("agent")
tracer = trace.get_tracer("bt.agent")
TOKEN_EXCHANGE = "urn:ietf:params:oauth:grant-type:token-exchange"
DELEGATION_HEADER = os.getenv("DELEGATION_HEADER", "x-bt-delegation")
AGENT_TOKEN_HEADER = "x-bt-agent-token"
STS_URL = os.getenv("DELEGATION_STS_URL", "http://demo-idp.bt-demo.svc.cluster.local:8080/token")
http = httpx.Client(timeout=30)
NO_RETRY_TOOLS = set(os.getenv("NO_RETRY_TOOLS", "sendClientEmail").split(","))  # Execute class: never resend
SERVICE = "agent"  # set to the agent's name by AVA / Client360Agent; tags its log lines


class AgentIdentityNotProvisioned(RuntimeError):
    pass


class AgentID:
    """The agent's own identity. With the external IdP (IDP_*): the agent signs in to Asgardeo with its Agent ID and
    secret. Otherwise: an Agent Manager AgentID token (client_credentials + RFC 8707 resource), as the AMP docs describe."""

    def __init__(self):
        self.idp_mode = idp.enabled() and bool(os.getenv("IDP_AGENT_ID"))
        self.client_id = os.getenv("IDP_AGENT_ID") if self.idp_mode else os.getenv("AMP_AGENTID_CLIENT_ID")
        self.client_secret = os.getenv("IDP_AGENT_SECRET") if self.idp_mode else os.getenv("AMP_AGENTID_CLIENT_SECRET")
        self.token_endpoint = f"{idp.name()} (agent sign-in)" if self.idp_mode else os.getenv("AMP_AGENTID_TOKEN_ENDPOINT")
        self.scopes = os.getenv("AMP_AGENTID_SCOPES", "")
        self._cache: dict[str, tuple[str, float]] = {}
        self.last_minted = False  # whether the last token() call went to ThunderID (vs the cache) — shown in the portal

    @property
    def provisioned(self) -> bool:
        return all([self.client_id, self.client_secret, self.token_endpoint])

    def cached(self, resource: str | None = None) -> str | None:
        """The current token for a resource, without a request or a log line (for showing its claims)."""
        return self._cache.get("" if self.idp_mode else (resource or ""), (None, 0))[0]

    def token(self, resource: str | None = None) -> str:
        if not self.provisioned:
            raise AgentIdentityNotProvisioned("AMP_AGENTID_* not injected — identity not provisioned")
        key = "" if self.idp_mode else (resource or "")
        tok, exp = self._cache.get(key, (None, 0))
        if tok and exp - 60 > time.time():
            self.last_minted = False
            evlog.emit(SERVICE, "agentid.token.reused", resource=resource, expires_in=int(exp - time.time()))
            return tok
        if self.idp_mode:
            try:
                tok = idp.native_login(self.client_id, self.client_secret)
            except idp.IdPError as e:
                evlog.emit(SERVICE, "agentid.token.refused", token_endpoint=self.token_endpoint, error=str(e)[:200])
                raise AgentIdentityNotProvisioned(str(e)) from e
            self._cache[key] = (tok, float(claims_summary(tok, ("exp",)).get("exp", time.time() + 300)))
            self.last_minted = True
            evlog.emit(SERVICE, "agentid.token.issued", token_endpoint=self.token_endpoint, grant_type="agent sign-in",
                       claims=claims_summary(tok, ("iss", "sub", "aut", "scope")))
            return tok
        data = {"grant_type": "client_credentials"}
        if self.scopes:
            data["scope"] = self.scopes
        if resource:
            data["resource"] = resource
        r = http.post(self.token_endpoint, data=data, auth=(self.client_id, self.client_secret))
        if r.status_code != 200:
            evlog.emit(SERVICE, "agentid.token.refused", token_endpoint=self.token_endpoint, resource=resource,
                       http_status=r.status_code, error=r.text[:200])
        r.raise_for_status()
        body = r.json()
        self._cache[key] = (body["access_token"], time.time() + int(body.get("expires_in", 3600)))
        self.last_minted = True
        evlog.emit(SERVICE, "agentid.token.issued", token_endpoint=self.token_endpoint, grant_type="client_credentials",
                   resource=resource, claims=claims_summary(body["access_token"]))
        return body["access_token"]


def token_view(token: str | None) -> dict | None:
    """What the portal shows for a token: a short preview (never the whole token) and its decoded claims."""
    if not token:
        return None
    import jwt
    try:
        claims = jwt.decode(token, options={"verify_signature": False})
    except jwt.PyJWTError:
        return None
    return {"preview": f"{token[:14]}…{token[-6:]}", "claims": claims}


def claims_summary(token: str | None, keys=("iss", "sub", "username", "aut", "client_id", "aud", "scope", "roles", "role", "accounts",
                                          "agent_chain", "act", "token_use", "exp")) -> dict:
    """Decoded claims (never the token itself) for showing who a token says the caller is."""
    if not token:
        return {}
    import jwt
    try:
        c = jwt.decode(token, options={"verify_signature": False})
    except jwt.PyJWTError:
        return {}
    out = {k: c[k] for k in keys if k in c}
    if "agent_chain" in keys and agent_chain(c):
        out["agent_chain"] = agent_chain(c)
    if "role" in keys and "role" not in out and c.get("roles"):
        out["role"] = c["roles"][0]
    return out


def exchange(agent: AgentID, subject_token: str, audience: str = "bt-gateway", scope: str | None = None) -> str:
    """RFC 8693: user (or upstream delegated) token + this agent's identity -> delegated token. With the external IdP
    the exchange happens there (actor = the agent's own token); otherwise at the demo STS (actor = AgentID)."""
    if agent.idp_mode:
        return idp_exchange(agent, subject_token)
    with tracer.start_as_current_span("bt.delegation.exchange") as span:
        data = {"grant_type": TOKEN_EXCHANGE, "subject_token": subject_token,
                "subject_token_type": "urn:ietf:params:oauth:token-type:access_token",
                "actor_token": agent.token(os.getenv("DELEGATION_ACTOR_RESOURCE") or None),
                "actor_token_type": "urn:ietf:params:oauth:token-type:access_token", "audience": audience}
        if scope:
            data["scope"] = scope
        h: dict = {}
        propagate.inject(h)  # the STS logs the same trace id
        r = http.post(STS_URL, data=data, headers=h)
        if r.status_code != 200:
            span.set_status(trace.Status(trace.StatusCode.ERROR, r.text))
            evlog.emit(SERVICE, "delegation.refused", sts=STS_URL, http_status=r.status_code, error=r.text[:200])
            raise PermissionError(f"delegation refused: {r.text}")
        tok = r.json()["access_token"]
        evlog.emit(SERVICE, "delegation.exchanged", sts=STS_URL, grant_type="token-exchange (RFC 8693)",
                   subject=claims_summary(subject_token, ("sub", "agent_chain", "iss")),
                   actor=claims_summary(data["actor_token"], ("client_id", "iss")),
                   delegated=claims_summary(tok, ("sub", "role", "agent_chain", "scope", "aud")))
        return tok


def idp_exchange(agent: AgentID, subject_token: str) -> str:
    """Asgardeo token exchange: subject = the user's (or previous agent's) token, actor = this agent's token. The IdP
    checks the agent is registered and active, and narrows the scopes to what the subject holds."""
    with tracer.start_as_current_span("bt.delegation.exchange") as span:
        span.set_attribute("bt.sts", idp.name())
        held = (claims_summary(subject_token, ("scope",)).get("scope") or "").split()
        scopes = [x for x in held if x in idp.api_scopes() or x.startswith(("account:", "approval:"))]
        try:
            tok = idp.exchange(subject_token, agent.token(), scopes)
        except idp.IdPError as e:
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)[:200]))
            evlog.emit(SERVICE, "delegation.refused", sts=idp.name(), error=str(e)[:200])
            raise PermissionError(str(e)) from e
        evlog.emit(SERVICE, "delegation.exchanged", sts=idp.name(), grant_type="token-exchange (RFC 8693)",
                   subject=claims_summary(subject_token, ("sub", "username", "agent_chain", "iss")),
                   actor=claims_summary(agent.cached(), ("sub", "aut", "iss")),
                   delegated=claims_summary(tok, ("sub", "username", "agent_chain", "scope", "aud")))
        return tok


def _headers(agent: AgentID, resource: str, delegated: str | None, extra: dict | None = None) -> dict:
    if agent.idp_mode:
        # One IdP token carries both identities (sub = user, act = agent); the gateway's mcp-auth verifies it.
        h = {"authorization": f"Bearer {delegated or agent.token()}", "content-type": "application/json",
             "accept": "application/json, text/event-stream", "mcp-protocol-version": "2025-06-18"}
        h.update(extra or {})
        propagate.inject(h)
        return h
    tok = agent.token(resource)
    # Authorization is consumed by AMP's managed mcp-auth; the copy lets bt-agent-governance verify the agent too, because
    # the proxy's upstream credential replaces Authorization before endpoint policies run (BLOCKERS B-09).
    h = {"authorization": f"Bearer {tok}", AGENT_TOKEN_HEADER: tok, "content-type": "application/json",
         "accept": "application/json, text/event-stream", "mcp-protocol-version": "2025-06-18"}
    if delegated:
        h[DELEGATION_HEADER] = f"Bearer {delegated}"  # validated by the gateway's jwt-auth policy
    h.update(extra or {})
    propagate.inject(h)
    return h


def _parse_mcp(r: httpx.Response) -> dict:
    text = r.text
    if r.headers.get("content-type", "").startswith("text/event-stream"):
        text = next((ln[5:].strip() for ln in text.splitlines() if ln.startswith("data:")), "{}")
    try:
        return json.loads(text) if text else {}
    except ValueError:
        return {"raw": text}


GUARDRAIL_TEXT = {
    ("regex-guardrail", "RESPONSE"): "Withheld by the gateway: the tool's output contained a blocked pattern "
                                     "(e.g. an injected instruction)",
    ("regex-guardrail", "REQUEST"): "Blocked by the gateway: the request contained a blocked pattern (e.g. client PII)",
    ("llm", "regex-guardrail", "REQUEST"): "This model is not on the firm's approved list",
    ("llm", "word-count-guardrail", "REQUEST"): "The prompt is longer than the firm allows",
    ("llm", "content-length-guardrail", "REQUEST"): "The request is larger than AVA's limit",
    ("llm", "SemanticPromptGuard", "REQUEST"): "The request reads like an attempt to override AVA's rules or move client "
                                               "data out of the firm",
}
# Envoy's own replies when the upstream itself is unreachable (plain text, not a policy decision).
UPSTREAM_DOWN = ("no healthy upstream", "upstream connect error", "upstream_reset", "connection termination",
                 "error 1033")  # 1033: the public tunnel to the tool servers is reconnecting (B-04)
DOWN_STATUS = (502, 503, 530)


def gateway_error(status: int, out: dict, text: str, path: str = "mcp", www: str = "", tool: str = "") -> dict:
    """Every gateway / backend failure as one shape: {code, policy, retryable, userMessage, http_status, detail}."""
    err = out.get("error") if isinstance(out, dict) else None
    if isinstance(err, dict) and err.get("code") == -32000 and "not allowed" in str(err.get("message", "")):
        return {"code": "TOOL_NOT_ALLOWED", "policy": "mcp-acl-list", "retryable": False, "http_status": status,
                "userMessage": f"Access control does not let this user or agent use {tool or 'this tool'} here",
                "detail": err}  # bundled mcp-acl-list (Agent Manager → Access Control)
    if isinstance(err, dict) and isinstance(err.get("data"), dict) and err["data"].get("code"):
        return {**err["data"], "http_status": status}  # bt-agent-governance / respond: already structured
    msg = out.get("message") if isinstance(out, dict) else None
    if isinstance(msg, dict) and msg.get("action") == "GUARDRAIL_INTERVENED":  # bundled guardrails
        g, d = msg.get("interveningGuardrail", out.get("type", "guardrail")), msg.get("direction", "")
        return {"code": "GUARDRAIL_INTERVENED", "policy": g, "direction": d, "retryable": False, "http_status": status,
                "userMessage": GUARDRAIL_TEXT.get((path, g, d)) or GUARDRAIL_TEXT.get((g, d), f"Stopped by the gateway guardrail {g}"),
                "detail": msg.get("assessments") or msg.get("actionReason")}
    if status == 403 and "insufficient_scope" in www:  # bundled mcp-authz: the token lacks the tool's scope
        need = re.search(r'scope="([^"]*)"', www)
        return {"code": "INSUFFICIENT_SCOPE", "policy": "mcp-authz", "retryable": False, "http_status": status,
                "userMessage": f"Your role doesn't allow {tool or 'this tool'}" + (f" (needs {need.group(1)})" if need else ""),
                "detail": www}
    raw = (out.get("raw") if isinstance(out, dict) else None) or text or ""
    if status == 530 or status in (*DOWN_STATUS, 504) and any(k in raw.lower() for k in UPSTREAM_DOWN):
        return {"code": "UPSTREAM_UNAVAILABLE", "policy": None, "retryable": True, "http_status": status,
                "userMessage": "The service behind the gateway is not reachable right now", "detail": raw[:200]}
    if status == 401:
        return {"code": "GATEWAY_AUTH_FAILED", "policy": "mcp-auth / jwt-auth", "retryable": False, "http_status": status,
                "userMessage": "The gateway did not accept the agent's or the user's token",
                "detail": msg if isinstance(msg, str) else raw[:200]}
    if isinstance(err, dict):
        return {"code": str(err.get("code", "GATEWAY_ERROR")), "policy": err.get("policy"), "retryable": False,
                "http_status": status, "userMessage": err.get("message") or json.dumps(err)[:200], "detail": err}
    return {"code": f"HTTP_{status}", "policy": None, "retryable": status >= 500, "http_status": status,
            "userMessage": (msg if isinstance(msg, str) else None) or (str(err) if err else raw[:200]) or "request failed",
            "detail": out}


def describe(err) -> str:
    """One sentence for the chat: what happened, then the code and policy."""
    if not isinstance(err, dict):
        return str(err)
    tail = ", ".join(x for x in (err.get("code"), err.get("policy")) if x)
    return f"{err.get('userMessage') or err.get('message') or 'request failed'}" + (f" ({tail})" if tail else "")


def status_of(out: dict) -> str:
    """ok / denied (a control said no) / error (something was down)."""
    if out.get("ok"):
        return "ok"
    return "error" if (out.get("error") or {}).get("code") in ("UPSTREAM_UNAVAILABLE",) else "denied"


def mcp(agent: AgentID, url: str, method: str, params: dict | None, delegated: str | None, extra: dict | None = None) -> dict:
    """One JSON-RPC call to an MCP proxy on the gateway. Returns a normalized outcome dict."""
    with tracer.start_as_current_span(f"mcp {method} {(params or {}).get('name', '')}".strip()) as span:
        span.set_attributes({"mcp.method": method, "mcp.url": url,
                             "mcp.tool": (params or {}).get("name", ""),
                             "mcp.arguments": json.dumps((params or {}).get("arguments", {}))})
        body = {"jsonrpc": "2.0", "id": str(uuid.uuid4()), "method": method, "params": params or {}}
        span_id = format(span.get_span_context().span_id, "016x")  # the gateway's spans for this call are its children
        try:
            headers = _headers(agent, url, delegated, extra)
            auth = {"agent_token": claims_summary(agent.cached(url) or headers["authorization"][7:]),
                    "agent_token_minted": agent.last_minted, "single_token": agent.idp_mode,
                    "agent_token_endpoint": agent.token_endpoint, "delegation": claims_summary(delegated)}
        except AgentIdentityNotProvisioned as e:  # the IdP refused the agent's own sign-in
            err = {"code": "AGENT_SIGNIN_REFUSED", "retryable": False, "policy": idp.name(),
                   "userMessage": f"{idp.name()} did not let this agent sign in", "detail": str(e)[:200]}
            span.set_status(trace.Status(trace.StatusCode.ERROR, err["detail"]))
            return {"ok": False, "http_status": 401, "policy": idp.name(), "error": err, "url": url, "method": method,
                    "tool": (params or {}).get("name"), "span_id": span_id,
                    "auth": {"agent_token_refused": True, "agent_token_endpoint": agent.token_endpoint,
                             "delegation": claims_summary(delegated)}}
        except httpx.HTTPStatusError as e:
            # ThunderID refused to mint an AgentID token for this resource (agent not granted / unknown resource).
            err = {"code": "AGENTID_TOKEN_REFUSED", "retryable": False, "policy": "AMP-AgentID",
                   "userMessage": "Agent Manager does not let this agent use this tool server (no AgentID token issued)",
                   "message": f"AgentID token refused for {url}: {e.response.text}", "detail": e.response.text[:200]}
            span.set_status(trace.Status(trace.StatusCode.ERROR, err["message"][:200]))
            return {"ok": False, "http_status": e.response.status_code, "policy": "AMP-AgentID", "error": err, "url": url,
                    "method": method, "tool": (params or {}).get("name"), "span_id": span_id,
                    "auth": {"agent_token_refused": True, "agent_token_endpoint": agent.token_endpoint,
                             "delegation": claims_summary(delegated)}}
        if not url:
            return {"ok": False, "http_status": 0, "error": {"code": "NOT_ATTACHED", "message": "proxy not attached"}}
        evlog.emit(SERVICE, "mcp.request", url=url, method=method, tool=(params or {}).get("name"),
                   authorization=f"AgentID {auth['agent_token'].get('client_id')}",
                   x_bt_delegation=f"user {auth['delegation'].get('sub')} via {' → '.join(auth['delegation'].get('agent_chain') or [])}")
        auth["request"] = {"method": "POST", "url": url, "tool": (params or {}).get("name") or method,
                           "authorization": token_view(headers["authorization"][7:])}  # what the gateway receives
        r = http.post(url, json=body, headers=headers)
        if ((r.status_code == 530 or r.status_code in DOWN_STATUS and any(k in r.text.lower() for k in UPSTREAM_DOWN))
                and (params or {}).get("name") not in NO_RETRY_TOOLS):
            # The public tunnel to the mock tools can blip; the gateway did not reach the tool, so one retry is safe
            # (never for Execute-class tools).
            evlog.emit(SERVICE, "mcp.retry", url=url, tool=(params or {}).get("name"), http_status=r.status_code,
                       reason=r.text[:120])
            time.sleep(2)
            r = http.post(url, json=body, headers=headers)
        out = _parse_mcp(r)
        outcome = {"http_status": r.status_code, "policy": r.headers.get("x-bt-policy"),
                   "guardrail": r.headers.get("x-bt-guardrail"), "url": url, "method": method,
                   "tool": (params or {}).get("name"), "span_id": span_id, "auth": auth}
        if r.status_code >= 400 or "error" in out or "result" not in out:
            outcome.update({"ok": False, "error": gateway_error(r.status_code, out, r.text, www=r.headers.get("www-authenticate", ""),
                                                                tool=(params or {}).get("name") or "")})
            evlog.emit(SERVICE, "mcp.denied", url=url, tool=(params or {}).get("name"), http_status=r.status_code,
                       error=outcome["error"])
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(outcome["error"])[:200]))
            return outcome
        result = out["result"]
        if method == "tools/call":
            structured = result.get("structuredContent")
            if structured is None:
                texts = [c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"]
                try:
                    structured = json.loads(texts[0]) if len(texts) == 1 else texts
                except ValueError:
                    structured = texts
            if isinstance(structured, dict) and set(structured) == {"result"}:
                structured = structured["result"]
            outcome.update({"ok": not result.get("isError"), "result": structured})
        else:
            outcome.update({"ok": True, "result": result})
        evlog.emit(SERVICE, "mcp.response", url=url, tool=(params or {}).get("name"), http_status=r.status_code,
                   ok=outcome["ok"])
        return outcome


def llm_chat(agent: AgentID, messages: list[dict], model: str, delegated: str | None) -> dict:
    """Chat completion through the AMP LLM Service Provider on the gateway (OpenAI-compatible)."""
    from openai import APIStatusError, OpenAI

    base = os.environ.get("LLM_PROVIDER_URL") or os.environ.get("OPENAI_URL", "")
    if not base:
        return {"ok": False, "error": {"message": "no LLM provider attached to this agent"}}
    auth = os.getenv("LLM_AUTH", "agentid")  # agentid | apikey
    key = agent.token(base) if auth == "agentid" else (os.getenv("LLM_PROVIDER_KEY") or os.getenv("OPENAI_API_KEY", ""))
    extra = {DELEGATION_HEADER: f"Bearer {delegated}"} if delegated else {}
    propagate.inject(extra)
    header = os.getenv("LLM_KEY_HEADER", "X-API-Key")
    client = OpenAI(base_url=base, api_key="gateway-managed", max_retries=0, timeout=60,
                    default_headers={header: key if auth == "apikey" else f"Bearer {key}"})
    try:
        resp = client.chat.completions.with_raw_response.create(model=model, messages=messages, extra_headers=extra)
        parsed = resp.parse()
        evlog.emit(SERVICE, "llm.response", provider=base, model=parsed.model, http_status=resp.status_code,
                   total_tokens=parsed.usage.total_tokens if parsed.usage else None)
        return {"ok": True, "model": parsed.model, "content": parsed.choices[0].message.content,
                "usage": parsed.usage.model_dump() if parsed.usage else None}
    except APIStatusError as e:
        try:
            body = e.response.json()
        except ValueError:
            body = {"raw": e.response.text}
        body = gateway_error(e.status_code, body, e.response.text, path="llm")
        evlog.emit(SERVICE, "llm.denied", provider=base, model=model, http_status=e.status_code, error=body)
        return {"ok": False, "http_status": e.status_code, "error": body}
