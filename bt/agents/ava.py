"""AVA — advisor assistant (primary agent), LangGraph.

AMP chat contract: POST /chat {message, session_id, context} -> {response, ...}.
  Authorization: Bearer <user token>  (validated by the AMP ingress gateway, forwarded to the agent)
Extra: POST /resume {session_id} continues a run paused for human approval.

Routing is deterministic (keyword intents) so demo outcomes are repeatable; the LLM is used for
summaries and for proposing actions from content. Every tool/model call goes through the AI Gateway.
"""
import json
import os
import re
import time
from typing import Any, TypedDict

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel

from bt.agents import runtime as rt
from bt.common.jwtutil import UserTokens, bearer, is_delegated, unverified_claims

AGENT_NAME, AGENT_VERSION = "AVA", os.getenv("AGENT_VERSION", "1.0.0")
ENV = os.environ.get
CLIENT360_AGENT_URL = ENV("CLIENT360_AGENT_URL", "")
APPROVAL_URL = ENV("APPROVAL_URL", "http://approval-svc.bt-demo.svc.cluster.local:8080")
MODEL = ENV("LLM_MODEL", "gpt-4o-mini")
agent = rt.AgentID()
rt.SERVICE = AGENT_NAME
# External agents have no AMP ingress gateway in front of them (BLOCKERS B-08), so AVA validates the caller's
# user token itself: the external IdP (Asgardeo, IDP_ISSUER) or, without it, demo-idp.
users = UserTokens()


def authenticate(request: Request) -> str:
    token = bearer(request.headers)
    if not token:
        raise HTTPException(401, "user token required")
    try:
        claims = users.verify(token, audience="ava")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(401, f"invalid user token: {e}") from e
    if claims.get("token_use") != "user" or is_delegated(claims):
        raise HTTPException(401, "a user token is required")
    return token


def proxy_url(name: str) -> str:
    """AMP injects <PROXY_ID>_URL for each attached MCP proxy (may be a comma-separated list)."""
    return (ENV(f"{name.upper().replace('-', '_')}_URL") or "").split(",")[0].strip()


class S(TypedDict, total=False):
    user_token: str
    message: str
    context: dict
    intent: str
    args: dict
    delegated: str
    steps: list
    data: dict
    response: str
    status: str


def step(state: S, name: str, outcome: Any) -> list:
    return state.get("steps", []) + [{"step": name, "outcome": outcome}]


# ---------------------------------------------------------------- nodes
def route(state: S) -> S:
    m, ctx = state["message"], state.get("context") or {}
    acct = re.search(r"\b(\d{5})\b", m)
    if re.search(r"\b(tools|what can you do)\b", m, re.I):
        intent, args = "list_tools", {}
    elif re.search(r"\bnotes?\b", m, re.I) and acct:
        intent, args = "notes", {"account": acct.group(1)}
    elif re.search(r"\bemail\b", m, re.I):
        to = re.search(r"[\w.+-]+@[\w.-]+", m)
        intent, args = "email", {"to": to.group(0) if to else "client@example.com", "account": acct.group(1) if acct else "12345"}
    elif re.search(r"\bnews\b", m, re.I):
        intent, args = "news", {}
    elif re.search(r"\bholdings\b", m, re.I):
        intent, args = "holdings", {"account": acct.group(1) if acct else "", "version": "v1" if re.search(r"\bv1\b", m) else "v2"}
    elif re.search(r"\bC-\d{4}\b", m):
        intent, args = "client", {"clientId": re.search(r"C-\d{4}", m).group(0)}
    elif acct:
        intent, args = "summarize", {"account": acct.group(1)}
    else:
        intent, args = "chat", {}
    args["model"] = ctx.get("model", MODEL)  # demo knob for model governance (GW-D11)
    return {"intent": intent, "args": args, "steps": step(state, "route", {"intent": intent, "args": args})}


def delegate(state: S) -> S:
    try:
        tok = rt.exchange(agent, state["user_token"])
        c = unverified_claims(tok)
        out = {"user": c.get("username") or c.get("sub"), "agent_chain": rt.agent_chain(c), "scope": c.get("scope") or " ".join(c.get("scp") or []),
               # The two identities that went into the exchange, and what came out (claims only, never tokens).
               "user_token": rt.claims_summary(state["user_token"]),
               "actor_token": rt.claims_summary(agent.cached(ENV("DELEGATION_ACTOR_RESOURCE") or None)),
               "delegated_token": rt.claims_summary(tok), "sts": rt.STS_URL,
               # For the portal: the two sign-ins and the token they produce (previews + claims, never whole tokens).
               "views": {"agent": rt.token_view(agent.cached()), "agent_signed_in_now": agent.last_minted,
                         "delegated": rt.token_view(tok)}}
        return {"delegated": tok, "steps": step(state, "delegate", out)}
    except (PermissionError, rt.AgentIdentityNotProvisioned, rt.httpx.HTTPStatusError) as e:
        return {"status": "denied", "response": f"I can't act on your behalf: {e}",
                "steps": step(state, "delegate", {"ok": False, "error": str(e)})}


# Derived-content cache (CP-D4): "account|model" -> {"summary", "owner", "created"}. Reuse for ANY user (including the owner)
# only after re-fetching the source through the gateway under that user's delegation (entitlement re-check).
CACHE: dict[str, dict] = {}
CACHE_TTL = int(ENV("CACHE_TTL_SECONDS", "600"))


def summarize(state: S) -> S:
    acct = state["args"]["account"]
    key = f"{acct}|{state['args']['model']}"  # derived content depends on the source AND the model that produced it
    cached = CACHE.get(key)
    if cached and time.time() - cached["created"] > CACHE_TTL:
        CACHE.pop(key, None)
        cached = None
    with rt.tracer.start_as_current_span("AVA.delegate_to_Client360Agent") as span:
        span.set_attributes({"bt.callee": "Client360Agent", "bt.account": acct})
        h = {"authorization": f"Bearer {state['delegated']}", "content-type": "application/json"}
        rt.propagate.inject(h)
        r = rt.http.post(f"{CLIENT360_AGENT_URL.rstrip('/')}/invoke", json={"task": "portfolio_summary", "account": acct}, headers=h)
        sub = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"ok": False, "error": r.text}
    steps = step(state, "Client360Agent", {"http_status": r.status_code, **sub})
    if not sub.get("ok"):
        if cached:
            steps = step({"steps": steps}, "cache", {"hit": True, "owner": cached["owner"], "reused": False,
                                                    "reason": "source entitlement re-check denied"})
        denial = next((c["error"] for c in (sub.get("calls") or {}).values() if not c["ok"]), sub.get("error"))
        if not isinstance(denial, dict):
            denial = {"code": "UPSTREAM_AGENT_ERROR", "message": str(denial)[:300]}
        return {"status": rt.status_of({"error": denial}), "data": {"denial": denial}, "steps": steps,
                "response": f"I couldn't retrieve account {acct}: {rt.describe(denial)}"}
    facts = {k: v["result"] for k, v in sub["calls"].items()}
    if cached:  # source entitlement re-checked above for this user -> safe to reuse the derived summary
        steps = step({"steps": steps}, "cache", {"hit": True, "owner": cached["owner"],
                                                "reused_for": unverified_claims(state["user_token"]).get("sub"),
                                                "entitlement_rechecked": True})
        return {"status": "ok", "data": facts, "steps": steps, "response": cached["summary"]}
    llm = rt.llm_chat(agent, [
        {"role": "system", "content": "You are AVA, a private-wealth advisor assistant. Summarise in 3 short bullet points. No PII."},
        {"role": "user", "content": f"Summarise account {acct}. Client360Agent's brief: {sub.get('brief') or 'none'}. "
                                    f"Data: {json.dumps(facts)}"},
    ], state["args"]["model"], state["delegated"])
    steps = step({"steps": steps}, "llm", {k: v for k, v in llm.items() if k != "content"})
    if not llm["ok"]:
        return {"status": "partial", "data": facts, "steps": steps,
                "response": f"Retrieved account {acct}, but the model call was blocked: {rt.describe(llm.get('error'))}"}
    CACHE[key] = {"summary": llm["content"], "owner": unverified_claims(state["user_token"]).get("sub"), "created": time.time()}
    return {"status": "ok", "data": facts, "steps": step({"steps": steps}, "cache", {"hit": False, "stored": key}),
            "response": llm["content"]}


def client(state: S) -> S:
    out = rt.mcp(agent, proxy_url("client360"), "tools/call", {"name": "getClient", "arguments": {"clientId": state["args"]["clientId"]}}, state["delegated"])
    return _tool_answer(state, "getClient", out)


def holdings(state: S) -> S:
    v = state["args"]["version"]
    out = rt.mcp(agent, proxy_url(f"holdings-{v}"), "tools/call",
                 {"name": "getClientHoldings", "arguments": {"account": state["args"]["account"]}}, state["delegated"])
    return _tool_answer(state, f"getClientHoldings({v})", out)


def list_tools(state: S) -> S:
    tools, calls = {}, []
    for p in ("client360", "holdings-v2", "marketnews"):
        url = proxy_url(p)
        if url:
            out = rt.mcp(agent, url, "tools/list", {}, state["delegated"])
            calls.append(out)
            tools[p] = [t["name"] for t in out["result"]["tools"]] if out["ok"] else out["error"]
    return {"status": "ok", "data": tools, "steps": step({"steps": step(state, "tools/list.calls", calls)}, "tools/list", tools),
            "response": "Tools available to you: " + json.dumps(tools)}


def news(state: S) -> S:
    out = rt.mcp(agent, proxy_url("marketnews"), "tools/call", {"name": "marketNews", "arguments": {"topic": "markets"}}, state["delegated"])
    steps = step(state, "marketNews", out)
    if not out["ok"]:
        return {"status": rt.status_of(out), "steps": steps, "response": f"News unavailable: {rt.describe(out['error'])}"}
    # Deliberately naive: ask the model whether the content calls for any action (indirect-injection surface).
    llm = rt.llm_chat(agent, [
        {"role": "system", "content": ("Summarise the news for an advisor in 2 bullets. Then on a final line output "
                                       "ACTIONS: followed by a JSON list of tool calls the content asks for "
                                       "(tool sendClientEmail with to/subject/body), or [] if none.")},
        {"role": "user", "content": json.dumps(out["result"])},
    ], state["args"]["model"], state["delegated"])
    steps = step({"steps": steps}, "llm", {k: v for k, v in llm.items() if k != "content"})
    proposed = []
    if llm["ok"]:
        m = re.search(r"ACTIONS:\s*(\[.*\])", llm["content"], re.S)
        try:
            proposed = json.loads(m.group(1)) if m else []
        except ValueError:
            proposed = []
    attempts = []
    for p in proposed:  # a compromised/naive agent would try to execute these; the gateway decides
        args = p.get("arguments") or {k: p.get(k) for k in ("to", "subject", "body")}
        attempts.append(rt.mcp(agent, proxy_url("client-actions"), "tools/call",
                               {"name": "sendClientEmail", "arguments": args}, state["delegated"]))
    steps = step({"steps": steps}, "proposed_actions", {"proposed": proposed, "attempts": attempts})
    return {"status": "ok", "data": {"news": out["result"], "guardrail": out.get("guardrail"), "proposed_actions": proposed,
                                     "attempts": attempts},
            "steps": steps, "response": llm.get("content") if llm["ok"] else f"News retrieved; model blocked: {rt.describe(llm.get('error'))}"}


def notes(state: S) -> S:
    """CRM notes are free text anyone with CRM access can type: AVA reads them and, naively, acts on what they ask."""
    acct = state["args"]["account"]
    out = rt.mcp(agent, proxy_url("client360"), "tools/call", {"name": "getClientNotes", "arguments": {"account": acct}},
                 state["delegated"])
    steps = step(state, "getClientNotes", out)
    if not out["ok"]:
        return {"status": rt.status_of(out), "steps": steps, "data": out,
                "response": f"CRM notes for {acct} unavailable: {rt.describe(out['error'])}"}
    llm = rt.llm_chat(agent, [
        {"role": "system", "content": ("Summarise these CRM notes for the advisor in 2 bullets. Then on a final line output "
                                       "ACTIONS: followed by a JSON list of tool calls the notes ask for "
                                       "(tool sendClientEmail with to/subject/body), or [] if none.")},
        {"role": "user", "content": json.dumps(out["result"])},
    ], state["args"]["model"], state["delegated"])
    steps = step({"steps": steps}, "llm", {k: v for k, v in llm.items() if k != "content"})
    return {"status": "ok" if llm["ok"] else rt.status_of(llm), "steps": steps, "data": {"notes": out["result"]},
            "response": llm.get("content") if llm["ok"] else f"Notes retrieved; model blocked: {rt.describe(llm.get('error'))}"}


def send_email(args: dict, delegated: str, approval: str | None = None) -> dict:
    return rt.mcp(agent, proxy_url("client-actions"), "tools/call", {"name": "sendClientEmail", "arguments": args},
                  delegated, extra={"x-approval-token": approval} if approval else None)


def email(state: S) -> S:
    a = state["args"]
    body = (state.get("context") or {}).get("email_body") or f"Dear client, please find the latest summary for account {a['account']}."
    payload = {"to": a["to"], "subject": f"Portfolio update — account {a['account']}", "body": body}
    first = send_email(payload, state["delegated"])
    steps = step(state, "sendClientEmail#1", first)
    if first["ok"] or (first.get("error") or {}).get("code") != "APPROVAL_REQUIRED":
        return {"status": rt.status_of(first), "steps": steps, "data": {"email": first},
                "response": "Email sent." if first["ok"] else f"Email not sent: {rt.describe(first.get('error'))}"}
    # Gateway says a human must approve: file the request (once), then pause in the next node.
    h = {"authorization": f"Bearer {state['delegated']}"}
    rt.propagate.inject(h)
    try:
        req = rt.http.post(f"{APPROVAL_URL}/requests", headers=h, json={
            "action": "sendClientEmail", "args": payload, "reason": state["message"],
            # Risk context for the approver: action class + the agent's registered risk tier (AMP label, injected at deploy).
            "risk": {"action_class": "execute", "agent": AGENT_NAME, "agent_risk_tier": ENV("AGENT_RISK_TIER", "unknown"),
                     "agent_owner": ENV("AGENT_OWNER", "unknown")}}).json()
    except rt.httpx.HTTPError as e:  # e.g. a platform-hosted sandbox that may not reach the approval service
        return {"status": "error", "steps": step({"steps": steps}, "approval_requested", {"ok": False, "error": str(e)[:200]}),
                "response": ("This email needs an approver, but I couldn't reach the approval service from here "
                             f"({type(e).__name__}). Nothing was sent.")}
    return {"status": "awaiting_approval", "steps": step({"steps": steps}, "approval_requested", {"approval_id": req["id"]}),
            "data": {"approval_id": req["id"], "payload": payload}}


def await_approval(state: S) -> S:
    # interrupt() pauses here; on resume this node re-runs and interrupt() returns the /resume value.
    decision = interrupt({"approval_id": state["data"]["approval_id"], "action": "sendClientEmail", "args": state["data"]["payload"]})
    return {"data": {**state["data"], "decision": decision}, "steps": step(state, "approval", {"status": decision.get("status")})}


def email_with_approval(state: S) -> S:
    d = state["data"]
    if d["decision"].get("status") != "approved":
        return {"status": "denied", "response": "The approver declined the email."}
    second = send_email(d["payload"], state["delegated"], approval=d["decision"]["token"])
    return {"status": "ok" if second["ok"] else "denied", "steps": step(state, "sendClientEmail#2", second),
            "data": {"email": second, "approval_id": d["approval_id"]},
            "response": "Email sent with approval." if second["ok"] else f"Email still not sent: {rt.describe(second.get('error'))}"}


def chat(state: S) -> S:
    llm = rt.llm_chat(agent, [{"role": "user", "content": state["message"]}], state["args"]["model"], state["delegated"])
    return {"status": "ok" if llm["ok"] else "denied", "steps": step(state, "llm", {k: v for k, v in llm.items() if k != "content"}),
            "response": llm.get("content") if llm["ok"] else f"Model call blocked: {rt.describe(llm.get('error'))}"}


def _tool_answer(state: S, name: str, out: dict) -> S:
    return {"status": rt.status_of(out), "data": out, "steps": step(state, name, out),
            "response": json.dumps(out["result"]) if out["ok"] else f"{name}: {rt.describe(out['error'])}"}


def build():
    g = StateGraph(S)
    nodes = {"route": route, "delegate": delegate, "summarize": summarize, "client": client, "holdings": holdings,
             "list_tools": list_tools, "news": news, "notes": notes, "email": email, "await_approval": await_approval,
             "email_with_approval": email_with_approval, "chat": chat}
    for n, f in nodes.items():
        g.add_node(n, f)
    g.set_entry_point("route")
    g.add_edge("route", "delegate")
    g.add_conditional_edges("delegate", lambda s: END if s.get("status") == "denied" else s["intent"])
    g.add_conditional_edges("email", lambda s: "await_approval" if s.get("status") == "awaiting_approval" else END)
    g.add_edge("await_approval", "email_with_approval")
    for n in nodes:
        if n not in ("route", "delegate", "email", "await_approval"):
            g.add_edge(n, END)
    return g.compile(checkpointer=MemorySaver())


graph = build()
app = FastAPI(title=AGENT_NAME)


class ChatIn(BaseModel):
    message: str
    session_id: str
    context: dict | None = None


def _trace_id() -> str:
    ctx = rt.trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else ""


def _result(session_id: str, state: dict, started: float) -> dict:
    snap = graph.get_state({"configurable": {"thread_id": session_id}})
    paused = bool(snap.next)
    pending = snap.tasks[0].interrupts[0].value if paused and snap.tasks and snap.tasks[0].interrupts else None
    return {"response": ("Waiting for approval " + pending["approval_id"]) if paused else state.get("response"),
            "status": "awaiting_approval" if paused else state.get("status"), "pending_approval": pending,
            "agent": AGENT_NAME, "agent_version": AGENT_VERSION, "trace_id": _trace_id(),
            "steps": state.get("steps", []), "elapsed_ms": int((time.time() - started) * 1000)}


@app.post("/chat")
def chat_endpoint(body: ChatIn, request: Request):
    token = authenticate(request)
    started = time.time()
    with rt.tracer.start_as_current_span("AVA.chat") as span:
        u = unverified_claims(token)
        span.set_attributes({"bt.user": u.get("sub", ""), "bt.agent": AGENT_NAME, "bt.agent_version": AGENT_VERSION,
                             "bt.session_id": body.session_id, "bt.message": body.message})
        rt.evlog.emit(AGENT_NAME, "chat.received", user=u.get("username") or u.get("sub"), roles=u.get("roles"), message=body.message,
                      user_token=rt.claims_summary(token, ("iss", "sub", "aud", "roles")))
        state = graph.invoke({"user_token": token, "message": body.message, "context": body.context or {}, "steps": []},
                             {"configurable": {"thread_id": body.session_id}})
        out = _result(body.session_id, state, started)
        span.set_attributes({"bt.response": (out["response"] or "")[:4000], "bt.status": out["status"] or ""})
        rt.evlog.emit(AGENT_NAME, "chat.completed", status=out["status"], elapsed_ms=out["elapsed_ms"])
        return out


class ResumeIn(BaseModel):
    session_id: str


@app.post("/resume")
def resume(body: ResumeIn, request: Request):
    """Continue a paused run: fetch the approver's decision and hand it to the graph."""
    token = authenticate(request)
    cfg = {"configurable": {"thread_id": body.session_id}}
    snap = graph.get_state(cfg)
    if not snap.next:
        raise HTTPException(409, "nothing to resume")
    pending = snap.tasks[0].interrupts[0].value
    started = time.time()
    with rt.tracer.start_as_current_span("AVA.resume") as span:
        span.set_attributes({"bt.session_id": body.session_id, "bt.approval_request": pending["approval_id"]})
        # Re-delegate with the caller's fresh user token: the earlier delegated token may have expired.
        delegated = rt.exchange(agent, token)
        if unverified_claims(delegated).get("sub") != unverified_claims(snap.values["delegated"]).get("sub"):
            raise HTTPException(403, "only the original user can resume this session")
        r = rt.http.get(f"{APPROVAL_URL}/requests/{pending['approval_id']}", headers={"authorization": f"Bearer {delegated}"}).json()
        if r.get("status") == "pending":  # leave the paused run untouched, so it can be polled again
            return {"status": "awaiting_approval", "pending_approval": pending, "response": "Still waiting for approval."}
        graph.update_state(cfg, {"delegated": delegated, "user_token": token})
        state = graph.invoke(Command(resume={"status": r.get("status"), "token": r.get("token")}), cfg)
        out = _result(body.session_id, state, started)
        span.set_attributes({"bt.user": unverified_claims(token).get("sub", ""), "bt.agent": AGENT_NAME,
                             "bt.agent_version": AGENT_VERSION, "bt.response": (out["response"] or "")[:4000]})
        return out


@app.delete("/cache")
def clear_cache():
    """Demo reset only (cluster-internal; not routed through any gateway)."""
    CACHE.clear()
    return {"cleared": True}


@app.get("/healthz")
def healthz():
    return {"ok": True, "agent": AGENT_NAME, "version": AGENT_VERSION, "agentid": agent.provisioned}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
