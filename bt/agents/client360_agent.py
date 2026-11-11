"""Client360Agent — sub-agent that fetches client data via the client360 MCP proxy.

POST /invoke {"task": "portfolio_summary", "account": "12345"}
  Authorization: Bearer <delegated token issued to AVA>   (validated by the AMP ingress gateway)
It exchanges that token for its own hop (sub=user, act={Client360Agent, act={AVA}}) and calls tools.
"""
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from bt.agents import runtime as rt
from bt.common.jwtutil import UserTokens, agent_chain, bearer, is_delegated, unverified_claims

AGENT_NAME, AGENT_VERSION = "Client360Agent", os.getenv("AGENT_VERSION", "1.0.0")
MCP_URL = os.getenv("CLIENT360_MCP_URL") or os.getenv("CLIENT360_URL", "")
agent = rt.AgentID()
rt.SERVICE = AGENT_NAME
# Accept only delegated tokens issued to AVA (the one agent allowed to call this sub-agent).
callers = UserTokens()  # delegated tokens from Okta (on-behalf-of exchange) or the demo STS
app = FastAPI(title=AGENT_NAME)


class Task(BaseModel):
    task: str = "portfolio_summary"
    account: str


@app.get("/healthz")
def healthz():
    return {"ok": True, "agent": AGENT_NAME, "version": AGENT_VERSION, "agentid": agent.provisioned}


@app.post("/invoke")
def invoke(task: Task, request: Request):
    upstream = bearer(request.headers)
    if not upstream:
        raise HTTPException(401, "delegated token required")
    try:
        inbound = callers.verify(upstream, audience="bt-gateway")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(401, f"invalid delegated token: {e}") from e
    if not is_delegated(inbound) or (agent_chain(inbound) or [None])[-1] != "AVA":
        raise HTTPException(403, "only AVA may delegate to Client360Agent")
    # Continue AVA's trace (W3C traceparent) so one trace id covers the whole delegation chain.
    with rt.tracer.start_as_current_span("Client360Agent.invoke", context=rt.propagate.extract(dict(request.headers))) as span:
        span.set_attributes({"bt.agent": AGENT_NAME, "bt.agent_version": AGENT_VERSION, "bt.task": task.task,
                             "bt.account": task.account})
        rt.evlog.emit(AGENT_NAME, "a2a.accepted", caller=agent_chain(inbound), user=inbound.get("sub"),
                      task=task.task, account=task.account)
        try:
            delegated = rt.exchange(agent, upstream)
        except PermissionError as e:
            return {"ok": False, "agent": AGENT_NAME, "error": {"code": "DELEGATION_REFUSED", "message": str(e)}}
        except (rt.httpx.HTTPStatusError, rt.AgentIdentityNotProvisioned) as e:
            # ThunderID refused this agent's own credential (e.g. AgentID revoked/suspended in Agent Manager).
            msg = e.response.text if isinstance(e, rt.httpx.HTTPStatusError) else str(e)
            return {"ok": False, "agent": AGENT_NAME, "error": {
                "code": "AGENTID_REVOKED", "retryable": False, "policy": "AMP-AgentID",
                "message": f"Client360Agent has no valid AgentID (suspended in Agent Manager): {msg[:200]}"}}
        claims = unverified_claims(delegated)
        calls = {name: rt.mcp(agent, MCP_URL, "tools/call", {"name": name, "arguments": {"account": task.account}}, delegated)
                 for name in ("getPortfolio", "getPerformance")}
        return {
            "ok": all(c["ok"] for c in calls.values()), "agent": AGENT_NAME, "agent_version": AGENT_VERSION,
            "hop": {"user": claims.get("username") or claims.get("sub"), "agent_chain": agent_chain(claims), "act": claims.get("act"),
                    "delegated_scope": claims.get("scope") or " ".join(claims.get("scp") or []), "inbound_token": rt.claims_summary(upstream),
                    "actor_token": rt.claims_summary(agent.cached(os.getenv("DELEGATION_ACTOR_RESOURCE") or None)),
                    "delegated_token": rt.claims_summary(delegated)},
            "calls": calls,
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
