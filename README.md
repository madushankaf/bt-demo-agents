# bt-demo-agents

The two agents of the Bessemer Trust demo on **WSO2 Agent Manager**, built and run by Agent Manager as
platform-hosted agents:

| Agent | What it does | `APP` | Interface |
|---|---|---|---|
| **AVA** | Advisor assistant (LangGraph). Delegates portfolio work to Client360Agent; calls tools and the approved model through the AI Gateway; pauses for human approval on high-risk actions. | `ava` | chat API (`POST /chat`, `POST /resume`) |
| **Client360Agent** | AVA's sub-agent: fetches portfolio data through the AI Gateway and writes AVA a short client brief with the approved model. | `client360-agent` | custom API (`POST /invoke`, `openapi/client360-agent.json`) |

One image for both: `Dockerfile`, with `APP` selecting the agent.

**No credentials live in this repository.** At run time Agent Manager injects:
- each agent's identity (Agent ID + secret from the firm's identity provider, the BT IdP);
- the AI Gateway URLs of its MCP servers and LLM provider;
- tracing settings.

The agents never hold a backend or model provider credential: the gateway injects those.

How each user request flows: the user signs in to the BT IdP, then AVA signs in with its own agent credentials. AVA
exchanges the user's token for one delegated token (user in `sub`, AVA in `act`). Every tool call carries that token
to the AI Gateway, which verifies it and applies the firm's policies.
