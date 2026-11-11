"""The external identity provider (Asgardeo) as the demo uses it: one OIDC application, users and agents sign in to
it, and agents exchange tokens to act for a user. Configured by IDP_* environment variables (deploy-demo.sh writes
them from .secrets/asgardeo.env); `enabled()` is False without them.

  native_login(name, secret)  app-native sign-in (no browser): a user's username + password, or an agent's
                              Agent ID + Agent Secret -> that identity's access token (JWT)
  exchange(subject, actor)    RFC 8693 token exchange -> delegated token: sub = the user, act.sub = the agent
                              (nested when an agent re-exchanges a delegated token)
"""
import base64
import functools
import hashlib
import json
import os
import secrets

import httpx

http = httpx.Client(timeout=30)
TOKEN_EXCHANGE = "urn:ietf:params:oauth:grant-type:token-exchange"
JWT_TYPE = "urn:ietf:params:oauth:token-type:jwt"


def env(name: str, default: str = "") -> str:
    return (os.getenv(name) or default).strip()


def enabled() -> bool:
    return bool(env("IDP_ISSUER") and env("IDP_CLIENT_ID"))


def name() -> str:
    return env("IDP_NAME", "BT IdP")


def api_scopes() -> list[str]:
    return env("IDP_API_SCOPES", "client:read portfolio:read news:read email:send").split()


def request_scopes() -> list[str]:
    """Everything a sign-in asks for; the IdP grants only what the user's (or agent's) roles allow."""
    extra = env("IDP_EXTRA_SCOPES", "account:12345 account:98765 approval:decide").split()
    return ["openid", "profile", *api_scopes(), *extra]


@functools.cache
def endpoints() -> dict:
    """From the IdP's OpenID discovery document; Asgardeo's app-native `authn` endpoint sits next to `token`."""
    d = http.get(env("IDP_ISSUER").rstrip("/") + "/.well-known/openid-configuration").json()
    return {"authorize": d["authorization_endpoint"], "token": d["token_endpoint"], "jwks": d["jwks_uri"],
            "authn": d["token_endpoint"].rsplit("/", 1)[0] + "/authn"}


def _client() -> tuple[str, str]:
    return env("IDP_CLIENT_ID"), env("IDP_CLIENT_SECRET")


class IdPError(RuntimeError):
    pass


def native_login(username: str, password: str, scopes: list[str] | None = None) -> str:
    """Sign in once more if the IdP drops the connection mid-flow (it's an internet service)."""
    try:
        return _native_login(username, password, scopes)
    except httpx.TransportError:
        return _native_login(username, password, scopes)


def _native_login(username: str, password: str, scopes: list[str] | None = None) -> str:
    """App-native authentication: authorize (response_mode=direct) -> authn with the credentials -> code -> token.
    Each sign-in uses its own connection: a shared cookie jar would carry the previous identity's IdP session."""
    http = httpx.Client(timeout=30)
    cid, secret = _client()
    redirect = env("IDP_REDIRECT_URI", "http://localhost:8091/callback")
    verifier = secrets.token_urlsafe(48)
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    r = http.post(endpoints()["authorize"], headers={"Accept": "application/json"}, data={
        "client_id": cid, "client_secret": secret, "response_type": "code", "redirect_uri": redirect,
        "scope": " ".join(scopes or request_scopes()), "response_mode": "direct",
        "code_challenge": challenge, "code_challenge_method": "S256"})
    if r.status_code != 200:
        raise IdPError(f"{name()} sign-in start failed: HTTP {r.status_code} {r.text[:200]}")
    flow = r.json()
    auths = (flow.get("nextStep") or {}).get("authenticators") or []
    basic = next((a for a in auths if "username" in json.dumps(a).lower()), None)
    if not basic:
        raise IdPError(f"{name()} offered no username/password step: {[a.get('authenticator') for a in auths]}")
    r = http.post(endpoints()["authn"], json={"flowId": flow["flowId"], "selectedAuthenticator": {
        "authenticatorId": basic["authenticatorId"], "params": {"username": username, "password": password}}})
    body = r.json()
    code = (body.get("authData") or {}).get("code") or body.get("code")
    if not code:
        msg = ((body.get("nextStep") or {}).get("messages") or [{}])[0].get("message") or body.get("description")
        raise IdPError(f"{name()} sign-in refused: {msg or r.text[:200]}")
    r = http.post(endpoints()["token"], auth=(cid, secret), data={
        "grant_type": "authorization_code", "code": code, "redirect_uri": redirect, "code_verifier": verifier})
    if r.status_code != 200:
        raise IdPError(f"{name()} token request failed: HTTP {r.status_code} {r.text[:200]}")
    return r.json()["access_token"]


def exchange(subject_token: str, actor_token: str, scopes: list[str]) -> str:
    """Delegated token for the actor (agent) acting for the subject (user). The IdP narrows the scopes to what the
    subject already holds; it checks the actor is a registered, active agent."""
    r = http.post(endpoints()["token"], auth=_client(), data={
        "grant_type": TOKEN_EXCHANGE, "subject_token": subject_token, "subject_token_type": JWT_TYPE,
        "actor_token": actor_token, "actor_token_type": JWT_TYPE,
        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token", "scope": " ".join(scopes)})
    if r.status_code != 200:
        raise IdPError(f"{name()} token exchange refused: HTTP {r.status_code} {r.text[:300]}")
    return r.json()["access_token"]
