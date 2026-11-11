"""JWT signing/verification helpers shared by demo services.

Keys are loaded from a PEM file (mounted Kubernetes secret) so tokens stay valid across
restarts; a key is generated in-memory only when no path is configured (local dev/tests).
"""
import json
import os
import time
import uuid

import httpx
import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from jwt.algorithms import ECAlgorithm, RSAAlgorithm


class Signer:
    def __init__(self, key_path: str | None, kid: str, alg: str = "RS256"):
        self.kid, self.alg = kid, alg
        if key_path and os.path.exists(key_path):
            with open(key_path, "rb") as f:
                self.key = serialization.load_pem_private_key(f.read(), password=None)
        elif alg == "ES256":
            self.key = ec.generate_private_key(ec.SECP256R1())
        else:
            self.key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def jwk(self) -> dict:
        algo = ECAlgorithm if self.alg == "ES256" else RSAAlgorithm
        jwk = json.loads(algo.to_jwk(self.key.public_key()))
        jwk.update({"kid": self.kid, "use": "sig", "alg": self.alg})
        return jwk

    def sign(self, claims: dict, ttl: int) -> str:
        now = int(time.time())
        body = {"iat": now, "nbf": now, "exp": now + ttl, "jti": str(uuid.uuid4()), **claims}
        return jwt.encode(body, self.key, algorithm=self.alg, headers={"kid": self.kid})


class JWKSVerifier:
    """Verifies tokens against a remote JWKS, with a small cache."""

    def __init__(self, jwks_url: str, issuer: str | None = None, ttl: int = 300):
        self.jwks_url, self.issuer, self.ttl = jwks_url, issuer, ttl
        self._client = jwt.PyJWKClient(jwks_url, cache_jwk_set=True, lifespan=ttl)

    def verify(self, token: str, audience: str | list[str] | None = None) -> dict:
        key = self._client.get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            key.key,
            algorithms=["RS256", "ES256", "PS256"],
            audience=audience,
            issuer=self.issuer,
            options={"verify_aud": audience is not None, "verify_iss": self.issuer is not None},
            leeway=30,
        )


def unverified_claims(token: str) -> dict:
    """Decode without verification — only for display/evidence of tokens the gateway already validated."""
    try:
        return jwt.decode(token, options={"verify_signature": False})
    except jwt.PyJWTError:
        return {}


def bearer(headers) -> str | None:
    """The caller's bearer token. Behind Agent Manager's agent gateway (OAuth front door), the verified token arrives
    as x-forwarded-authorization; services still verify it themselves."""
    value = (headers.get("authorization") or headers.get("Authorization") or headers.get("x-forwarded-authorization")
             or headers.get("X-Forwarded-Authorization") or "")
    return value[7:].strip() if value.lower().startswith("bearer ") else None


def client_credentials_token(token_url: str, client_id: str, client_secret: str, **params) -> dict:
    resp = httpx.post(
        token_url,
        data={"grant_type": "client_credentials", **params},
        auth=(client_id, client_secret),
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


class UserTokens:
    """Tokens from the external IdP (IDP_ISSUER, e.g. Asgardeo) and, without it, the demo IdP (USER_ISSUER).

    IdP claims are normalised to what the demo reads: `token_use` (user / agent / delegated, from Asgardeo's `aut`
    and `act`), `preferred_username` (the `username` attribute) and `roles` (`approver` when the user holds the
    approval scope)."""

    def __init__(self):
        local_jwks, local_iss = os.getenv("USER_JWKS_URL"), os.getenv("USER_ISSUER")
        self.local = JWKSVerifier(local_jwks, local_iss) if local_jwks else None
        self.idp_issuer = (os.getenv("IDP_ISSUER") or "").strip()
        self.idp_audience = (os.getenv("IDP_AUDIENCE") or os.getenv("IDP_CLIENT_ID") or "").strip() or None
        jwks = (os.getenv("IDP_JWKS_URL") or "").strip()
        self.idp = JWKSVerifier(jwks, self.idp_issuer) if jwks and self.idp_issuer else None

    def is_idp(self, token: str) -> bool:
        return bool(self.idp) and (unverified_claims(token).get("iss") or "") == self.idp_issuer

    def verify(self, token: str, audience: str | list[str] | None = None) -> dict:
        """`audience` applies to demo-IdP tokens; IdP tokens are checked against IDP_AUDIENCE (the app's client id)."""
        if self.is_idp(token):
            c = self.idp.verify(token, audience=self.idp_audience)
            c.setdefault("token_use", "delegated" if isinstance(c.get("act"), dict)
                         else "agent" if c.get("aut") == "AGENT" else "user")
            c.setdefault("preferred_username", c.get("username") or c.get("sub"))
            scopes = (c.get("scope") or "").split()
            c.setdefault("roles", ["approver"] if os.getenv("IDP_APPROVER_SCOPE", "approval:decide") in scopes else [])
            return c
        if self.local is None:
            raise ValueError("no user identity provider configured")
        return self.local.verify(token, audience=audience)


def agent_chain(claims: dict) -> list[str]:
    """The agents a delegated token passes through, oldest first: the demo STS writes `agent_chain`; the IdP nests the
    agents under `act` (agent ids, named via IDP_AGENT_NAMES)."""
    if claims.get("agent_chain"):
        return list(claims["agent_chain"])
    names = json.loads(os.getenv("IDP_AGENT_NAMES") or "{}")
    chain, act = [], claims.get("act")
    while isinstance(act, dict):
        chain.insert(0, names.get(act.get("sub"), act.get("sub")))
        act = act.get("act")
    return chain


def is_delegated(claims: dict) -> bool:
    return claims.get("token_use") == "delegated" or isinstance(claims.get("act"), dict)
