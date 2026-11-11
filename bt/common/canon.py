"""Canonical hashing of action arguments — binds approval tokens to exact parameters.
Must match the gateway policy bt-agent-governance (Go: sorted keys, compact, UTF-8, no HTML escaping)."""
import hashlib
import json


def args_sha256(args) -> str:
    return hashlib.sha256(json.dumps(args or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
