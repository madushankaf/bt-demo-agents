"""One structured log line per security-relevant event, tagged with the W3C trace id so a request's lines can be
pulled from every component's logs (the advisor portal's Technical log does this). Claims only, never tokens."""
import json
import sys
from datetime import datetime, timezone

from opentelemetry import trace


def trace_id_from(traceparent: str | None) -> str:
    parts = (traceparent or "").split("-")
    return parts[1] if len(parts) == 4 else ""


def emit(service: str, event: str, trace_id: str | None = None, **fields):
    if trace_id is None:
        ctx = trace.get_current_span().get_span_context()
        trace_id = format(ctx.trace_id, "032x") if ctx.is_valid else ""
    line = {"time": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), "svc": service, "event": event,
            "trace_id": trace_id, **fields}
    print(json.dumps(line, default=str), file=sys.stdout, flush=True)
