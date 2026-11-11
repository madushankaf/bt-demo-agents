# AVA and Client360Agent, built and run by WSO2 Agent Manager (platform-hosted). APP picks the agent:
#   ava | client360-agent
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PORT=8000
WORKDIR /app
COPY pyproject.toml requirements.lock ./
RUN pip install -r requirements.lock
COPY bt ./bt
RUN pip install --no-deps .
ENV APP=ava
# Agent Manager runs agents as a non-root user only.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin agent
USER 10001
EXPOSE 8000
# With Agent Manager's tracing settings present (AMP_OTEL_ENDPOINT), run under its instrumentation wrapper.
CMD ["sh", "-c", "case \"$APP\" in ava) M=bt.agents.ava ;; client360-agent) M=bt.agents.client360_agent ;; \
  *) echo unknown APP $APP; exit 1 ;; esac; \
  if [ -n \"$AMP_OTEL_ENDPOINT\" ] && command -v amp-instrument >/dev/null; then exec amp-instrument python -m $M; \
  else exec python -m $M; fi"]
