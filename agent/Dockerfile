FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --only-group agent --no-install-project
COPY agent ./agent
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8010
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8010/health')"
CMD ["uvicorn","agent.api:app","--host","0.0.0.0","--port","8010"]
