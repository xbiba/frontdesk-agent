FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
WORKDIR /app
# UV_NO_CACHE: uv's download cache would stay in a layer (about 200 MB) and is not read at run time.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY . .
# Editable install: alembic.ini, migrations/ and agents/ are read from /app at runtime.
RUN uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:$PATH"
USER 10001
EXPOSE 8080
CMD ["uvicorn", "frontdesk.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
