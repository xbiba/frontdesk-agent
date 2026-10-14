# Base image and uv are pinned by version and digest; Renovate keeps both current.
FROM python:3.12.15-slim-trixie@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
COPY --from=ghcr.io/astral-sh/uv:0.11.26@sha256:3d868e555f8f1dbc324afa005066cd11e1053fc4743b9808ca8025283e65efa5 /uv /uvx /bin/
WORKDIR /app
# UV_NO_CACHE: uv's download cache would stay in a layer (about 200 MB) and is not read at run time.
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_NO_CACHE=1 PYTHONUNBUFFERED=1
# The extras a deployment needs: `demo` brings the local fastembed embedder some packs were
# evaluated with (Clinica MCU's). CI builds the image without any.
ARG EXTRAS=""
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project ${EXTRAS:+--extra ${EXTRAS}}
COPY . .
# Editable install: alembic.ini, migrations/ and agents/ are read from /app at runtime.
RUN uv sync --locked --no-dev ${EXTRAS:+--extra ${EXTRAS}}
ENV PATH="/app/.venv/bin:$PATH"
USER 10001:10001
EXPOSE 8080
CMD ["uvicorn", "frontdesk.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
