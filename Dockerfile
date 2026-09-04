FROM python:3.14-slim@sha256:cad9a2c871761c413caa6fdd6441c783451e740a48aaeba60ae62a8b53525ef6 AS runtime

ARG DEFECTDOCK_INSTALL_TRAIN=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    DEFECTDOCK_WORKSPACE=/data \
    DEFECTDOCK_SECURITY_MODE=network

WORKDIR /app

RUN apt-get -o Acquire::Retries=5 update \
    && apt-get -o Acquire::Retries=5 upgrade --no-install-recommends -y \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN python -m pip install --no-cache-dir uv==0.11.13 \
    && if [ "$DEFECTDOCK_INSTALL_TRAIN" = "1" ]; then \
         uv sync --locked --no-dev --no-install-project --extra train --extra export; \
       else \
         uv sync --locked --no-dev --no-install-project; \
       fi \
    && uv cache clean

COPY README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
RUN if [ "$DEFECTDOCK_INSTALL_TRAIN" = "1" ]; then \
      uv sync --locked --no-dev --no-editable --extra train --extra export; \
    else \
      uv sync --locked --no-dev --no-editable; \
    fi \
    && uv cache clean \
    && python -m pip uninstall --yes uv

ENV PATH="/app/.venv/bin:$PATH"

ARG DEFECTDOCK_BUILD_REVISION=unknown
ENV DEFECTDOCK_BUILD_REVISION=$DEFECTDOCK_BUILD_REVISION
LABEL org.opencontainers.image.revision=$DEFECTDOCK_BUILD_REVISION

RUN useradd --create-home --uid 10001 defectdock \
    && mkdir -p /data/.defectdock /data/datasets /data/outputs \
    && chown -R defectdock:defectdock /data

USER defectdock
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["uvicorn", "defectdock.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
