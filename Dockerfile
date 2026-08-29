FROM python:3.12-slim AS runtime

ARG DEFECTDOCK_INSTALL_TRAIN=0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEFECTDOCK_WORKSPACE=/data

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE THIRD_PARTY_NOTICES.md ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && if [ "$DEFECTDOCK_INSTALL_TRAIN" = "1" ]; then \
         python -m pip install ".[train]"; \
       else \
         python -m pip install "."; \
       fi

RUN useradd --create-home --uid 10001 defectdock \
    && mkdir -p /data/.defectdock /data/datasets /data/outputs \
    && chown -R defectdock:defectdock /app /data

USER defectdock
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["uvicorn", "defectdock.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
