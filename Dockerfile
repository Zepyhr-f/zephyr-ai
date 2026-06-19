ARG PYTHON_IMAGE=python:3.11-slim
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG PIP_INDEX_URL
ARG PIP_TRUSTED_HOST

WORKDIR /app

RUN if [ -n "${PIP_INDEX_URL:-}" ]; then pip config set global.index-url "${PIP_INDEX_URL}"; fi \
    && if [ -n "${PIP_TRUSTED_HOST:-}" ]; then pip config set global.trusted-host "${PIP_TRUSTED_HOST}"; fi \
    && pip install --no-cache-dir \
        alembic==1.14.1 \
        asyncpg==0.30.0 \
        fastapi==0.115.14 \
        greenlet==3.2.4 \
        httpx==0.27.2 \
        pgvector==0.3.6 \
        pydantic-settings==2.6.1 \
        pytest==8.3.5 \
        pytest-asyncio==0.24.0 \
        pytest-cov==6.0.0 \
        sqlalchemy==2.0.45 \
        uvicorn[standard]==0.32.1

COPY app ./app
COPY cli ./cli
COPY migrations ./migrations
COPY alembic.ini ./alembic.ini

EXPOSE 8000

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
