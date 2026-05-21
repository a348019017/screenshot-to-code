FROM python:3.12.3-slim-bullseye

ENV POETRY_VERSION=1.8.0 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app
COPY poetry.lock pyproject.toml /app/

RUN pip install "poetry==$POETRY_VERSION" && \
    poetry install --only main --no-root && \
    pip uninstall -y poetry && \
    rm -rf /root/.cache/pip /root/.cache/pypoetry /tmp/*

COPY ./ /app/

EXPOSE 7001
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7001"]
