FROM python:3.14-slim AS builder

WORKDIR /app

COPY config/requirements.lock ./config/

RUN pip install --no-cache-dir --prefix=/install -r ./config/requirements.lock

FROM python:3.14-slim

WORKDIR /app

RUN apt-get update && apt-get upgrade -y && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
RUN pip install --no-cache-dir --upgrade setuptools

COPY src ./src
COPY data/sales_data.csv ./data/sales_data.csv

ENV PYTHONPATH=/app/src

RUN python src/main.py

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
