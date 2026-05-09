FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY llm-gateway/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY llm-gateway/ .

EXPOSE 8002

CMD ["python", "main.py"]
