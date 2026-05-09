FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY build-service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY build-service/ .

EXPOSE 8004

CMD ["python", "main.py"]
