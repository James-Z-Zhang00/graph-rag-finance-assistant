FROM python:3.10-slim

WORKDIR /app

# System dependencies for PDF/image parsing libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libgl1 \
        libglib2.0-0 \
        ghostscript \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer cache)
COPY sec-parser/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service source
COPY sec-parser/ .

EXPOSE 8001

CMD ["python", "main.py"]
