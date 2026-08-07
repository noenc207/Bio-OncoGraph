FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-cpu.txt ./
RUN pip install --no-cache-dir -r requirements-cpu.txt

COPY . .

CMD ["python", "-m", "bio_oncograph"]
