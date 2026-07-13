FROM python:3.10-slim

WORKDIR /app

# Sistema kutubxonalari (Pillow uchun kerak bo'lishi mumkin)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libjpeg-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render.com PORT muhit o'zgaruvchisini o'zi beradi (odatda 10000)
EXPOSE 10000

# Shell shaklida CMD, chunki $PORT muhit o'zgaruvchisini o'qish kerak
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
