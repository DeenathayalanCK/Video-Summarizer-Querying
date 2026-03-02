FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLO model weights at build time so first run doesn't
# fetch them from the internet inside the container.
# Uses CPU device explicitly — this is a CPU-only build.
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')" || true

COPY . .

CMD ["python", "main.py"]