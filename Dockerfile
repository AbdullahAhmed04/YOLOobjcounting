FROM python:3.11-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL=yolov8s.pt \
    TTA=0 \
    ON_CLOUD=1 \
    PORT=8000

# libraries OpenCV needs on a slim image
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch keeps the image much smaller than the default (GPU) build
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# download the YOLO weights at build time so the app starts without internet
RUN python -c "from ultralytics import YOLO; YOLO('yolov8s.pt')"

COPY app.py index.html ./

# the app listens on 0.0.0.0 and the PORT value (8000 by default; Cloud Run overrides it)
CMD ["python", "app.py"]
