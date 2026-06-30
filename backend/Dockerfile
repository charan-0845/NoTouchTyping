# --- Base image -------------------------------------------------------
# Start from an official slim Python image. "slim" = smaller download,
# fewer pre-installed system packages than the full image.
FROM python:3.11-slim

# --- System dependencies ------------------------------------------------
# OpenCV and MediaPipe need some system-level libraries (for handling
# images/video) that aren't included in the slim Python image by default.
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ----------------------------------------------------
# All following commands run from /app inside the container.
WORKDIR /app

# --- Install Python dependencies ------------------------------------------
# Copying just requirements.txt first (before the rest of the code) lets
# Docker cache this layer — if your code changes but requirements.txt
# doesn't, Docker skips reinstalling everything, making rebuilds faster.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Copy application code -------------------------------------------------
# This copies main.py, your model file, etc. into the container.
COPY . .

# --- Expose port -------------------------------------------------------
# Documents that the app listens on port 8000 (FastAPI/uvicorn default we'll use).
EXPOSE 8000

# --- Start the server -------------------------------------------------
# --host 0.0.0.0 is required so the server is reachable from outside the
# container (127.0.0.1 would only be reachable from inside it).
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
