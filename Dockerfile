FROM pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/data \
    MODEL_DIR=/app/models \
    SUBMISSION_PATH=/app/submissions/submission.csv

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git curl unzip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["bash"]
