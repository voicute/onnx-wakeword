FROM python:3.10-slim

WORKDIR /app

COPY python/ ./python/
COPY wyoming/ ./wyoming/
COPY models/melspectrogram.onnx ./models/

RUN pip install --no-cache-dir onnxruntime numpy

EXPOSE 10400
ENTRYPOINT ["python", "wyoming/wyoming_voicute.py"]
CMD ["--help"]
