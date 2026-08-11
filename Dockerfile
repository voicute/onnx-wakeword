FROM python:3.10-slim

WORKDIR /app

COPY python/ ./python/
COPY wyoming/ ./wyoming/
COPY models/melspectrogram.onnx ./models/

RUN pip install --no-cache-dir onnxruntime numpy

LABEL org.opencontainers.image.source="https://github.com/voicute/onnx-wakeword" \
      org.opencontainers.image.description="Custom wake word detection — Wyoming protocol service for Home Assistant" \
      org.opencontainers.image.licenses="MIT"

EXPOSE 10400
ENTRYPOINT ["python", "wyoming/wyoming_voicute.py"]
CMD ["--help"]
