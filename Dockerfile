FROM python:3.10-slim

WORKDIR /app

COPY python/ ./python/
COPY wyoming/ ./wyoming/
COPY models/melspectrogram.onnx ./models/
# Bundle the demo keywords so the image (and the HA add-on) works out of the
# box. Must match every model_file referenced by models/model_info.json.
COPY models/model_info.json ./models/
COPY models/en/hey_friday.onnx ./models/en/
COPY models/zh/xiaona_r1.onnx ./models/zh/
COPY models/zh/xiaona_r0.onnx ./models/zh/
COPY models/zh/nihaoxiaona_r1.onnx ./models/zh/
COPY models/zh/nihaoxiaona_r0.onnx ./models/zh/
COPY models/zh/doubaodoubao_r1.onnx ./models/zh/
COPY models/zh/doubaodoubao_r0.onnx ./models/zh/

RUN pip install --no-cache-dir onnxruntime numpy

LABEL org.opencontainers.image.source="https://github.com/voicute/onnx-wakeword" \
      org.opencontainers.image.description="Custom wake word detection — Wyoming protocol service for Home Assistant" \
      org.opencontainers.image.licenses="MIT"

EXPOSE 10400
ENTRYPOINT ["python", "wyoming/wyoming_voicute.py"]
CMD ["--help"]
