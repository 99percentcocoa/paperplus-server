# Build context is paperplus-v2/ (the monorepo root), not vision-service/ itself -- see
# infra/docker-compose.yml's `context: ..` -- so this Dockerfile can also COPY the sibling
# shared/ package (imported at runtime as `shared.contracts`/`shared.logging_config`), which
# would otherwise be physically outside a context scoped to vision-service/ alone.
FROM python:3.11-slim

WORKDIR /app

COPY vision-service/requirements.txt .
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir paddlepaddle==3.3.1 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
RUN pip install --no-cache-dir paddleocr==3.7.0
# trigger model download during build time to avoid download at runtime
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(text_detection_model_name='PP-OCRv6_small_det', text_recognition_model_name='PP-OCRv5_mobile_rec', use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False, enable_mkldnn=False)"
RUN pip install --no-cache-dir -r requirements.txt

COPY vision-service/ .
COPY shared/ ./shared

EXPOSE 8100
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8100"]
