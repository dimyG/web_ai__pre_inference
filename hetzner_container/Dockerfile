# Production image for the pre_inference service (FastAPI). No entrypoint script: there are
# no migrations and uvicorn is started directly.
FROM python:3.10.10-bullseye

WORKDIR /usr/src/pre_inference_src

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY ./ ./

EXPOSE 8002

# No --reload: WATCHFILES_FORCE_POLLING in the dev compose exists only for hot reload on Windows.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002", "--timeout-keep-alive", "60"]
