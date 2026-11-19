#!/bin/sh
# Backend stays private on localhost:8000. Streamlit listens on the port Render gives us.# Backend stays private on localhost:8000. Streamlit listens on $PORT (7860 on Hugging Face).set -e
export BACKEND_URL="http://localhost:8000"

uvicorn backend.main:app --host 127.0.0.1 --port 8000 &

exec streamlit run frontend/app.py \
    --server.port "${PORT:-8501}" \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false
