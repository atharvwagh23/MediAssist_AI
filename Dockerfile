# MediAssist AI: one container runs FastAPI (internal, port 8000) and Streamlit (public, $PORT).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=7860

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
RUN python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']; open('requirements.txt','w').write('\n'.join(x for x in d if not x.startswith(('torch','torchvision'))))" \
    && pip install -r requirements.txt

RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)" \
    && python -c "from transformers import BlipProcessor, BlipForConditionalGeneration as B; n='Salesforce/blip-image-captioning-base'; BlipProcessor.from_pretrained(n); B.from_pretrained(n)" \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY . .
RUN python -m app.rag.ingest && chmod +x start.sh

EXPOSE 7860
CMD ["./start.sh"]