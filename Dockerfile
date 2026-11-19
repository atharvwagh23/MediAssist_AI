# MediAssist AI: one container runs FastAPI (internal, port 8000) and Streamlit (public, $PORT).
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=7860

WORKDIR /app

# libraries OpenCV (used by EasyOCR) needs
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# CPU-only PyTorch + matching torchvision (installed together so versions match)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# install the rest of the dependencies straight from pyproject.toml
COPY pyproject.toml ./
RUN python -c "import tomllib; d=tomllib.load(open('pyproject.toml','rb'))['project']['dependencies']; open('requirements.txt','w').write('\n'.join(x for x in d if not x.startswith(('torch','torchvision'))))" \
    && pip install -r requirements.txt

COPY . .

# bake the models and the knowledge-base index into the image
RUN python -c "import easyocr; easyocr.Reader(['en'], gpu=False)" \
    && python -m app.rag.ingest

# Hugging Face runs the container as a non-root user: let it write users.json and uploads
RUN chmod -R a+rwX /app && chmod +x start.sh
EXPOSE 7860
CMD ["./start.sh"]