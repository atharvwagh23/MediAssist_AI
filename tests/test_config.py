"""Config test: print every setting without revealing secrets."""

from app import config  # our central settings file

SECRETS = {  # settings that must never be printed
    "GROQ_API_KEY": config.GROQ_API_KEY,
    "OPENAI_API_KEY": config.OPENAI_API_KEY,
    "DB_PASSWORD": config.DB_PASSWORD,
    "LANGSMITH_API_KEY": __import__("os").getenv("LANGSMITH_API_KEY"),
}

SETTINGS = {  # safe settings to show as-is
    "LLM_PROVIDER": config.LLM_PROVIDER,
    "GROQ_MODEL": config.GROQ_MODEL,
    "JUDGE_MODEL": config.JUDGE_MODEL,
    "DB_HOST": config.DB_HOST,
    "DB_PORT": config.DB_PORT,
    "DB_NAME": config.DB_NAME,
    "DB_USER": config.DB_USER,
    "DB_SSLMODE": config.DB_SSLMODE,
    "EMBEDDING_MODEL": config.EMBEDDING_MODEL,
    "CHUNK_SIZE": config.CHUNK_SIZE,
    "CHUNK_OVERLAP": config.CHUNK_OVERLAP,
    "TOP_K": config.TOP_K,
    "SIMILARITY_THRESHOLD": config.SIMILARITY_THRESHOLD,
    "MAX_UPLOAD_MB": config.MAX_UPLOAD_MB,
    "ALLOWED_DOC_TYPES": config.ALLOWED_DOC_TYPES,
    "ALLOWED_IMAGE_TYPES": config.ALLOWED_IMAGE_TYPES,
    "MAX_HISTORY": config.MAX_HISTORY,
    "BACKEND_URL": config.BACKEND_URL,
    "VECTOR_STORE_DIR": config.VECTOR_STORE_DIR,
}

print(f"{'SETTING':<24}VALUE")  # table header
print("-" * 60)  # divider line
for name, value in SETTINGS.items():  # one row per safe setting
    print(f"{name:<24}{value}")

print("\n" + f"{'SECRET':<24}STATUS")  # second table header
print("-" * 60)
for name, value in SECRETS.items():  # show only SET or MISSING
    print(f"{name:<24}{'SET' if value else 'MISSING'}")

raw_files = sorted(p.name for p in config.RAW_DIR.glob("*.pdf"))  # PDFs found in data/raw
print(f"\nPDFs in data/raw: {len(raw_files)}")  # should be 5
for name in raw_files:
    print(f"  - {name}")