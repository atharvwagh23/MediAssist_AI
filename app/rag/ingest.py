"""
Build the knowledge base from every document in data/raw/.

Run from the project root:
    python -m app.rag.ingest
"""

from app import config
from app.rag.vector_store import add_file, list_sources, reset_store


def main() -> None:
    files = sorted(
        p for p in config.RAW_DIR.iterdir()
        if p.is_file() and p.suffix.lower().lstrip(".") in config.ALLOWED_DOC_TYPES
    )
    print(f"Found {len(files)} documents in {config.RAW_DIR}")
    print(f"Chunk size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP}, model={config.EMBEDDING_MODEL}\n")

    reset_store()  # start from an empty index
    rows = []
    for file in files:
        try:
            info = add_file(file)
            rows.append((file.name, info["pages"], info["chunks"], "OK"))
        except Exception as exc:  # one bad file must not stop the others
            rows.append((file.name, 0, 0, f"FAILED: {exc}"))

    print(f"{'FILE':<38}{'PAGES':>6}{'CHUNKS':>8}  STATUS")
    print("-" * 70)
    for name, pages, chunks, status in rows:
        print(f"{name:<38}{pages:>6}{chunks:>8}  {status}")
    print("-" * 70)
    print(f"{'TOTAL':<38}{sum(r[1] for r in rows):>6}{sum(r[2] for r in rows):>8}")
    print("\nIndexed sources:", list_sources())


if __name__ == "__main__":
    main()