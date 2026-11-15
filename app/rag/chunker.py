"""
Chunking: cut long pages into small overlapping pieces.

Why: an embedding model works best on short, focused text, and the chatbot
only needs the few passages that answer the question.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter  # smart text splitter

from app import config  # CHUNK_SIZE and CHUNK_OVERLAP


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Split each page into chunks and keep the source file and page number."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,  # max characters per chunk
        chunk_overlap=config.CHUNK_OVERLAP,  # characters repeated between neighbours
        separators=["\n\n", "\n", ". ", " ", ""],  # try to split at paragraphs first
    )
    chunks = []
    for page in pages:
        for number, piece in enumerate(splitter.split_text(page["text"]), start=1):
            piece = piece.strip()
            if not piece:  # ignore empty pieces
                continue
            chunks.append({
                "chunk_id": f"{page['source']}::p{page['page']}::c{number}",  # unique name
                "text": piece,
                "source": page["source"],  # file name (shown as the citation)
                "page": page["page"],
            })
    return chunks
