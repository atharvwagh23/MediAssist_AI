"""
Retrieval test: ask sample questions and show which chunk answers each one.

Run from the project root (after python -m app.rag.ingest):
    python -m tests.test_rag
"""

from app import config
from app.rag.retriever import retrieve

QUESTIONS = [
    "What is the admission process?",
    "How does cashless claim pre-authorization work?",
    "What are the escalation levels for emergencies?",
    "Show the discharge summary for the diabetes patient",
    "What do NABH compliance guidelines require?",
    "How are refunds handled?",           # may not exist in the documents
    "What is the capital of France?",     # NOT medical: should find nothing
]

print(f"TOP_K={config.TOP_K}  SIMILARITY_THRESHOLD={config.SIMILARITY_THRESHOLD}\n")
print(f"{'QUESTION':<52}{'FOUND':<7}{'TOP SOURCE':<34}{'PG':>3}{'VEC':>7}{'FINAL':>7}")
print("-" * 110)

for question in QUESTIONS:
    result = retrieve(question)  # run the full retrieval pipeline
    if result["found"]:
        top = result["results"][0]  # best chunk
        print(f"{question:<52}{'yes':<7}{top['source']:<34}{top['page']:>3}"
              f"{top['vector_score']:>7.2f}{top['final_score']:>7.2f}")
    else:
        print(f"{question:<52}{'NO':<7}{'-':<34}{'-':>3}{'-':>7}{'-':>7}")

print("\nDetail for the first question (all returned chunks):")
detail = retrieve(QUESTIONS[0])
for item in detail["results"]:
    print(f"  #{item['rank']} {item['source']} p{item['page']}  "
          f"vector={item['vector_score']}  keyword={item['keyword_score']}  final={item['final_score']}")
    print(f"     {item['text'][:110].replace(chr(10), ' ')}...")