"""
search_basic.py: meaning search (vector search), the Module 4 baseline.

Module 5 technique: this is BASIC RAG, the starting point we measure everything against.
The question is turned into 1536 numbers (an embedding), and Qdrant returns the product
pages whose numbers are closest. This is good at MEANING ("something for the rain")
but weak at exact CODES (TS-4412 and TS-4414 look almost the same to an embedding).

It also has search_reviews(), a meaning search on the "reviews" collection,
used later in Part 7 (multi-source search).

How to run (run python -m src.build_index first):
    python -m src.search_basic
"""

from src import config
from src.build_index import close_qdrant_client, open_vector_store


def search_products_basic(question, how_many, qdrant_filter=None):
    """Meaning search on product pages. Returns a list like [{"code": "TS-4412", "score": 0.61}, ...]."""
    vector_store = open_vector_store(config.PRODUCTS_COLLECTION)
    # similarity_search_with_score embeds the question, asks Qdrant for the closest
    # points (only those that pass qdrant_filter, if we give one) and returns
    # (document, score) pairs, best first. Score = cosine similarity (higher = closer).
    found = vector_store.similarity_search_with_score(question, k=how_many, filter=qdrant_filter)
    results = []
    for document, score in found:
        results.append({"code": document.metadata["code"], "score": round(score, 4)})
    return results


def search_reviews(question, how_many):
    """Meaning search on reviews. Returns the matching reviews, best first, with their product code."""
    vector_store = open_vector_store(config.REVIEWS_COLLECTION)
    found = vector_store.similarity_search_with_score(question, k=how_many)
    results = []
    for document, score in found:
        results.append({
            "code": document.metadata["code"],
            "review_id": document.metadata["review_id"],
            "score": round(score, 4),
            "text": document.page_content,
        })
    return results


def codes_from_results(results):
    """Take a list of results and return only the product codes, best first, each code once."""
    codes = []
    for result in results:
        if result["code"] not in codes:
            codes.append(result["code"])
    return codes


# Small demo: show where basic meaning search works and where it fails.
if __name__ == "__main__":
    print("Step 1: a MEANING question (meaning search is good at this)")
    for result in search_products_basic("shoes for snow treks in the mountains", 5):
        print("  ", result["code"], result["score"])

    print("\nStep 2: an exact CODE question (meaning search is weak at this)")
    print("   We want HK-3320. Look where it lands:")
    for result in search_products_basic("Tell me about HK-3320", 5):
        print("  ", result["code"], result["score"])

    print("\nStep 3: a review search (reviews carry their product in a header)")
    for result in search_reviews("kept my feet dry in the monsoon", 5):
        print("  ", result["score"], result["text"])

    # Always close the local Qdrant folder at the end.
    close_qdrant_client()
