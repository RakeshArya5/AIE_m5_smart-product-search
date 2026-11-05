"""
search_hybrid.py: hybrid search = keyword search (BM25) + meaning search, merged with RRF.

Module 5 technique (Part 3): hybrid search.
- BM25 is classic KEYWORD search: it counts matching words, and rare words (like a code
  "ts-4412") count a lot. It is great at codes and names, but knows nothing about meaning.
- Meaning search (src/search_basic.py) is great at meaning, but weak at codes.
- Reciprocal Rank Fusion (RRF) merges the two ranked lists. Each list gives every product
  1 / (RRF_K + rank) points, and we add the points up. A product near the top of BOTH
  lists wins. RRF only uses the RANK, so we never have to compare BM25 scores with cosine scores.

How to run (run python -m src.build_index first):
    python -m src.search_hybrid
"""

import re

from rank_bm25 import BM25Okapi

from src import config
from src.build_index import close_qdrant_client
from src.load_data import load_products, make_product_page
from src.search_basic import codes_from_results, search_products_basic


def split_into_words(text):
    """Split text into lowercase words, keeping codes like "ts-4412" and "ts-4412k" as ONE word."""
    # [a-z0-9]+ is a run of letters or digits, and (-[a-z0-9]+)* lets it continue after a
    # dash. So "TS-4412" becomes "ts-4412" and not the two words "ts" and "4412".
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def keyword_search(question, how_many, allowed_codes=None):
    """BM25 keyword search on the product pages. Returns [{"code": ..., "score": ...}, ...]."""
    products = load_products()
    # BM25Okapi needs every product page as a list of words. With 120 products this
    # takes a few milliseconds, so we simply build it again for each search.
    page_words = []
    for product in products:
        page_words.append(split_into_words(make_product_page(product)))
    bm25 = BM25Okapi(page_words)
    scores = bm25.get_scores(split_into_words(question))

    # Pair each product with its score, skip products that share no word with the
    # question (score 0) or that the filters do not allow, then sort best first.
    results = []
    for position in range(len(products)):
        code = products[position]["code"]
        if scores[position] <= 0:
            continue
        if allowed_codes is not None and code not in allowed_codes:
            continue
        results.append({"code": code, "score": round(float(scores[position]), 4)})
    results.sort(key=get_score, reverse=True)
    return results[:how_many]


def get_score(result):
    """Return the score of one result (used to sort results from best to worst)."""
    return result["score"]


def rrf_merge(code_lists):
    """Merge several ranked lists of codes with Reciprocal Rank Fusion. Best first."""
    points = {}
    # Every list gives each code 1 / (RRF_K + rank) points. Rank 1 = best.
    for code_list in code_lists:
        for position in range(len(code_list)):
            rank = position + 1
            code = code_list[position]
            points[code] = points.get(code, 0) + 1 / (config.RRF_K + rank)

    results = []
    for code in points:
        results.append({"code": code, "score": round(points[code], 5)})
    results.sort(key=get_score, reverse=True)
    return results


def hybrid_search(question, how_many, qdrant_filter=None, allowed_codes=None, meaning_text=None):
    """Run keyword search AND meaning search, and merge them with RRF.

    qdrant_filter and allowed_codes are the same filter rules in two forms (see filters.py):
    Qdrant uses qdrant_filter, and BM25 keeps only the codes in allowed_codes.
    meaning_text lets the meaning search use a different text (used by HyDE).
    """
    if meaning_text is None:
        meaning_text = question
    keyword_codes = codes_from_results(keyword_search(question, how_many, allowed_codes))
    meaning_codes = codes_from_results(search_products_basic(meaning_text, how_many, qdrant_filter))
    merged = rrf_merge([keyword_codes, meaning_codes])
    return merged[:how_many]


# Small demo: the code question that basic search got wrong.
if __name__ == "__main__":
    question = "Tell me about HK-3320"
    print("Question:", question)
    print("Step 1: how the question is split into words:", split_into_words(question))

    print("\nStep 2: keyword search (BM25): the code is a rare word, so it wins")
    for result in keyword_search(question, 5):
        print("  ", result["code"], result["score"])

    print("\nStep 3: meaning search: HK-3320 is not at the top")
    for result in search_products_basic(question, 5):
        print("  ", result["code"], result["score"])

    print("\nStep 4: hybrid = both lists merged with RRF (score = 1/(60+rank), added up)")
    for result in hybrid_search(question, 5):
        print("  ", result["code"], result["score"])

    # Always close the local Qdrant folder at the end.
    close_qdrant_client()
