"""
rerank.py: reranking: fetch up to 20 candidates, let the LLM re-score them, keep the best.

Module 5 technique (Part 5): reranking.
Search is fast but rough: it compares the question and each product SEPARATELY
(one embedding each). A reranker reads the question and a product TOGETHER, so it
can notice details like "size 9" or "reviewers say it runs small". That is slower, so we
only rerank a short list of candidates that search already found.

Here we use an "LLM as reranker": we send the question plus up to 20 candidate product
pages (with their reviews) to the LLM in ONE call and ask for a relevance score from
0 to 10 for each code. Then we sort by that score.

A real cross-encoder (for example a small BERT-style reranking model) would do the same
job: it reads (question, product) pairs and outputs a relevance number. It is cheaper and
faster per pair, but it needs extra libraries (like PyTorch), which this project does not use.

How to run:
    python -m src.rerank
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src import config
from src.load_data import load_products, load_reviews, make_product_page, products_by_code, reviews_by_code

RERANK_PROMPT = """You are a careful shoe shop assistant.
Score how well EACH product below answers the shopper's question, from 0 (not useful at all)
to 10 (exactly what they asked for). Check type of shoe, price, sizes, waterproof,
who it is for, and what the reviews say. Products that break a clear wish
(for example a price limit or a size) get a low score.
If the shopper names a specific product (by name or code), that exact product gets 10,
and products with a similar but different name get a lower score.
Reply with JSON only: {{"scores": {{"CODE-1": 7, "CODE-2": 3}}}} with every code listed.

Question: {question}

Products:
{products_text}"""


def make_candidates_text(codes):
    """Write each candidate as its product page plus its reviews, one block per product."""
    product_lookup = products_by_code(load_products())
    review_lookup = reviews_by_code(load_reviews())
    blocks = []
    for code in codes:
        block = "[" + code + "] " + make_product_page(product_lookup[code])
        # Add the reviews under the product, so the reranker can use what customers said.
        for review in review_lookup.get(code, []):
            block = block + "\n   Review: " + review["text"]
        blocks.append(block)
    return "\n\n".join(blocks)


def clean_score(value):
    """Turn the LLM's score into a number from 0 to 10. Anything strange becomes 0."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(10.0, score))


def get_score(result):
    """Return the score of one result (used to sort results from best to worst)."""
    return result["score"]


def rerank(question, codes, how_many):
    """Re-score up to RERANK_CANDIDATES codes with the LLM and return the best how_many."""
    candidate_codes = codes[:config.RERANK_CANDIDATES]
    prompt = ChatPromptTemplate.from_template(RERANK_PROMPT)
    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)
    chain = prompt | llm | JsonOutputParser()
    try:
        reply = chain.invoke({"question": question, "products_text": make_candidates_text(candidate_codes)})
        scores = reply.get("scores", {})
    except Exception as error:
        # If the reranker fails, keep the search order instead of crashing.
        print("   (rerank failed, keeping the search order:", error, ")")
        scores = {}

    # Give every candidate its score (a code the LLM forgot gets 0), then sort.
    # Python's sort is "stable": equal scores keep the order search gave them.
    results = []
    for code in candidate_codes:
        results.append({"code": code, "score": clean_score(scores.get(code))})
    results.sort(key=get_score, reverse=True)
    return results[:how_many]


# Small demo: rerank a rough candidate list for a question with a size wish.
if __name__ == "__main__":
    demo_question = "Which shoes run small according to reviews?"
    demo_codes = ["TS-4412", "TS-4471", "HK-3305", "HK-3370", "WK-5050", "FM-7030", "SN-6114", "RN-2210"]
    print("Question:", demo_question)
    print("Step 1: candidates in the order search found them:", demo_codes)
    print("\nStep 2: the LLM reads the question with every product page + reviews, and scores them")
    for result in rerank(demo_question, demo_codes, 5):
        print("  ", result["code"], "score", result["score"])
    print("\n   Expected near the top: TS-4471, HK-3370, FM-7030, SN-6114 (their reviews say 'runs small').")
