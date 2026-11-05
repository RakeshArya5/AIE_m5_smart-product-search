"""
answer.py: turn the products we found into a helpful answer for the shopper.

Module 5 technique: the "G" in RAG (generation), done safely.
1. The finder (src/finder.py) finds the product codes.
2. We collect each product's full details (the "parent") and its reviews (the "children").
3. The LLM writes a recommendation using ONLY those products and reviews,
   with price, sizes and a reason quoted from a review.
4. The list of product codes under the answer is added by OUR code, not by the LLM,
   so it is always correct and can never be made up.

How to run (run python -m src.build_index first):
    python -m src.answer
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src import config
from src.build_index import close_qdrant_client
from src.finder import find_products
from src.load_data import load_products, load_reviews, make_product_page, products_by_code, reviews_by_code

MAX_REVIEWS_PER_PRODUCT = 5

ANSWER_PROMPT = """You are the friendly shopping assistant of TrailMart, an online shoe store.
Answer the shopper's question using ONLY the products and reviews below.

Rules:
- Recommend the products that really match the question. For each one give the name, code,
  price (Rs), sizes, and a short reason QUOTED from a review (in "quotes").
- If a product does not match (wrong price, size, type...), do not recommend it.
- If NOTHING below matches, say clearly that this item is not available at TrailMart.
- Never invent products, prices, sizes or reviews.
- If the question asks "how many", give the exact number from the database note,
  never a number you counted yourself. A flip-flop counts as a sandal.
- Keep it short: at most 5 products, a few lines each.

Question: {question}

{database_note}
Products and their reviews:
{products_text}"""


def collect_products(codes):
    """Return the full product dictionaries for the codes, each with its reviews added."""
    product_lookup = products_by_code(load_products())
    review_lookup = reviews_by_code(load_reviews())
    products = []
    for code in codes:
        # Copy the product, so we do not change the shared dictionary.
        product = dict(product_lookup[code])
        # Best-rated reviews first, so the "top reviews" are the most useful ones.
        product_reviews = sorted(review_lookup.get(code, []), key=get_review_rating, reverse=True)
        product["reviews"] = product_reviews[:MAX_REVIEWS_PER_PRODUCT]
        products.append(product)
    return products


def get_review_rating(review):
    """Return the rating of one review (used to sort reviews from best to worst)."""
    return review["rating"]


def make_products_text(products):
    """Write every product page with its reviews, as one block of text for the LLM."""
    if len(products) == 0:
        return "(no products found)"
    blocks = []
    for product in products:
        block = make_product_page(product)
        for review in product["reviews"]:
            block = block + f'\n   Review ({review["rating"]}/5): "{review["text"]}"'
        blocks.append(block)
    return "\n\n".join(blocks)


def make_database_note(search_details):
    """If SQL was used, tell the LLM how many rows the database found (for "how many")."""
    if search_details["sql_rows"] is None:
        return ""
    row_count = len(search_details["sql_rows"]["rows"])
    # We show only the first TOP_K products, so we tell the LLM the TRUE count here.
    # Otherwise it counts the products it can see and gives the wrong number.
    return (
        f"Database note: the database found exactly {row_count} matching products "
        f"(only the first {config.TOP_K} are listed below). "
        f"If the question asks how many, the answer is {row_count}.\n"
    )


def make_codes_line(products):
    """The product code list under the answer, written by our code (never by the LLM)."""
    if len(products) == 0:
        return "Products checked by the search: none"
    code_texts = []
    for product in products:
        code_texts.append(product["code"] + " (" + product["name"] + ")")
    return "Products checked by the search: " + ", ".join(code_texts)


def answer_question(question, settings):
    """Find products, then ask the LLM to recommend them. Returns a dictionary with everything."""
    search_details = find_products(question, settings)
    products = collect_products(search_details["codes"])

    # prompt -> LLM -> StrOutputParser (which just takes the reply text out of the message)
    prompt = ChatPromptTemplate.from_template(ANSWER_PROMPT)
    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)
    chain = prompt | llm | StrOutputParser()
    answer_text = chain.invoke({
        "question": question,
        "database_note": make_database_note(search_details),
        "products_text": make_products_text(products),
    })
    return {
        "question": question,
        "answer": answer_text.strip() + "\n\n" + make_codes_line(products),
        "products": products,
        "search": search_details,
    }


# Small demo: answer one normal question and one "not in catalogue" question.
if __name__ == "__main__":
    demo_settings = config.all_switches_on()
    for demo_question in ["Waterproof trail shoes under Rs 5,000, size 9", "Do you sell laptop bags?"]:
        print("Question:", demo_question)
        print("Step 1: find products, Step 2: collect details + reviews, Step 3: ask the LLM\n")
        result = answer_question(demo_question, demo_settings)
        print(result["answer"])
        print("\n" + "-" * 70 + "\n")

    # Always close the local Qdrant folder at the end.
    close_qdrant_client()
