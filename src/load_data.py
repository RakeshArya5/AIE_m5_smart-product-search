"""
load_data.py: read products.csv and reviews.csv and turn them into texts we can search.

Module 5 techniques taught here (Part 6):
- "Product page": all fields of one product joined into one readable paragraph.
  One product = one search chunk. Products are short, so no chunk splitting is needed.
- Contextual chunk headers: every review gets a header saying WHICH product it is about,
  like "[Product: TrailRunner X | TS-4412 | trail running shoe | Review] Stayed dry...".
  Without the header, the review "Stayed dry in the monsoon" does not say which shoe it means.
- Parent-child idea: each review is a small "child" linked to its "parent" product by `code`.
  We search the small children, then show the full parent product.

How to run:
    python -m src.load_data
"""

import pandas as pd

from src import config


def load_products():
    """Read products.csv and return a list of product dictionaries with clean types."""
    table = pd.read_csv(config.PRODUCTS_CSV)
    products = []
    # Go through the rows and convert each value to the right Python type,
    # because we need numbers for filters (price <= 5000) and a list for sizes.
    for row in table.to_dict("records"):
        product = {
            "code": str(row["code"]),
            "name": str(row["name"]),
            "brand": str(row["brand"]),
            "category": str(row["category"]),
            "gender": str(row["gender"]),
            "price": int(row["price"]),
            "sizes": sizes_text_to_list(str(row["sizes"])),
            "waterproof": str(row["waterproof"]).lower() == "yes",
            "colour": str(row["colour"]),
            "rating": float(row["rating"]),
            "review_count": int(row["review_count"]),
            "description": str(row["description"]),
        }
        products.append(product)
    return products


def sizes_text_to_list(sizes_text):
    """Turn the text "7|8|9|10" into the list [7, 8, 9, 10]."""
    sizes = []
    for size_text in sizes_text.split("|"):
        sizes.append(int(size_text))
    return sizes


def load_reviews():
    """Read reviews.csv and return a list of review dictionaries."""
    table = pd.read_csv(config.REVIEWS_CSV)
    reviews = []
    for row in table.to_dict("records"):
        review = {
            "review_id": str(row["review_id"]),
            "code": str(row["code"]),
            "reviewer": str(row["reviewer"]),
            "rating": int(row["rating"]),
            "text": str(row["text"]),
        }
        reviews.append(review)
    return reviews


def make_product_page(product):
    """Join all fields of one product into one readable paragraph (the "product page")."""
    # Turn the list of sizes back into readable text, like "7, 8, 9, 10".
    size_words = []
    for size in product["sizes"]:
        size_words.append(str(size))
    sizes_text = ", ".join(size_words)

    waterproof_text = "yes" if product["waterproof"] else "no"

    page = (
        f"{product['name']} (code {product['code']}) is a {product['category']} "
        f"by {product['brand']} for {product['gender']}. "
        f"Price: Rs {product['price']}. Sizes: {sizes_text}. "
        f"Waterproof: {waterproof_text}. Colour: {product['colour']}. "
        f"Rating: {product['rating']} out of 5 from {product['review_count']} reviews. "
        f"{product['description']}"
    )
    return page


def make_review_text(review, product):
    """Put a contextual header in front of a review, so the review says which product it is about."""
    header = (
        f"[Product: {product['name']} | {product['code']} | "
        f"{product['category']} | Review]"
    )
    return header + " " + review["text"]


def products_by_code(products):
    """Return a dictionary code -> product, so we can find a product by its code quickly."""
    lookup = {}
    for product in products:
        lookup[product["code"]] = product
    return lookup


def reviews_by_code(reviews):
    """Return a dictionary code -> list of reviews (the "children" of each parent product)."""
    lookup = {}
    for review in reviews:
        if review["code"] not in lookup:
            lookup[review["code"]] = []
        lookup[review["code"]].append(review)
    return lookup


# Small demo: load the data and print one product page and one review with its header.
if __name__ == "__main__":
    print("Step 1: reading the CSV files")
    all_products = load_products()
    all_reviews = load_reviews()
    print("  Products loaded:", len(all_products))
    print("  Reviews loaded:", len(all_reviews))

    print("\nStep 2: one product as a 'product page' (one search chunk)")
    first_product = all_products[0]
    print(" ", make_product_page(first_product))

    print("\nStep 3: one review WITH its contextual header")
    product_lookup = products_by_code(all_products)
    first_review = all_reviews[0]
    print(" ", make_review_text(first_review, product_lookup[first_review["code"]]))

    print("\nStep 4: parent-child: the reviews (children) of", first_product["code"])
    review_lookup = reviews_by_code(all_reviews)
    for child_review in review_lookup[first_product["code"]]:
        print("   -", child_review["review_id"], child_review["text"])
