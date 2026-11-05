"""
build_index.py: build everything the finder searches in.

It builds three things, deleting the old ones first (so we never get duplicates):
1. Qdrant collection "products": one point per product page, with metadata we can filter on.
2. Qdrant collection "reviews": one point per review (with its contextual header).
3. SQLite database data/shop.db: a "products" table for questions like "cheapest" or "how many".

Module 5 techniques taught here:
- Metadata payloads (price, sizes, waterproof...) that make exact filters possible (Part 3).
- Embedding cache (Part 8): the same text is never embedded (and paid for) twice.

This file also holds three small helpers that the search files use:
get_embeddings(), get_qdrant_client() and close_qdrant_client().

How to run:
    python -m src.build_index
"""

import atexit
import sqlite3

from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, Filter, FilterSelector, VectorParams

from src import config
from src.load_data import (
    load_products,
    load_reviews,
    make_product_page,
    make_review_text,
    products_by_code,
)

# Qdrant in local folder mode allows only ONE open connection to qdrant_data/.
# So we open it once, keep it in this variable, and reuse it everywhere.
open_qdrant_client = None

def get_embeddings():
    """Return an embeddings model that remembers (caches) every text it has embedded."""
    # OpenAIEmbeddings sends texts to OpenAI and gets back 1536 numbers per text.
    openai_embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)

    # LocalFileStore saves small files in data/cache/ (one file per embedded text).
    cache_store = LocalFileStore(str(config.CACHE_FOLDER))

    # CacheBackedEmbeddings checks the cache first. Only texts it has never seen
    # are sent to OpenAI. query_embedding_cache=True also caches search questions.
    # The namespace keeps caches of different embedding models apart.
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        openai_embeddings,
        cache_store,
        namespace=config.EMBEDDING_MODEL,
        query_embedding_cache=True,
        key_encoder="sha256",
    )
    return cached_embeddings


def get_qdrant_client():
    """Open the local Qdrant folder once and return the same connection every time."""
    global open_qdrant_client
    if open_qdrant_client is None:
        open_qdrant_client = QdrantClient(path=str(config.QDRANT_FOLDER))
        # When the program stops (for example Ctrl+C on the Streamlit app), close
        # Qdrant cleanly FIRST. Otherwise Python closes it too late, while shutting
        # down, and prints a scary (but harmless) "sys.meta_path is None" message.
        atexit.register(close_qdrant_client)
    return open_qdrant_client


def close_qdrant_client():
    """Close the Qdrant connection, so another program can open qdrant_data/ later."""
    global open_qdrant_client
    if open_qdrant_client is not None:
        open_qdrant_client.close()
        open_qdrant_client = None


def open_vector_store(collection_name):
    """Return a LangChain vector store for one Qdrant collection (for searching or adding)."""
    # QdrantVectorStore is LangChain's wrapper around Qdrant. Behind the scenes it
    # embeds our text, stores it as a "point" with the text and metadata as payload,
    # and turns a search question into a Qdrant vector search.
    vector_store = QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=collection_name,
        embedding=get_embeddings(),
    )
    return vector_store


def make_empty_collection(collection_name):
    """Delete the collection if it exists, then create it again, empty."""
    qdrant_client = get_qdrant_client()
    if qdrant_client.collection_exists(collection_name):
        qdrant_client.delete_collection(collection_name)
    # Cosine distance compares the direction of two vectors: close meaning = high score.
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=config.EMBEDDING_SIZE, distance=Distance.COSINE),
    )
    # On Windows the old storage file can stay behind after delete_collection
    # (the file is still locked), and its old points come back. So we also delete
    # every point that is left: an empty Filter() matches all points.
    qdrant_client.delete(collection_name, points_selector=FilterSelector(filter=Filter()))


def build_products_collection(products):
    """Put one point per product page into the Qdrant collection "products"."""
    make_empty_collection(config.PRODUCTS_COLLECTION)
    documents = []
    # A LangChain Document = the text to embed + a metadata dictionary.
    # The metadata becomes the Qdrant payload, so we can filter on it later.
    for product in products:
        metadata = {
            "code": product["code"],
            "name": product["name"],
            "category": product["category"],
            "gender": product["gender"],
            "price": product["price"],
            "sizes": product["sizes"],
            "waterproof": product["waterproof"],
            "rating": product["rating"],
        }
        documents.append(Document(page_content=make_product_page(product), metadata=metadata))
    open_vector_store(config.PRODUCTS_COLLECTION).add_documents(documents)
    return len(documents)


def build_reviews_collection(products, reviews):
    """Put one point per review (with its contextual header) into the collection "reviews"."""
    make_empty_collection(config.REVIEWS_COLLECTION)
    product_lookup = products_by_code(products)
    documents = []
    for review in reviews:
        product = product_lookup[review["code"]]
        metadata = {
            "code": review["code"],
            "review_id": review["review_id"],
            "rating": review["rating"],
        }
        documents.append(Document(page_content=make_review_text(review, product), metadata=metadata))
    open_vector_store(config.REVIEWS_COLLECTION).add_documents(documents)
    return len(documents)


def build_sqlite_database(products):
    """Create data/shop.db with one table "products" (deleted and rebuilt every time)."""
    connection = sqlite3.connect(config.SQLITE_FILE)
    connection.execute("DROP TABLE IF EXISTS products")
    connection.execute(
        "CREATE TABLE products (code TEXT, name TEXT, brand TEXT, category TEXT, "
        "gender TEXT, price INTEGER, waterproof INTEGER, rating REAL, "
        "review_count INTEGER, min_size INTEGER, max_size INTEGER)"
    )
    # One row per product. waterproof is stored as 1 (yes) or 0 (no).
    for product in products:
        waterproof_number = 1 if product["waterproof"] else 0
        connection.execute(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (product["code"], product["name"], product["brand"], product["category"],
             product["gender"], product["price"], waterproof_number, product["rating"],
             product["review_count"], min(product["sizes"]), max(product["sizes"])),
        )
    connection.commit()
    connection.close()
    return len(products)


def build_everything():
    """Build the two Qdrant collections and the SQLite database, printing each step."""
    print("Step 1: reading the data")
    products = load_products()
    reviews = load_reviews()
    print("  ", len(products), "products and", len(reviews), "reviews")

    print("\nStep 2: building Qdrant collection 'products' (uses the embedding cache)")
    print("   points added:", build_products_collection(products))

    print("\nStep 3: building Qdrant collection 'reviews' (uses the embedding cache)")
    print("   points added:", build_reviews_collection(products, reviews))

    print("\nStep 4: building SQLite database", config.SQLITE_FILE.name)
    print("   rows added:", build_sqlite_database(products))


# Run the full build, then show a tiny check that everything is there.
if __name__ == "__main__":
    build_everything()

    print("\nStep 5: checking the result")
    client = get_qdrant_client()
    print("   points in 'products':", client.count(config.PRODUCTS_COLLECTION).count)
    print("   points in 'reviews':", client.count(config.REVIEWS_COLLECTION).count)
    print("   files in the embedding cache:", len(list(config.CACHE_FOLDER.iterdir())))

    # Always close the local Qdrant folder at the end, so other programs can open it.
    close_qdrant_client()
    print("\nDone. Run it again: nothing new is paid for, the cache already has every text.")
