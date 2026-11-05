"""
config.py: ALL the settings of the Smart Product Finder live here, and only here.

Module 5 technique: "measure one change at a time".
Every improvement has its own ON/OFF switch below. All switches start as False,
which means we start with BASIC RAG (like Module 4). Later we turn them on one by one
and measure each one with the scorecard (src/evaluate.py).

Other files never change this file. They call default_settings() to get the switches
as a dictionary, and they may change that dictionary (for example the app sidebar).

How to run:
    python -m src.config
"""

from pathlib import Path

from dotenv import load_dotenv

# Read the OpenAI API key from the .env file into the environment.
# We do this here because every other file imports config first.
load_dotenv()

# ---------------------------------------------------------------
# Paths (folders and files)
# ---------------------------------------------------------------

# The project folder is the folder ABOVE src/ (this file is src/config.py).
PROJECT_FOLDER = Path(__file__).resolve().parent.parent

DATA_FOLDER = PROJECT_FOLDER / "data"
PRODUCTS_CSV = DATA_FOLDER / "raw" / "products.csv"
REVIEWS_CSV = DATA_FOLDER / "raw" / "reviews.csv"
TEST_QUESTIONS_CSV = DATA_FOLDER / "test" / "test_questions.csv"
CACHE_FOLDER = DATA_FOLDER / "cache"
RESULTS_FOLDER = DATA_FOLDER / "results"
SCORECARD_CSV = RESULTS_FOLDER / "scorecard.csv"
SQLITE_FILE = DATA_FOLDER / "shop.db"
QDRANT_FOLDER = PROJECT_FOLDER / "qdrant_data"

# ---------------------------------------------------------------
# Models
# ---------------------------------------------------------------

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_SIZE = 1536  # text-embedding-3-small turns every text into 1536 numbers

# ---------------------------------------------------------------
# Qdrant collection names
# ---------------------------------------------------------------

PRODUCTS_COLLECTION = "products"
REVIEWS_COLLECTION = "reviews"

# ---------------------------------------------------------------
# Search numbers
# ---------------------------------------------------------------

TOP_K = 5                # how many products we finally show (and score with @5)
CANDIDATES = 20          # how many products each search step fetches before we cut to TOP_K
RERANK_CANDIDATES = 20   # how many candidates the reranker is allowed to look at
RRF_K = 60               # the "k" in Reciprocal Rank Fusion: score = 1 / (RRF_K + rank)
REVIEW_TOP_K = 15        # how many reviews the review search fetches
MULTI_QUERY_COUNT = 3    # how many ways we ask the question in multi-query (original included)
SQL_ROW_LIMIT = 20       # the most rows a SQL query may return

# ---------------------------------------------------------------
# The switches: one per improvement. All False = basic RAG.
# ---------------------------------------------------------------

USE_HYBRID = False          # Part 3: BM25 keyword search + meaning search, merged with RRF
USE_FILTERS = False         # Part 3: self-querying: the LLM turns "under 5000, size 9" into exact filters
USE_QUERY_REWRITE = False   # Part 4: the LLM cleans up messy questions before searching
USE_MULTI_QUERY = False     # Part 4: ask the question 3 ways and merge the results
USE_HYDE = False            # Part 4: search with a draft answer instead of the question
USE_RERANK = False          # Part 5: fetch 20 candidates, the LLM re-scores them, keep the best
USE_REVIEW_SEARCH = False   # Part 7: also search the reviews and bring their products in (multi-source)
USE_ROUTER = False          # Part 7: send table questions (cheapest, how many, highest rated) to SQL


def default_settings():
    """Return the switches above as a dictionary, like {"USE_HYBRID": False, ...}.

    Other files pass this dictionary around, so they can turn a switch on or off
    for one run WITHOUT editing this file.
    """
    settings = {
        "USE_HYBRID": USE_HYBRID,
        "USE_FILTERS": USE_FILTERS,
        "USE_QUERY_REWRITE": USE_QUERY_REWRITE,
        "USE_MULTI_QUERY": USE_MULTI_QUERY,
        "USE_HYDE": USE_HYDE,
        "USE_RERANK": USE_RERANK,
        "USE_REVIEW_SEARCH": USE_REVIEW_SEARCH,
        "USE_ROUTER": USE_ROUTER,
    }
    return settings


def all_switches_on():
    """Return the settings dictionary with EVERY switch turned on (used by tests)."""
    settings = default_settings()
    for switch_name in settings:
        settings[switch_name] = True
    return settings


# Small demo: print the main settings when this file is run on its own.
if __name__ == "__main__":
    print("Step 1: paths")
    print("  Project folder:", PROJECT_FOLDER)
    print("  Products file exists:", PRODUCTS_CSV.exists())
    print("  Reviews file exists:", REVIEWS_CSV.exists())
    print("  Test questions file exists:", TEST_QUESTIONS_CSV.exists())

    print("\nStep 2: models")
    print("  Chat model:", CHAT_MODEL)
    print("  Embedding model:", EMBEDDING_MODEL, "(", EMBEDDING_SIZE, "numbers )")

    print("\nStep 3: search numbers")
    print("  TOP_K =", TOP_K, "| CANDIDATES =", CANDIDATES,
          "| RERANK_CANDIDATES =", RERANK_CANDIDATES, "| RRF_K =", RRF_K)

    print("\nStep 4: the switches from default_settings()")
    for switch_name, is_on in default_settings().items():
        print("  ", switch_name, "=", is_on)
