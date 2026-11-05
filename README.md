# TrailMart Smart Product Finder (Module 5 project)

A Streamlit app for an online shoe store, TrailMart. Shoppers ask questions like
*"Waterproof trail shoes under Rs 5,000, size 9, good reviews"* and get the right shoes,
with reasons quoted from customer reviews.

**The story:** we start with **basic RAG** (Module 4) and watch it fail on real questions.
Then we add **one improvement at a time** and **measure each one** with a scorecard.

## Setup (once)

1. Activate the virtual environment `.venv` (it already has all the libraries from `requirements.txt`).
2. Put your OpenAI key in `.env` as `OPENAI_API_KEY=...`.
3. Build the search indexes (Qdrant + SQLite). Embeddings are cached in `data/cache/`,
   so running it again costs nothing:

```bash
python -m src.build_index
```

## Run it

| What | Command |
|---|---|
| The scorecard (all versions, KEEP / UNDO) | `python -m src.evaluate` |
| 8 sample answers with all switches on | `python -m tests.quick_test` |
| The app | `streamlit run app.py` |
| Any single file's demo | `python -m src.<file_name>` (for example `python -m src.search_hybrid`) |

Qdrant runs in local folder mode: only **one** program can open `qdrant_data/` at a time.
Stop the app before running a script, and the other way round.

## The switches

All switches live in `src/config.py` and start as `False` (= basic RAG).
The app's sidebar and `evaluate.py` turn them on through a `settings` dictionary,
so you never have to edit `config.py` to try them.

| Switch | Part | What it does |
|---|---|---|
| `USE_HYBRID` | 3 | BM25 keyword search + meaning search, merged with Reciprocal Rank Fusion |
| `USE_FILTERS` | 3 | Self-querying: the LLM turns "under 5000, size 9" into exact filters |
| `USE_QUERY_REWRITE` | 4 | The LLM cleans up messy questions before searching |
| `USE_MULTI_QUERY` | 4 | Ask the question 3 ways and merge the results; split "compare A and B" |
| `USE_HYDE` | 4 | Search with a draft product description instead of the question |
| `USE_RERANK` | 5 | Fetch 20 candidates, the LLM re-scores them, keep the best |
| `USE_REVIEW_SEARCH` | 7 | Also search the reviews and bring their products in |
| `USE_ROUTER` | 7 | Send table questions (cheapest, how many, highest rated) to SQL |

## Files

| File | What it does |
|---|---|
| `src/config.py` | All settings: paths, models, numbers, the switches, `default_settings()` |
| `src/load_data.py` | Reads the CSVs; makes product pages and reviews with contextual headers (Part 6) |
| `src/build_index.py` | Builds Qdrant `products` + `reviews` and SQLite `shop.db`, with the embedding cache (Part 8) |
| `src/search_basic.py` | Basic meaning search (the Module 4 baseline) and review search |
| `src/search_hybrid.py` | BM25 + meaning search merged with RRF (Part 3) |
| `src/filters.py` | Self-query: the LLM makes filters; Qdrant filter + the same rules in Python (Part 3) |
| `src/query_tricks.py` | Rewrite, multi-query, HyDE, sub-questions (Part 4) |
| `src/rerank.py` | The LLM as a reranker (Part 5) |
| `src/sql_tools.py` | Router, text-to-SQL, safety check, read-only SQL (Part 7) |
| `src/finder.py` | Runs the steps whose switch is on, in the right order |
| `src/answer.py` | Asks the LLM for a recommendation using only the found products and reviews |
| `src/evaluate.py` | The scorecard: hit rate@5, MRR, precision@5, recall@5, time, KEEP/UNDO |
| `app.py` | The Streamlit app with a checkbox per switch |
| `tests/quick_test.py` | 8 sample questions (one per type) with all switches on |

## The data

- `data/raw/products.csv`: 120 shoes. Codes look alike on purpose (TS-4412, TS-4414, TS-4412K).
- `data/raw/reviews.csv`: about 390 short reviews.
- `data/test/test_questions.csv`: 31 test questions with the expected product codes.
