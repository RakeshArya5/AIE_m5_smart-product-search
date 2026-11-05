# Smart Product Finder (Module 5 project)

## What we are building
A Streamlit app for an online shoe store called TrailMart. Shoppers ask things like
"Waterproof trail shoes under Rs 5,000, size 9, good reviews" and get the right products,
with reasons taken from customer reviews.

The project tells a story: we start with BASIC RAG (like Module 4), watch it fail on real
questions, then add one improvement at a time and MEASURE each one with a scorecard.

This is a TEACHING project for beginner students. Students must be able to read and understand every line.

## Code style rules (most important)
1. Write very, very easy code. A beginner who knows basic Python must understand it.
2. Plain functions only. No classes, no type hints, no decorators, no lambda, no list/dict comprehensions longer than one short line, no clever one-liners.
3. One job per file. Keep each function short (under about 20 lines).
4. Put a short comment above every block of code explaining WHAT it does and WHY, in simple English.
5. Start every file with a docstring: what this file does, which Module 5 technique it teaches, and how to run it.
6. Use clear, long names: `product_codes`, `review_chunks`, not `pc` or `rc`.
7. Every file in `src/` must be runnable on its own with `python -m src.<file_name>` and must print what it did, step by step, with a small demo.
8. Keep all settings in `src/config.py` only. Never repeat a setting in another file.
9. Use only the libraries already installed (see requirements.txt). Do not install anything new without asking me first. No PyTorch, no sentence-transformers.
10. Prefer LangChain's ready-made building blocks where they keep the code short, and add a comment explaining what each LangChain piece does behind the scenes.

## How to work with me
- After writing each file, run it and fix any errors before moving on.
- We are on Windows. Run files from the project root with `python -m src.<file_name>`. The virtual environment `.venv` is already active.
- Never read, print or change `.env`. Never touch `.venv/`.
- Qdrant runs in local folder mode: only one program can open `qdrant_data/` at a time. Always call `qdrant_client.close()` at the end of a script.

## Tech stack
- Python, LangChain, OpenAI, Qdrant (local folder mode), rank_bm25, SQLite (built into Python), pandas, Streamlit.
- Models (set only in config.py): chat `gpt-4o-mini`, embeddings `text-embedding-3-small` (1536 numbers).
- API key from `.env` with `python-dotenv`.
- Embedding cache (Part 8): the same text must never be embedded (and paid for) twice. Use LangChain's `CacheBackedEmbeddings` + `LocalFileStore` in `data/cache/` if the installed LangChain version has them (check the import first); otherwise write a tiny cache with plain functions that saves embeddings in a JSON file in `data/cache/`.

## The data (read before writing code)
- `data/raw/products.csv`: 120 shoes. Columns: code, name, brand, category, gender (men/women/unisex/kids), price (rupees), sizes (like `7|8|9|10`), waterproof (yes/no), colour, rating, review_count, description.
  Codes look alike on purpose (TS-4412, TS-4414, TS-4412K), because meaning search is weak with codes.
- `data/raw/reviews.csv`: about 390 short reviews. Columns: review_id, code, reviewer, rating, text.
- `data/test/test_questions.csv`: 31 test questions. Columns: question, type, expected_codes (codes separated by `|`), note.
  Two questions ("not in catalogue") have no expected codes: the answer must say the item is not available.

## The switches (the heart of the project)
`config.py` has one switch per improvement. All are `False` at the start (= basic RAG):
```python
USE_HYBRID = False          # Part 3: BM25 keyword search + meaning search, merged with RRF
USE_FILTERS = False         # Part 3: self-querying: the LLM turns "under 5000, size 9" into exact filters
USE_QUERY_REWRITE = False   # Part 4: the LLM cleans up messy questions before searching
USE_MULTI_QUERY = False     # Part 4: ask the question 3 ways and merge the results
USE_HYDE = False            # Part 4: search with a draft answer instead of the question
USE_RERANK = False          # Part 5: fetch 20 candidates, the LLM re-scores them, keep the best
USE_REVIEW_SEARCH = False   # Part 7: also search the reviews and bring their products in (multi-source)
USE_ROUTER = False          # Part 7: send table questions (cheapest, how many, highest rated) to SQL
```
IMPORTANT: `finder.py`, `evaluate.py` and `app.py` must pass the switches as a simple dictionary
(`settings`), built from config by a helper `default_settings()`. This lets `evaluate.py` and the app
turn switches on and off WITHOUT editing config.py.

## Folder structure (final)
```text
m5_smart-product-finder/
├── CLAUDE.md, .env, .gitignore, requirements.txt, README.md
├── app.py
├── data/
│   ├── raw/          products.csv, reviews.csv
│   ├── test/         test_questions.csv
│   ├── cache/        embedding cache (created automatically)
│   ├── results/      scorecard.csv (created by evaluate.py)
│   └── shop.db       SQLite database (created by build_index.py)
├── qdrant_data/      created automatically
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── load_data.py
│   ├── build_index.py
│   ├── search_basic.py
│   ├── search_hybrid.py
│   ├── filters.py
│   ├── query_tricks.py
│   ├── rerank.py
│   ├── sql_tools.py
│   ├── finder.py
│   ├── answer.py
│   └── evaluate.py
└── tests/
    └── quick_test.py
```

## What each file does
1. `config.py`: paths, models, TOP_K = 5, RERANK_CANDIDATES = 20, RRF_K = 60, the switches, and `default_settings()` which returns the switches as a dictionary.
2. `load_data.py`: read the two CSVs with pandas.
   - Make one "product page" text per product (all its fields in one readable paragraph).
   - Make one text per review WITH a contextual header, like `[Product: TrailRunner X | TS-4412 | trail running shoe | Review] Stayed dry...` (Part 6: contextual chunk headers).
   - Products and reviews are short, so no chunk splitting is needed. Each review is a small "child" linked to its "parent" product by `code` (Part 6: parent-child idea).
3. `build_index.py`:
   - Qdrant collection `products`: one point per product page. Payload metadata must include code, name, category, gender, price (int), sizes (list of ints), waterproof (true/false), rating (float), so we can filter on them.
   - Qdrant collection `reviews`: one point per review. Payload metadata: code, review_id, rating.
   - SQLite `data/shop.db`: table `products` with code, name, brand, category, gender, price, waterproof (1/0), rating, review_count, min_size, max_size.
   - Delete and rebuild everything each time it runs (no duplicates). Use the embedding cache.
4. `search_basic.py`: the Module 4 baseline. Meaning search on `products`, returns the top product codes with scores.
5. `search_hybrid.py`: BM25 over the product pages with `rank_bm25` (keep codes like `ts-4412` as one word when splitting text), plus meaning search, merged with Reciprocal Rank Fusion: score = 1 / (RRF_K + rank), added up across lists.
6. `filters.py`: `self_query(question)` asks the LLM for JSON with: search_text, max_price, min_price, size, waterproof, gender, category (null when not mentioned). Check every value before using it. `build_qdrant_filter(rules)` and `passes_rules(product, rules)` (the same rules in plain Python, for BM25 results).
7. `query_tricks.py`: `rewrite_question`, `multi_query` (3 versions, merged with RRF), `hyde_draft` (a short fake product description to search with), and `split_into_sub_questions` (for "compare A and B" questions).
8. `rerank.py`: reranking WITHOUT any new library: send the question and up to 20 candidate product pages to the LLM in ONE call, ask for a JSON relevance score 0 to 10 for each code, and sort by score. Explain in comments that this is an "LLM as reranker" and what a real cross-encoder would do instead.
9. `sql_tools.py`: `text_to_sql(question)` (the LLM writes ONE SELECT for the `products` table), `is_safe_sql(sql)` (must start with SELECT, no second statement, no INSERT/UPDATE/DELETE/DROP/ALTER/CREATE, add LIMIT 20 if missing), `run_sql`, and `route_question(question)` which returns "SQL", "RAG" or "BOTH".
10. `finder.py`: `find_products(question, settings)` runs the search using ONLY the switches that are on, and returns a dictionary: the product codes found (best first), plus "how we searched" details (route, rewritten question, filters, SQL used, time taken). Order: route (if on) → rewrite (if on) → self-query filters (if on) → search (basic or hybrid, multi-query, HyDE) → review search (if on) → rerank (if on).
    For "compare A and B" questions, use `split_into_sub_questions` when USE_MULTI_QUERY is on, and keep the best product for each sub-question.
11. `answer.py`: `answer_question(question, settings)`: call the finder, collect the found products' details and their top reviews, and ask the LLM to recommend products with price, sizes and a reason quoted from reviews.
    Rules for the LLM: use only the given products and reviews; if nothing matches, say the item is not available at TrailMart; never invent products, prices or sizes. The list of product codes shown under the answer is added IN CODE.
12. `evaluate.py`: the scorecard.
    - Metrics (plain functions, easy to read): hit rate@5, MRR, precision@5, recall@5, over the questions that have expected codes. Also the average time per question.
    - Run these versions in order, each adding one switch to the one before: 1 Basic RAG → 2 + Hybrid → 3 + Filters → 4 + Rerank → 5 + Review search → 6 + Router.
      Then test the question tricks (rewrite, multi-query, HyDE) one at a time ON TOP of version 6, marked "(test)".
    - Print the scorecard as a table, say KEEP or UNDO for each step (better or not than the step before), save it to `data/results/scorecard.csv`, and print which questions each version still misses.
13. `app.py`: Streamlit. Main page: question box, product cards (name, code, price, sizes, waterproof, rating), the answer, and an expander "How we searched". Sidebar: one checkbox per switch, so switches can be turned on live in class.
14. `tests/quick_test.py`: ask 8 sample questions (one of each type, including one "not in catalogue") with ALL switches on, and print the answers.
