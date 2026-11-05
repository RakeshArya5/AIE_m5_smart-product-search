"""
finder.py: find the right products for a question, using ONLY the switches that are on.

Module 5 technique: this file puts all the improvements together in one pipeline.
The order of the steps is:
    route (SQL / RAG / BOTH)       if USE_ROUTER is on
    -> rewrite the question         if USE_QUERY_REWRITE is on
    -> self-query filters           if USE_FILTERS is on
    -> search: basic or hybrid,     USE_HYBRID
       asked several ways,          USE_MULTI_QUERY (also splits "compare A and B")
       with a HyDE draft            USE_HYDE
    -> review search                if USE_REVIEW_SEARCH is on
    -> rerank                       if USE_RERANK is on
With every switch off, this is exactly basic RAG (a meaning search).

The switches come in a dictionary called settings (see config.default_settings()),
so evaluate.py and the app can turn switches on and off without editing config.py.

How to run (run python -m src.build_index first):
    python -m src.finder
"""

import sqlite3
import time

from src import config
from src.build_index import close_qdrant_client
from src.filters import allowed_codes_for_rules, build_qdrant_filter, self_query
from src.query_tricks import hyde_draft, multi_query, rewrite_question, split_into_sub_questions
from src.rerank import rerank
from src.search_basic import codes_from_results, search_products_basic, search_reviews
from src.search_hybrid import hybrid_search, rrf_merge
from src.sql_tools import add_limit_if_missing, is_safe_sql, route_question, run_sql, text_to_sql


def new_search_details(question):
    """Make the empty "how we searched" dictionary that every step fills in."""
    return {
        "question": question,
        "codes": [],               # the final product codes, best first
        "route": "RAG",            # SQL, RAG or BOTH
        "rewritten_question": None,
        "filters": None,           # the rules from self-query
        "sub_questions": None,     # for "compare A and B" questions
        "search_versions": None,   # the versions of the question (multi-query)
        "hyde_drafts": [],         # the fake product descriptions (HyDE)
        "sql": None,
        "sql_rows": None,
        "review_hits": [],         # the reviews found by review search
        "rerank_scores": None,
        "steps": [],               # a readable list of what happened
        "seconds": 0,
    }


def add_step(details, text):
    """Remember one readable line about what the finder just did."""
    details["steps"].append(text)


def find_products(question, settings):
    """Find the best products for the question. Returns the "how we searched" dictionary."""
    start_time = time.time()
    details = new_search_details(question)

    # Step 1: the router decides if this is a table question (SQL), a meaning question (RAG) or both.
    if settings["USE_ROUTER"]:
        details["route"] = route_question(question)
        add_step(details, "Router chose: " + details["route"])

    sql_codes = []
    if details["route"] in ["SQL", "BOTH"]:
        sql_codes = find_with_sql(question, details)

    # A pure SQL question is done if SQL found rows. Otherwise (RAG, BOTH, or SQL that
    # found nothing) we run the RAG search too.
    if details["route"] == "SQL" and len(sql_codes) > 0:
        final_codes = sql_codes
    else:
        rag_codes = find_with_rag(question, settings, details)
        final_codes = combine_sql_and_rag(sql_codes, rag_codes)

    details["codes"] = final_codes[:config.TOP_K]
    details["seconds"] = round(time.time() - start_time, 2)
    return details


def find_with_sql(question, details):
    """Let the LLM write SQL, check it is safe, run it, and return the product codes it found."""
    sql = text_to_sql(question)
    details["sql"] = sql
    if not is_safe_sql(sql):
        add_step(details, "The SQL was NOT safe, so we did not run it: " + sql)
        return []
    sql = add_limit_if_missing(sql)
    details["sql"] = sql
    try:
        result = run_sql(sql)
    except sqlite3.Error as error:
        add_step(details, "The SQL failed (" + str(error) + "), so we use RAG instead")
        return []
    details["sql_rows"] = result
    add_step(details, "SQL found " + str(len(result["rows"])) + " rows: " + sql)
    return codes_from_sql_result(result)


def codes_from_sql_result(result):
    """Take the "code" column out of the SQL rows (empty list if there is no code column)."""
    if "code" not in result["columns"]:
        return []
    code_position = result["columns"].index("code")
    codes = []
    for row in result["rows"]:
        if row[code_position] not in codes:
            codes.append(row[code_position])
    return codes


def combine_sql_and_rag(sql_codes, rag_codes):
    """For BOTH questions: SQL decides WHICH products are allowed, RAG decides the ORDER."""
    if len(sql_codes) == 0:
        return rag_codes
    # Products that RAG found AND that match the SQL, in RAG order.
    combined = []
    for code in rag_codes:
        if code in sql_codes:
            combined.append(code)
    # Then the other SQL products (in SQL order), so we always show something.
    for code in sql_codes:
        if code not in combined:
            combined.append(code)
    return combined


def find_with_rag(question, settings, details):
    """The search pipeline: rewrite -> filters -> search -> review search -> rerank."""
    search_question = question
    if settings["USE_QUERY_REWRITE"]:
        search_question = rewrite_question(question)
        details["rewritten_question"] = search_question
        add_step(details, "Rewritten question: " + search_question)

    rules = {}
    if settings["USE_FILTERS"]:
        rules = self_query(search_question)
        details["filters"] = rules
        search_question = rules["search_text"]
        add_step(details, "Filters: " + str(rules))

    codes = search_all_ways(search_question, settings, rules, details)
    if settings["USE_REVIEW_SEARCH"]:
        codes = add_review_search(search_question, codes, rules, details)
    if settings["USE_RERANK"]:
        # The reranker gets the ORIGINAL question, with all its details (price, size...).
        codes = rerank_codes(question, codes, details)
    return codes


def search_all_ways(search_question, settings, rules, details):
    """Search once, or (with multi-query) in several ways merged with RRF."""
    if not settings["USE_MULTI_QUERY"]:
        return search_one_text(search_question, settings, rules, details)

    # "Compare A and B": search A and B separately (see search_sub_questions).
    sub_questions = split_into_sub_questions(search_question)
    if len(sub_questions) > 1:
        details["sub_questions"] = sub_questions
        add_step(details, "Sub-questions: " + str(sub_questions))
        return search_sub_questions(sub_questions, settings, rules, details)

    # Multi-query: search every version of the question and merge the lists with RRF.
    versions = multi_query(search_question)
    details["search_versions"] = versions
    add_step(details, "Multi-query versions: " + str(versions))
    code_lists = []
    for version in versions:
        code_lists.append(search_one_text(version, settings, rules, details))
    return codes_from_results(rrf_merge(code_lists))


def search_sub_questions(sub_questions, settings, rules, details):
    """Search each sub-question, keep the BEST product of each first, then the rest."""
    best_codes = []
    code_lists = []
    for sub_question in sub_questions:
        codes = search_one_text(sub_question, settings, rules, details)
        code_lists.append(codes)
        # The best product for this sub-question (skipping one we already picked).
        for code in codes:
            if code not in best_codes:
                best_codes.append(code)
                break
    # After the best ones, add the other products from all lists (merged with RRF).
    for code in codes_from_results(rrf_merge(code_lists)):
        if code not in best_codes:
            best_codes.append(code)
    return best_codes


def search_one_text(text, settings, rules, details):
    """Run ONE search (basic or hybrid, maybe with a HyDE draft). Returns codes, best first."""
    # HyDE: the meaning search uses a fake product description instead of the question.
    # Keyword search (BM25) still uses the question itself, because codes must match exactly.
    meaning_text = text
    if settings["USE_HYDE"]:
        meaning_text = hyde_draft(text)
        details["hyde_drafts"].append(meaning_text)
        add_step(details, "HyDE draft: " + meaning_text)

    qdrant_filter = build_qdrant_filter(rules)
    if settings["USE_HYBRID"]:
        allowed_codes = allowed_codes_for_rules(rules)
        results = hybrid_search(text, config.CANDIDATES, qdrant_filter, allowed_codes, meaning_text)
    else:
        results = search_products_basic(meaning_text, config.CANDIDATES, qdrant_filter)
    return codes_from_results(results)


def add_review_search(search_question, codes, rules, details):
    """Also search the reviews, and merge the products of those reviews in with RRF."""
    review_hits = search_reviews(search_question, config.REVIEW_TOP_K)
    details["review_hits"] = review_hits
    # Reviews have no price or size, so we apply the filter rules to their products here.
    allowed_codes = allowed_codes_for_rules(rules)
    review_codes = []
    for hit in review_hits:
        is_allowed = allowed_codes is None or hit["code"] in allowed_codes
        if is_allowed and hit["code"] not in review_codes:
            review_codes.append(hit["code"])
    add_step(details, "Review search found products: " + str(review_codes))
    return codes_from_results(rrf_merge([codes, review_codes]))


def rerank_codes(question, codes, details):
    """Let the LLM re-score the candidates, and return the codes in the new order."""
    reranked = rerank(question, codes, config.RERANK_CANDIDATES)
    details["rerank_scores"] = reranked
    add_step(details, "Rerank top scores: " + str(reranked[:config.TOP_K]))
    return codes_from_results(reranked)


def print_details(details):
    """Print the codes found and every step, so we can see how the finder worked."""
    print("   codes found:", details["codes"], "in", details["seconds"], "seconds")
    for step in details["steps"]:
        print("   -", step[:160])


# Small demo: the same question with all switches off, then with all switches on.
if __name__ == "__main__":
    demo_question = "Waterproof trail shoes under Rs 5,000, size 9"
    print("Question:", demo_question, "(expected: TS-4412, TS-4527)")

    print("\nStep 1: basic RAG (all switches off)")
    print_details(find_products(demo_question, config.default_settings()))

    print("\nStep 2: all switches on")
    print_details(find_products(demo_question, config.all_switches_on()))

    demo_question = "What is the cheapest waterproof hiking boot?"
    print("\nStep 3: a table question with all switches on:", demo_question, "(expected: HK-3340)")
    print_details(find_products(demo_question, config.all_switches_on()))

    demo_question = "Which shoes run small according to reviews?"
    print("\nStep 4: a review question with all switches on:", demo_question)
    print_details(find_products(demo_question, config.all_switches_on()))

    # Always close the local Qdrant folder at the end.
    close_qdrant_client()
