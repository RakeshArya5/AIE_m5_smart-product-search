"""
evaluate.py: the scorecard. Measure every improvement, one switch at a time.

Module 5 technique: evaluation. "Does it feel better?" is not good enough, so we
ask all 31 test questions, compare the products found with the expected products,
and compute four numbers (all look at the top 5 products, "@5"):
- hit rate@5: in how many questions was at least ONE right product in the top 5?
- MRR (mean reciprocal rank): 1 / the position of the first right product
  (1st place = 1.0, 2nd = 0.5, 3rd = 0.33...), averaged. Rewards putting it at the TOP.
- precision@5: how many of the 5 shown products are right (3 right = 0.6)?
- recall@5: how many of ALL the right products did we show (2 of 4 = 0.5)?
The 2 "not in catalogue" questions have no expected products, so they are not scored
(test those with tests/quick_test.py: the ANSWER must say "not available").

Versions, each adding one switch to the one before:
    1 Basic RAG -> 2 + Hybrid -> 3 + Filters -> 4 + Rerank -> 5 + Review search -> 6 + Router
Then the question tricks (rewrite, multi-query, HyDE), each tested ALONE on top of version 6.
KEEP = the overall score went up compared to the step before. UNDO = it did not.

How to run (run python -m src.build_index first; this makes a few hundred cheap LLM calls):
    python -m src.evaluate
"""

import pandas as pd

from src import config
from src.build_index import close_qdrant_client
from src.finder import find_products

# ---------------------------------------------------------------
# Reading the test questions
# ---------------------------------------------------------------


def load_test_questions():
    """Read the test questions. expected_codes becomes a list (empty for "not in catalogue")."""
    table = pd.read_csv(config.TEST_QUESTIONS_CSV).fillna("")
    test_questions = []
    for row in table.to_dict("records"):
        expected_codes = []
        if row["expected_codes"] != "":
            expected_codes = row["expected_codes"].split("|")
        test_questions.append({
            "question": row["question"],
            "type": row["type"],
            "expected_codes": expected_codes,
        })
    return test_questions

# ---------------------------------------------------------------
# The four metrics (for ONE question; we average them later)
# ---------------------------------------------------------------


def hit_at_k(found_codes, expected_codes, k):
    """1 if at least one expected product is in the top k, otherwise 0."""
    for code in found_codes[:k]:
        if code in expected_codes:
            return 1
    return 0


def reciprocal_rank(found_codes, expected_codes, k):
    """1 / position of the first expected product in the top k (0 if none is there)."""
    for position in range(len(found_codes[:k])):
        if found_codes[position] in expected_codes:
            return 1 / (position + 1)
    return 0


def precision_at_k(found_codes, expected_codes, k):
    """Share of the k shown places that hold an expected product."""
    right_count = 0
    for code in found_codes[:k]:
        if code in expected_codes:
            right_count = right_count + 1
    return right_count / k


def recall_at_k(found_codes, expected_codes, k):
    """Share of ALL expected products that appear in the top k."""
    right_count = 0
    for code in found_codes[:k]:
        if code in expected_codes:
            right_count = right_count + 1
    return right_count / len(expected_codes)

# ---------------------------------------------------------------
# Running and scoring one version
# ---------------------------------------------------------------


def make_versions():
    """Return the list of versions to test: (name, settings, compare-with-name)."""
    versions = []
    settings = config.default_settings()
    versions.append(("1 Basic RAG", dict(settings), None))
    # Each step turns ONE more switch on, on top of the step before.
    steps = [("2 + Hybrid", "USE_HYBRID"), ("3 + Filters", "USE_FILTERS"),
             ("4 + Rerank", "USE_RERANK"), ("5 + Review search", "USE_REVIEW_SEARCH"),
             ("6 + Router", "USE_ROUTER")]
    previous_name = "1 Basic RAG"
    for name, switch_name in steps:
        settings[switch_name] = True
        versions.append((name, dict(settings), previous_name))
        previous_name = name
    # The question tricks: each one ALONE on top of version 6, compared with version 6.
    for name, switch_name in [("6 + Rewrite (test)", "USE_QUERY_REWRITE"),
                              ("6 + Multi-query (test)", "USE_MULTI_QUERY"),
                              ("6 + HyDE (test)", "USE_HYDE")]:
        test_settings = dict(settings)
        test_settings[switch_name] = True
        versions.append((name, test_settings, "6 + Router"))
    return versions


def run_version(settings, test_questions):
    """Ask every test question with these settings. Returns the codes found and the time taken."""
    runs = []
    for test_question in test_questions:
        details = find_products(test_question["question"], settings)
        runs.append({"codes": details["codes"], "seconds": details["seconds"]})
        # Print a dot per question, so we can see it is working.
        print(".", end="", flush=True)
    print()
    return runs


def score_version(name, test_questions, runs):
    """Average the four metrics over the questions that have expected codes."""
    totals = {"hit": 0, "mrr": 0, "precision": 0, "recall": 0, "seconds": 0}
    scored_count = 0
    missed_questions = []
    for test_question, run in zip(test_questions, runs):
        totals["seconds"] = totals["seconds"] + run["seconds"]
        expected = test_question["expected_codes"]
        if len(expected) == 0:
            continue  # "not in catalogue": nothing to score here
        scored_count = scored_count + 1
        hit = hit_at_k(run["codes"], expected, config.TOP_K)
        totals["hit"] = totals["hit"] + hit
        totals["mrr"] = totals["mrr"] + reciprocal_rank(run["codes"], expected, config.TOP_K)
        totals["precision"] = totals["precision"] + precision_at_k(run["codes"], expected, config.TOP_K)
        totals["recall"] = totals["recall"] + recall_at_k(run["codes"], expected, config.TOP_K)
        if hit == 0:
            missed_questions.append(test_question)
    return make_score_row(name, totals, scored_count, len(runs)), missed_questions


def make_score_row(name, totals, scored_count, question_count):
    """Turn the totals into averages, plus one "overall" number (the average of the four)."""
    row = {
        "version": name,
        "hit_rate@5": round(totals["hit"] / scored_count, 3),
        "MRR": round(totals["mrr"] / scored_count, 3),
        "precision@5": round(totals["precision"] / scored_count, 3),
        "recall@5": round(totals["recall"] / scored_count, 3),
        "avg_seconds": round(totals["seconds"] / question_count, 2),
    }
    four_metrics = row["hit_rate@5"] + row["MRR"] + row["precision@5"] + row["recall@5"]
    row["overall"] = round(four_metrics / 4, 3)
    return row

# ---------------------------------------------------------------
# KEEP or UNDO, printing and saving
# ---------------------------------------------------------------


def add_decisions(score_rows, versions):
    """Say KEEP if the overall score went up compared with the step before, otherwise UNDO."""
    rows_by_name = {}
    for row in score_rows:
        rows_by_name[row["version"]] = row
    for row, version in zip(score_rows, versions):
        compare_name = version[2]
        if compare_name is None:
            row["decision"] = "(start)"
            row["compared_with"] = "-"
        elif row["overall"] > rows_by_name[compare_name]["overall"]:
            row["decision"] = "KEEP"
            row["compared_with"] = compare_name
        else:
            row["decision"] = "UNDO"
            row["compared_with"] = compare_name


def print_misses(name, missed_questions):
    """Print the questions a version still gets wrong (no right product in the top 5)."""
    print(f"\n{name}: misses {len(missed_questions)} question(s)")
    for test_question in missed_questions:
        print(f"   - [{test_question['type']}] {test_question['question']}")


def run_scorecard():
    """Run every version, print the scorecard and the misses, and save scorecard.csv."""
    test_questions = load_test_questions()
    versions = make_versions()
    print("Step 1:", len(test_questions), "test questions,", len(versions), "versions to test\n")

    score_rows = []
    all_misses = []
    for name, settings, compare_name in versions:
        print("Step 2: running", name, end=" ")
        runs = run_version(settings, test_questions)
        score_row, missed_questions = score_version(name, test_questions, runs)
        score_rows.append(score_row)
        all_misses.append((name, missed_questions))

    add_decisions(score_rows, versions)
    scorecard = pd.DataFrame(score_rows)
    print("\nStep 3: the scorecard (scored on questions with expected products)\n")
    print(scorecard.to_string(index=False))

    config.RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    scorecard.to_csv(config.SCORECARD_CSV, index=False)
    print("\nStep 4: saved to", config.SCORECARD_CSV)

    print("\nStep 5: the questions each version still misses")
    for name, missed_questions in all_misses:
        print_misses(name, missed_questions)


if __name__ == "__main__":
    run_scorecard()
    # Always close the local Qdrant folder at the end.
    close_qdrant_client()
