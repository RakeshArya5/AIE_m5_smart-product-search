"""
quick_test.py: ask 8 sample questions with ALL switches on, and print the answers.

Module 5 technique: a quick "eyeball" test of the full system. The scorecard
(src/evaluate.py) checks the product codes; this test lets us READ the final answers,
for example to check that a "not in catalogue" question says the item is not available.

How to run (run python -m src.build_index first):
    python -m tests.quick_test
"""

from src import config
from src.answer import answer_question
from src.build_index import close_qdrant_client

# One question of each type (from data/test/test_questions.csv).
SAMPLE_QUESTIONS = [
    ("exact code", "Do you have TS-4412 in size 9?"),
    ("filter", "Women's walking shoes under Rs 3,500"),
    ("vague", "shoes for my dad who can't tie laces"),
    ("reviews", "Which shoes do reviewers say are good for wide feet?"),
    ("multi-part", "Compare SpeedLine and SpeedLine Pro"),
    ("table (SQL)", "How many sandals do you have under Rs 1,500?"),
    ("table + reviews", "Cheapest walking shoe that reviewers call comfortable"),
    ("not in catalogue", "Do you have football boots?"),
]


def run_quick_test():
    """Ask every sample question with all switches on and print the answer."""
    settings = config.all_switches_on()
    print("All switches on:", settings, "\n")
    for number, (question_type, question) in enumerate(SAMPLE_QUESTIONS, start=1):
        print("=" * 80)
        print(f"Question {number} [{question_type}]: {question}")
        result = answer_question(question, settings)
        print(f"(route: {result['search']['route']}, {result['search']['seconds']} seconds of search)")
        print("-" * 80)
        print(result["answer"])
        print()


if __name__ == "__main__":
    run_quick_test()
    # Always close the local Qdrant folder at the end.
    close_qdrant_client()
