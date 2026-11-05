"""
sql_tools.py: answer TABLE questions with SQL, and decide which questions need SQL.

Module 5 technique (Part 7): routing + text-to-SQL.
Some questions are about the whole table, not about meaning:
"What is the cheapest waterproof hiking boot?", "How many sandals under Rs 1,500?".
Meaning search cannot sort by price or count. A database can, in one line of SQL.
- route_question: the LLM decides: "SQL" (table question), "RAG" (meaning question)
  or "BOTH" (a table part AND a review/meaning part).
- text_to_sql: the LLM writes ONE SELECT query for the products table.
- is_safe_sql: we NEVER run LLM-written SQL blindly. Only a single SELECT is allowed.
- run_sql: runs the query on data/shop.db, opened in read-only mode.

How to run (run python -m src.build_index first):
    python -m src.sql_tools
"""

import re
import sqlite3

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src import config

FORBIDDEN_WORDS = ["insert", "update", "delete", "drop", "alter", "create", "attach", "pragma", "replace"]

SQL_PROMPT = """You write ONE SQLite SELECT query for this table:
products(code TEXT, name TEXT, brand TEXT, category TEXT, gender TEXT,
         price INTEGER (rupees), waterproof INTEGER (1 = yes, 0 = no), rating REAL (0 to 5),
         review_count INTEGER, min_size INTEGER, max_size INTEGER)
Categories: {categories}
Genders: men, women, unisex, kids

Rules:
- Always SELECT code, name, price, rating (code first).
- For "how many" questions, select the matching rows, NOT COUNT(*). We count the rows ourselves.
- For "cheapest", "most expensive" or "highest rated", ORDER BY so the best answer comes first.
- Never use a LIMIT smaller than {row_limit}.
- A flip-flop is a kind of sandal: for sandals use category IN ('sandal', 'flip-flop').
- For a size, use min_size <= size AND max_size >= size.
- The table has no reviews: ignore any part of the question about reviews.
- Reply with the SQL only, no explanation, no markdown.

Question: {question}"""

ROUTE_PROMPT = """Decide how a shoe shop should answer this question. Reply with ONE word:
SQL  = a table question: cheapest, most expensive, highest or lowest rated, how many, sorting or counting.
RAG  = a meaning question: find or describe shoes, a code or name, a need, or what reviews say.
BOTH = a table part AND a part about what REVIEWS say (for example "cheapest shoe reviewers call light").
The table has these columns: code, name, brand, category, gender, price, waterproof, rating,
review_count, sizes. A question that only uses these columns (like "cheapest waterproof sneaker")
is SQL, not BOTH.

Question: {question}"""


def make_llm():
    """Return the chat model. temperature=0 makes the answers stable from run to run."""
    return ChatOpenAI(model=config.CHAT_MODEL, temperature=0)


def known_categories():
    """Read the list of categories from the database, so the LLM uses real values."""
    rows = run_sql("SELECT DISTINCT category FROM products")["rows"]
    categories = []
    for row in rows:
        categories.append(row[0])
    return categories


def text_to_sql(question):
    """Ask the LLM to write ONE SELECT query that answers the question."""
    # prompt -> LLM -> StrOutputParser (which just takes the reply text out of the message)
    chain = ChatPromptTemplate.from_template(SQL_PROMPT) | make_llm() | StrOutputParser()
    sql = chain.invoke({
        "question": question,
        "categories": ", ".join(known_categories()),
        "row_limit": config.SQL_ROW_LIMIT,
    })
    # Sometimes the model still wraps the SQL in ```sql ... ```, so we remove that.
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql


def is_safe_sql(sql):
    """Return True only if the SQL is ONE read-only SELECT statement."""
    clean_sql = sql.strip().rstrip(";").strip().lower()
    # 1. It must start with SELECT.
    if not clean_sql.startswith("select"):
        return False
    # 2. No second statement: a ";" in the middle means "run another command".
    if ";" in clean_sql:
        return False
    # 3. No words that change the database (checked as whole words, so "updated_at" is fine).
    for word in FORBIDDEN_WORDS:
        if re.search(r"\b" + word + r"\b", clean_sql):
            return False
    return True


def add_limit_if_missing(sql):
    """Add LIMIT 20 at the end if the query has no LIMIT, so we never fetch a huge table."""
    clean_sql = sql.strip().rstrip(";").strip()
    if not re.search(r"\blimit\b", clean_sql.lower()):
        clean_sql = clean_sql + " LIMIT " + str(config.SQL_ROW_LIMIT)
    return clean_sql


def run_sql(sql):
    """Run a SELECT on data/shop.db in READ-ONLY mode. Returns {"columns": [...], "rows": [...]}."""
    # "mode=ro" opens the file read-only: even a bad query cannot change the data.
    database_address = config.SQLITE_FILE.as_uri() + "?mode=ro"
    connection = sqlite3.connect(database_address, uri=True)
    try:
        cursor = connection.execute(sql)
        column_names = []
        for column in cursor.description:
            column_names.append(column[0])
        rows = cursor.fetchall()
    finally:
        connection.close()
    return {"columns": column_names, "rows": rows}


def route_question(question):
    """Ask the LLM if the question needs "SQL", "RAG" or "BOTH". Anything else counts as "RAG"."""
    chain = ChatPromptTemplate.from_template(ROUTE_PROMPT) | make_llm() | StrOutputParser()
    route = chain.invoke({"question": question}).strip().upper()
    if route in ["SQL", "RAG", "BOTH"]:
        return route
    return "RAG"


# Small demo: route three questions, then answer a table question with safe SQL.
if __name__ == "__main__":
    print("Step 1: routing questions")
    for demo_question in ["What is the cheapest waterproof hiking boot?",
                          "something for snow treks",
                          "Cheapest walking shoe that reviewers call comfortable"]:
        print("  ", route_question(demo_question), "<-", demo_question)

    demo_question = "How many sandals do you have under Rs 1,500?"
    print("\nStep 2: text-to-SQL for:", demo_question)
    demo_sql = text_to_sql(demo_question)
    print("  ", demo_sql)

    print("\nStep 3: safety check")
    print("   our query is safe:", is_safe_sql(demo_sql))
    print("   'DROP TABLE products' is safe:", is_safe_sql("DROP TABLE products"))
    print("   'SELECT 1; DELETE FROM products' is safe:", is_safe_sql("SELECT 1; DELETE FROM products"))

    print("\nStep 4: run it (LIMIT added if missing)")
    demo_result = run_sql(add_limit_if_missing(demo_sql))
    print("   columns:", demo_result["columns"])
    for demo_row in demo_result["rows"]:
        print("  ", demo_row)
    print("   number of rows:", len(demo_result["rows"]))
