"""
app.py: the TrailMart Smart Product Finder, as a Streamlit web page.

Module 5 technique: all techniques together, with a switch for each one.
- Main page: a question box, product cards, the answer, and "How we searched".
- Sidebar: one checkbox per switch, so you can turn improvements on live in class
  and see how the results change. The checkboxes build the settings dictionary;
  config.py is never edited.

How to run (run python -m src.build_index first, and close other scripts that use qdrant_data/):
    streamlit run app.py
"""

import pandas as pd
import streamlit as st

from src import config
from src.answer import answer_question

# The label shown next to each switch in the sidebar.
SWITCH_LABELS = {
    "USE_HYBRID": "Hybrid search (BM25 + meaning, RRF)",
    "USE_FILTERS": "Self-query filters (price, size...)",
    "USE_QUERY_REWRITE": "Rewrite the question",
    "USE_MULTI_QUERY": "Multi-query (ask 3 ways)",
    "USE_HYDE": "HyDE (search with a draft answer)",
    "USE_RERANK": "Rerank with the LLM",
    "USE_REVIEW_SEARCH": "Also search the reviews",
    "USE_ROUTER": "Router (SQL for table questions)",
}


def show_sidebar():
    """Show one checkbox per switch and return the settings dictionary they make."""
    st.sidebar.header("Switches")
    st.sidebar.caption("All off = basic RAG. Turn them on one at a time and compare.")
    settings = config.default_settings()
    for switch_name in settings:
        settings[switch_name] = st.sidebar.checkbox(SWITCH_LABELS[switch_name], value=settings[switch_name])
    return settings


def show_product_cards(products):
    """Show each product as a small card in a row: name, code, price, sizes, waterproof, rating."""
    if len(products) == 0:
        st.info("No products found.")
        return
    columns = st.columns(len(products))
    for column, product in zip(columns, products):
        with column.container(border=True):
            st.markdown(f"**{product['name']}**")
            st.caption(product["code"] + " · " + product["category"])
            st.write(f"Rs {product['price']}")
            st.write("Sizes: " + ", ".join(str(size) for size in product["sizes"]))
            st.write("Waterproof: " + ("yes" if product["waterproof"] else "no"))
            st.write(f"Rating: {product['rating']} ★")


def show_how_we_searched(details):
    """Show every step the finder took, inside a box the user can open."""
    with st.expander("How we searched"):
        st.write(f"**Route:** {details['route']}  ·  **Time:** {details['seconds']} seconds")
        if details["rewritten_question"]:
            st.write("**Rewritten question:**", details["rewritten_question"])
        if details["filters"]:
            st.write("**Filters from the question:**", details["filters"])
        if details["sub_questions"]:
            st.write("**Sub-questions:**", details["sub_questions"])
        if details["search_versions"]:
            st.write("**Multi-query versions:**", details["search_versions"])
        for draft in details["hyde_drafts"]:
            st.write("**HyDE draft:**", draft)
        show_sql(details)
        show_review_hits(details)
        if details["rerank_scores"]:
            st.write("**Rerank scores (0 to 10):**")
            st.dataframe(pd.DataFrame(details["rerank_scores"]), hide_index=True)
        st.write("**Final codes:**", details["codes"])


def show_sql(details):
    """Show the SQL the LLM wrote, and the rows it returned."""
    if details["sql"] is None:
        return
    st.write("**SQL used:**")
    st.code(details["sql"], language="sql")
    if details["sql_rows"] is not None:
        rows_table = pd.DataFrame(details["sql_rows"]["rows"], columns=details["sql_rows"]["columns"])
        st.dataframe(rows_table, hide_index=True)


def show_review_hits(details):
    """Show the reviews that the review search found (best first)."""
    if len(details["review_hits"]) == 0:
        return
    st.write("**Reviews found by review search:**")
    for hit in details["review_hits"][:config.TOP_K]:
        st.write(f"- ({hit['score']}) {hit['text']}")


def main():
    """Draw the page: sidebar, question box, and the results after the button is pressed."""
    st.set_page_config(page_title="TrailMart Smart Product Finder", layout="wide")
    settings = show_sidebar()
    st.title("TrailMart Smart Product Finder")
    question = st.text_input("What are you looking for?", value="Waterproof trail shoes under Rs 5,000, size 9")

    if st.button("Find shoes", type="primary") and question.strip() != "":
        with st.spinner("Searching..."):
            result = answer_question(question, settings)
        show_product_cards(result["products"])
        st.subheader("Answer")
        st.markdown(result["answer"])
        show_how_we_searched(result["search"])


main()
