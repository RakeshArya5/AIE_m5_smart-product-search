"""
query_tricks.py: four ways to improve the QUESTION before we search with it.

Module 5 techniques (Part 4): query transformation.
- rewrite_question: the LLM turns a messy question into a clear search query
  ("comfy shoes for my office" -> "comfortable formal or walking shoes for office wear").
- multi_query: ask the same question 3 different ways. Each version finds slightly
  different products; the finder merges the lists with RRF (see search_hybrid.rrf_merge).
- hyde_draft (HyDE = Hypothetical Document Embeddings): the LLM writes a short FAKE
  product description that would answer the question. We search with that draft,
  because a product description looks more like our product pages than a question does.
- split_into_sub_questions: "Compare A and B" becomes "A" and "B", searched separately.

How to run:
    python -m src.query_tricks
"""

from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src import config

REWRITE_PROMPT = """You help a shoe shop search engine.
Rewrite the shopper's question as ONE clear search query for a shoe catalogue.
- Keep product codes (like TS-4412), product names, prices and sizes exactly as written.
- Turn vague needs into the shoe types and features that would help, for example
  "something for the beach" -> "sandals or flip-flops for sand and water",
  "shoes for standing all day at work" -> "comfortable cushioned shoes for long hours on your feet".
- Reply with the search query only.

Question: {question}"""

MULTI_QUERY_PROMPT = """You help a shoe shop search engine.
Write {count} different versions of the shopper's question, each using different words,
so a search engine can find more matching shoes.
Keep product codes, product names, prices and sizes exactly as written.
Reply with JSON only: {{"questions": ["...", "..."]}}

Question: {question}"""

HYDE_PROMPT = """Write a short, plain product description (2 sentences) of a shoe that would
perfectly answer this shopper's question, in the style of an online shoe shop.
Mention the type of shoe and its most important features. No brand names, no marketing words.

Question: {question}"""

SPLIT_PROMPT = """Does this shopper question ask about SEVERAL different products
(for example "Compare A and B")? If yes, split it into one short search text per product,
containing only that product's name or code (no words like "compare").
If it is about one product or one kind of product, return the question unchanged as the only item.
Reply with JSON only: {{"sub_questions": ["...", "..."]}}

Question: {question}"""


def make_llm():
    """Return the chat model. temperature=0 makes the answers stable from run to run."""
    return ChatOpenAI(model=config.CHAT_MODEL, temperature=0)


def rewrite_question(question):
    """Ask the LLM to turn a messy question into one clear search query."""
    # prompt -> LLM -> StrOutputParser (which just takes the reply text out of the message)
    chain = ChatPromptTemplate.from_template(REWRITE_PROMPT) | make_llm() | StrOutputParser()
    return chain.invoke({"question": question}).strip()


def multi_query(question):
    """Return MULTI_QUERY_COUNT versions of the question: the original plus new versions from the LLM."""
    extra_count = config.MULTI_QUERY_COUNT - 1
    chain = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT) | make_llm() | JsonOutputParser()
    try:
        reply = chain.invoke({"question": question, "count": extra_count})
        new_versions = reply.get("questions", [])
    except Exception:
        new_versions = []
    # Keep the original question first, then add the new versions that are real text.
    versions = [question]
    for version in new_versions:
        if isinstance(version, str) and version.strip() != "" and len(versions) < config.MULTI_QUERY_COUNT:
            versions.append(version.strip())
    return versions


def hyde_draft(question):
    """Ask the LLM for a short fake product description that answers the question (HyDE)."""
    chain = ChatPromptTemplate.from_template(HYDE_PROMPT) | make_llm() | StrOutputParser()
    return chain.invoke({"question": question}).strip()


def split_into_sub_questions(question):
    """Split "Compare A and B" into ["A", "B"]. Other questions come back as [question]."""
    chain = ChatPromptTemplate.from_template(SPLIT_PROMPT) | make_llm() | JsonOutputParser()
    try:
        reply = chain.invoke({"question": question})
        sub_questions = reply.get("sub_questions", [])
    except Exception:
        sub_questions = []
    # Keep only real text. If we do not get at least 2 parts, the question is not
    # multi-part, so we return the original question unchanged.
    clean_sub_questions = []
    for sub_question in sub_questions:
        if isinstance(sub_question, str) and sub_question.strip() != "":
            clean_sub_questions.append(sub_question.strip())
    if len(clean_sub_questions) < 2:
        return [question]
    return clean_sub_questions


# Small demo: try each trick on a question it is meant for.
if __name__ == "__main__":
    print("Step 1: rewrite a messy question")
    print("   before: comfy shoes for my office")
    print("   after: ", rewrite_question("comfy shoes for my office"))

    print("\nStep 2: multi-query: the same question asked", config.MULTI_QUERY_COUNT, "ways")
    for version in multi_query("something for snow treks"):
        print("   -", version)

    print("\nStep 3: HyDE: a fake product description to search with")
    print("  ", hyde_draft("shoes that are safe for night running"))

    print("\nStep 4: split a compare question into sub-questions")
    print("  ", split_into_sub_questions("Compare SpeedLine and SpeedLine Pro"))
    print("   a normal question stays as one:", split_into_sub_questions("Sandals under Rs 1,000"))
