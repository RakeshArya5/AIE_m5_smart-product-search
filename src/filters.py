"""
filters.py: self-querying: the LLM turns "under Rs 5,000, size 9" into exact filters.

Module 5 technique (Part 3): self-querying retrieval.
Meaning search cannot do maths: "under Rs 5,000" is just words to an embedding.
So we ask the LLM to split the question into:
- search_text: the part to search by meaning ("waterproof trail running shoes")
- exact rules: max_price, min_price, size, waterproof, gender, category
Then Qdrant only searches products that pass the rules (a metadata filter).

The same rules exist in two forms:
- build_qdrant_filter(rules): a Qdrant Filter, for meaning search.
- passes_rules(product, rules): plain Python, for BM25 results (BM25 has no filters).

How to run:
    python -m src.filters
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue, Range

from src import config
from src.load_data import load_products

GENDERS = ["men", "women", "unisex", "kids"]

SELF_QUERY_PROMPT = """You turn a shoe shop question into search filters.
Reply with JSON only, with exactly these keys:
{{"search_text": "...", "max_price": null, "min_price": null, "size": null,
  "waterproof": null, "gender": null, "category": null}}

Rules:
- search_text: the question without prices and sizes. Keep the product type, features and who it is for.
- max_price / min_price: whole rupees as a number ("under Rs 5,000" -> max_price 5000,
  "above Rs 10,000" -> min_price 10000). Otherwise null.
- size: the shoe size as a number, or null.
- waterproof: true only if the shopper asks for waterproof or rainproof shoes, otherwise null.
- gender: one of {genders} if the shopper says who it is for, otherwise null.
- category: a LIST of categories from this list that the shopper clearly asks for, or null:
  {categories}
  A flip-flop is a kind of sandal. "Running shoes" without "road" or "trail" means both
  "road running shoe" and "trail running shoe". Hiking boots are only "hiking boot".
- Use null for anything the shopper did not mention. Never guess.
- If the shopper names a specific product (by its name or code), set category to null:
  a product name like "Monsoon Trekker" does NOT tell you its category.

Question: {question}"""


def known_categories():
    """Return the list of categories that exist in the shop, like ["sandal", "sneaker", ...]."""
    categories = []
    for product in load_products():
        if product["category"] not in categories:
            categories.append(product["category"])
    return categories


def self_query(question):
    """Ask the LLM to turn the question into search_text + exact rules, then check every value."""
    # A LangChain "chain": the prompt template fills in the blanks, ChatOpenAI sends it
    # to the model, and JsonOutputParser turns the reply text into a Python dictionary.
    prompt = ChatPromptTemplate.from_template(SELF_QUERY_PROMPT)
    llm = ChatOpenAI(model=config.CHAT_MODEL, temperature=0)
    chain = prompt | llm | JsonOutputParser()
    try:
        raw_rules = chain.invoke({
            "question": question,
            "genders": ", ".join(GENDERS),
            "categories": ", ".join(known_categories()),
        })
    except Exception as error:
        # If the LLM reply is not valid JSON, search without filters instead of crashing.
        print("   (self-query failed, searching without filters:", error, ")")
        raw_rules = {}
    return clean_rules(raw_rules, question)


def clean_rules(raw_rules, question):
    """Check every value the LLM gave us. Anything strange becomes None (= no filter)."""
    search_text = raw_rules.get("search_text")
    if not isinstance(search_text, str) or search_text.strip() == "":
        search_text = question
    rules = {
        "search_text": search_text,
        "max_price": clean_number(raw_rules.get("max_price")),
        "min_price": clean_number(raw_rules.get("min_price")),
        "size": clean_number(raw_rules.get("size")),
        "waterproof": clean_waterproof(raw_rules.get("waterproof")),
        "gender": clean_gender(raw_rules.get("gender")),
        "category": clean_categories(raw_rules.get("category")),
    }
    return rules


def clean_number(value):
    """Return a positive whole number, or None. Accepts 5000, 5000.0 or "5,000"."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(float(str(value).replace(",", "")))
    except ValueError:
        return None
    if number <= 0:
        return None
    return number


def clean_waterproof(value):
    """Only True counts as a filter. False or anything else means "no filter"."""
    if value is True or str(value).lower() in ["true", "yes"]:
        return True
    return None


def clean_gender(value):
    """Return the gender if it is one we know, otherwise None."""
    if isinstance(value, str) and value.lower() in GENDERS:
        return value.lower()
    return None


def clean_categories(value):
    """Return a list of categories that really exist in the shop, or None."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return None
    categories = []
    for category in value:
        if isinstance(category, str) and category.lower() in known_categories():
            categories.append(category.lower())
    if len(categories) == 0:
        return None
    return categories


def allowed_genders(gender):
    """A men's or women's shopper can also wear unisex shoes. Kids get kids shoes only."""
    if gender in ["men", "women"]:
        return [gender, "unisex"]
    return [gender]


def has_rules(rules):
    """Return True if at least one exact rule is set."""
    for rule_name in ["max_price", "min_price", "size", "waterproof", "gender", "category"]:
        if rules.get(rule_name) is not None:
            return True
    return False


def build_qdrant_filter(rules):
    """Turn the rules into a Qdrant Filter. Returns None when there are no rules."""
    if not has_rules(rules):
        return None
    # LangChain stores our metadata under the payload key "metadata",
    # so the field "price" is called "metadata.price" inside Qdrant.
    conditions = []
    if rules["max_price"] is not None:
        conditions.append(FieldCondition(key="metadata.price", range=Range(lte=rules["max_price"])))
    if rules["min_price"] is not None:
        conditions.append(FieldCondition(key="metadata.price", range=Range(gte=rules["min_price"])))
    # "sizes" is a list; MatchValue matches if ANY size in the list equals the value.
    if rules["size"] is not None:
        conditions.append(FieldCondition(key="metadata.sizes", match=MatchValue(value=rules["size"])))
    if rules["waterproof"] is not None:
        conditions.append(FieldCondition(key="metadata.waterproof", match=MatchValue(value=True)))
    if rules["gender"] is not None:
        conditions.append(FieldCondition(key="metadata.gender", match=MatchAny(any=allowed_genders(rules["gender"]))))
    if rules["category"] is not None:
        conditions.append(FieldCondition(key="metadata.category", match=MatchAny(any=rules["category"])))
    # "must" means: a product must pass ALL of these conditions.
    return Filter(must=conditions)


def passes_rules(product, rules):
    """The same rules as build_qdrant_filter, written in plain Python."""
    if rules.get("max_price") is not None and product["price"] > rules["max_price"]:
        return False
    if rules.get("min_price") is not None and product["price"] < rules["min_price"]:
        return False
    if rules.get("size") is not None and rules["size"] not in product["sizes"]:
        return False
    if rules.get("waterproof") is not None and not product["waterproof"]:
        return False
    if rules.get("gender") is not None and product["gender"] not in allowed_genders(rules["gender"]):
        return False
    if rules.get("category") is not None and product["category"] not in rules["category"]:
        return False
    return True


def allowed_codes_for_rules(rules):
    """Return the set of product codes that pass the rules, or None when there are no rules."""
    if not has_rules(rules):
        return None
    allowed_codes = set()
    for product in load_products():
        if passes_rules(product, rules):
            allowed_codes.add(product["code"])
    return allowed_codes


# Small demo: turn three questions into rules and count the products that pass.
if __name__ == "__main__":
    demo_questions = [
        "Waterproof trail shoes under Rs 5,000, size 9",
        "Women's walking shoes under Rs 3,500",
        "shoes for my dad who can't tie laces",
    ]
    for step_number, demo_question in enumerate(demo_questions, start=1):
        print(f"Step {step_number}: {demo_question}")
        demo_rules = self_query(demo_question)
        print("   rules from the LLM (checked):", demo_rules)
        demo_filter = build_qdrant_filter(demo_rules)
        if demo_filter is not None:
            # Print only the field names, because the full Filter object is very long.
            filter_fields = []
            for condition in demo_filter.must:
                filter_fields.append(condition.key)
            print("   Qdrant filter on these fields:", filter_fields)
        demo_codes = allowed_codes_for_rules(demo_rules)
        if demo_codes is None:
            print("   no exact rules, so all products stay in the search\n")
        else:
            print("   products that pass the rules:", sorted(demo_codes), "\n")
