"""
Unified configuration for search tool prompt templates.
"""

from textwrap import dedent

LOCAL_SEARCH_CONTEXT_PROMPT = dedent(
    """
    ---Analysis Report---
    Note: the analysis reports provided below are sorted in **descending order of importance**.

    {context}

    The user's question is:
    {input}

    Please use level-3 headings (###) to mark topics
    """
).strip()

LOCAL_SEARCH_KEYWORD_PROMPT = dedent(
    """
    You are an assistant specialized in extracting search keywords from user queries. Classify keywords into two categories:
    1. Low-level keywords: specific entity names, people, places, specific events, etc.
    2. High-level keywords: themes, concepts, relationship types, etc.

    The return format must be JSON:
    {{
        "low_level": ["keyword1", "keyword2", ...],
        "high_level": ["keyword1", "keyword2", ...]
    }}

    Notes:
    - Extract 3-5 keywords per category
    - Do not add any explanations or other text, return JSON only
    - If there are no keywords for a category, return an empty list
    """.strip()
)

GLOBAL_SEARCH_MAP_PROMPT = dedent(
    """
    ---Data Table---
    {context_data}

    The user's question is:
    {question}
    """
).strip()

GLOBAL_SEARCH_REDUCE_PROMPT = dedent(
    """
    ---Analysis Report---
    {report_data}

    The user's question is:
    {question}
    """
).strip()

GLOBAL_SEARCH_KEYWORD_PROMPT = dedent(
    """
    You are an assistant specialized in extracting search keywords from user queries. Extract the most relevant keywords to find information in the knowledge base.

    Return a keyword list in JSON array format:
    ["keyword1", "keyword2", ...]

    Notes:
    - Extract 5-8 keywords
    - Do not add any explanations or other text, return JSON array only
    - Keywords should be noun phrases, concepts, or proper nouns
    """
).strip()

HYBRID_TOOL_QUERY_PROMPT = dedent(
    """
    ---Analysis Report---
    Note: the following content combines low-level details, high-level thematic concepts, and structured numeric facts.

    ## Low-level Content (Entity Details):
    {low_level}

    ## High-level Content (Themes and Concepts):
    {high_level}

    ## Numeric Facts (Structured Financial Data from Filings):
    {numeric_facts}

    ## Filing Sections (SEC Document Sections):
    {filing_sections}

    The user's question is:
    {query}

    Please synthesize the above information to answer the question, ensuring the answer is comprehensive and in-depth.
    When numeric facts are available, use them as the authoritative source for specific figures.
    The answer format should include:
    1. Main content (presented in clear paragraphs)
    2. Citation data sources at the end
    """
).strip()

NAIVE_SEARCH_QUERY_PROMPT = dedent(
    """
    ---Document Chunks---
    {context}

    Question:
    {query}
    """
).strip()
