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
    You are an assistant specialized in extracting search keywords from user queries for SEC financial filings. Extract the following fields:

    1. low_level: specific entity names, financial metrics, products, people, etc.
    2. high_level: themes, concepts, relationship types, trends.
    3. companies: For each company mentioned include BOTH the stock ticker AND a short lowercase name variant so filename matching works regardless of format — e.g. ["WMT", "walmart"] for Walmart, ["AAPL", "apple"] for Apple, ["JPM", "jpmorgan"] for JPMorgan. Use lowercase for the name variant.
    4. period_end: the reporting period end date in ISO format YYYY-MM-DD when explicitly stated. "October 31, 2025" → "2025-10-31". Null if no specific date.
    5. period_type: the duration of the reporting period — "annual" (full year / 10-K), "quarterly" (single quarter, 3 months), "ytd" (year-to-date, e.g. nine months). Null if not clear.
    6. filing_type: "10k" for annual report, "10q" for quarterly report. Null if not specified.
    7. section: the SEC filing section explicitly referenced — "mda" (MD&A / management discussion), "risk_factors" (Item 1A), "financials" (financial statements / Item 8), "business" (Item 1 business overview). Null if not specified.
    8. fiscal_year: four-digit year (e.g. 2024) when the user mentions a fiscal year without an explicit date. Null if a specific date is already captured in period_end.

    The return format must be JSON:
    {{
        "low_level": ["keyword1", "keyword2", ...],
        "high_level": ["keyword1", "keyword2", ...],
        "companies": ["WMT", "walmart"],
        "period_end": "2025-10-31",
        "period_type": "ytd",
        "filing_type": "10q",
        "section": null,
        "fiscal_year": null
    }}

    Notes:
    - Extract 3-5 keywords per low_level and high_level
    - Do not add any explanations or other text, return JSON only
    - Set a field to null when it cannot be determined from the query
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
