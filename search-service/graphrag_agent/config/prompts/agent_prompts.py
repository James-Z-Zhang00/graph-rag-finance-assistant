"""
Centralized agent-layer prompt template definitions, maintaining consistent prompt content across different agents.
"""

from textwrap import dedent

GRAPH_AGENT_KEYWORD_PROMPT = dedent(
    """
    Extract keywords from the following query:
    Query: {query}

    Please extract two types of keywords:
    1. Low-level keywords: specific entities, names, terms
    2. High-level keywords: themes, concepts, domains

    Return in JSON format.
    """
).strip()

GRAPH_AGENT_GENERATE_PROMPT = dedent(
    """
    ---Analysis Report---
    Note: the analysis reports provided below are sorted in **descending order of importance**.

    {context}

    The user's question is:
    {question}

    Please output the answer strictly in the following format:
    1. Use level-3 headings (###) to mark topics
    2. Present main content in clear paragraphs
    3. At the end, use "#### Citation Data" to mark the citation section, listing data sources used
    """
).strip()

GRAPH_AGENT_REDUCE_PROMPT = dedent(
    """
    ---Analysis Report---
    {report_data}

    The user's question is:
    {question}
    """
).strip()

DEEP_RESEARCH_THINKING_SUMMARY_PROMPT = dedent(
    """
    The following is the thinking process for the question:

    {thinking}

    The original question is:
    {question}

    Please generate a comprehensive, in-depth answer. Do not repeat the thinking process; provide the final synthesized conclusion directly.
    The conclusion should clearly and directly answer the question, including relevant facts and insights.
    If there are different viewpoints or conflicting information, point them out and provide a balanced perspective.
    """
).strip()

EXPLORATION_SUMMARY_PROMPT = dedent(
    """
    Based on the following knowledge graph exploration paths and discovered content, generate a comprehensive summary about "{query}":

    Exploration paths:
    {path_summary}

    Key content:
    {content_summary}

    Please provide a comprehensive, in-depth analysis including key findings, associations, and insights.
    """
).strip()

CONTRADICTION_IMPACT_PROMPT = dedent(
    """
    While answering the question about "{query}", the following information contradictions were found:

    {contradictions_text}

    Please analyze the potential impact of these contradictions on the final answer, and how to provide the most accurate answer given these contradictions.
    """
).strip()

HYBRID_AGENT_GENERATE_PROMPT = dedent(
    """
    ---Analysis Report---
    The following is retrieved relevant information, sorted by importance:

    {context}

    The user's question is:
    {question}

    Please answer the question in a clear and comprehensive manner, ensuring:
    1. The answer combines retrieved low-level (entity details) and high-level (thematic concepts) information
    2. Use level-3 headings (###) to organize content for improved readability
    3. At the end, use "#### Citation Data" to mark citation sources
    """
).strip()

NAIVE_RAG_HUMAN_PROMPT = dedent(
    """
    ---Retrieved Results---
    {context}

    Question:
    {question}
    """
)
