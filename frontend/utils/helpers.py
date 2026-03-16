import re
from typing import List
import streamlit as st

def extract_source_ids(answer: str) -> List[str]:
    """Extract referenced source IDs from the answer"""
    source_ids = []

    # Extract Chunk IDs
    chunks_pattern = r"Chunks':\s*\[([^\]]*)\]"
    matches = re.findall(chunks_pattern, answer)

    if matches:
        for match in matches:
            # Handle quoted IDs
            quoted_ids = re.findall(r"'([^']*)'", match)
            if quoted_ids:
                source_ids.extend(quoted_ids)
            else:
                # Handle unquoted IDs
                ids = [id.strip() for id in match.split(',') if id.strip()]
                source_ids.extend(ids)

    # Remove duplicates
    return list(set(source_ids))

def display_source_content(content: str):
    """Display source content with improved formatting"""
    st.markdown("""
    <style>
    .source-content {
        white-space: pre-wrap;
        overflow-x: auto;
        font-family: monospace;
        line-height: 1.6;
        background-color: #f5f5f5;
        border-radius: 5px;
        padding: 15px;
        max-height: 600px;
        overflow-y: auto;
        border: 1px solid #e1e4e8;
        color: #24292e;
    }
    </style>
    """, unsafe_allow_html=True)

    # Convert newlines to HTML line breaks for correct formatting
    formatted_content = content.replace("\n", "<br>")
    st.markdown(f'<div class="source-content">{formatted_content}</div>', unsafe_allow_html=True)


def process_thinking_content(content: str, show_thinking: bool = False):
    """
    Process content that contains a reasoning/thinking section.

    Args:
        content: Raw content string
        show_thinking: Whether to include the thinking section in output

    Returns:
        dict: Processed content information
    """
    if not isinstance(content, str):
        return {"processed": content, "has_thinking": False}

    # Check whether a thinking section is present
    if "<think>" in content and "</think>" in content:
        # Extract the thinking section using regex
        think_match = re.search(r'<think>(.*?)</think>', content, re.DOTALL)
        if think_match:
            thinking_process = think_match.group(1).strip()
            # Remove the thinking section, keeping only the answer
            answer_only = content.replace(f"<think>{thinking_process}</think>", "").strip()

            # Format the thinking section as Markdown block-quote
            thinking_lines = thinking_process.split('\n')
            quoted_thinking = '\n'.join([f"> {line}" for line in thinking_lines])

            return {
                "processed": answer_only,
                "has_thinking": True,
                "thinking": quoted_thinking,
                "original": content
            }

    # No thinking section found (or extraction failed) — return original content
    return {"processed": content, "has_thinking": False}