"""
Eval module configuration — reads from environment variables.

Required env vars:
  OPENAI_API_KEY          — OpenAI key used by both runners and the judge
  NEO4J_URI               — neo4j+s://... for the Aura instance
  NEO4J_USERNAME          — Aura username
  NEO4J_PASSWORD          — Aura password
  SEARCH_SERVICE_URL      — deployed Cloud Run URL, e.g. https://search-service-xxx.run.app
  SEARCH_SERVICE_TOKEN    — gcloud auth print-identity-token (for Cloud Run auth)

Optional:
  JUDGE_MODEL             — model used for LLM-as-judge (default: gpt-4o)
  GRAPHRAG_MODEL          — production model under test (default: gpt-4.1-nano)
  COMPARE_MODEL           — model to compare against (default: gpt-4o)
  EVAL_TOP_K              — number of naive chunks to retrieve (default: 5)
"""

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
SEARCH_SERVICE_URL = os.environ["SEARCH_SERVICE_URL"].rstrip("/")
SEARCH_SERVICE_TOKEN = os.environ.get("SEARCH_SERVICE_TOKEN", "")

JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-4o")
GRAPHRAG_MODEL = os.environ.get("GRAPHRAG_MODEL", "gpt-4.1-nano")
COMPARE_MODEL = os.environ.get("COMPARE_MODEL", "gpt-4o")
EVAL_TOP_K = int(os.environ.get("EVAL_TOP_K", "5"))
