"""
Build service configuration.
"""

import os
from dotenv import load_dotenv

load_dotenv()

BUILD_SERVICE_HOST = os.getenv("BUILD_SERVICE_HOST", "0.0.0.0")
BUILD_SERVICE_PORT = int(os.getenv("BUILD_SERVICE_PORT", "8004"))
BUILD_SERVICE_RELOAD = os.getenv("BUILD_SERVICE_RELOAD", "false").lower() == "true"
BUILD_SERVICE_LOG_LEVEL = os.getenv("BUILD_SERVICE_LOG_LEVEL", "info")

# Directory where uploaded source files are stored (used by incremental builds)
FILES_DIR = os.getenv("FILES_DIR", "./files")

# File registry path for incremental updates
FILE_REGISTRY_PATH = os.getenv("FILE_REGISTRY_PATH", "./file_registry.json")

# SEC parser service URL
SEC_PARSER_URL = os.getenv("SEC_PARSER_URL", "http://localhost:8001")

# Directory of source files to pass to sec-parser for full builds (local fallback)
SEC_FILES_DIR = os.getenv("SEC_FILES_DIR", "../static-files/small")

# GCS — if GCS_BUCKET_NAME is set, full builds download from GCS instead of SEC_FILES_DIR
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", "")
GCS_FILES_PREFIX = os.getenv("GCS_FILES_PREFIX", "small/")
