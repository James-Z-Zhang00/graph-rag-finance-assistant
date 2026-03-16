"""
API key authentication dependency.

Clients must send:  Authorization: Bearer <GATEWAY_API_KEY>

If GATEWAY_API_KEY is empty in .env, auth is disabled (useful for local dev).
"""

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from config.settings import GATEWAY_API_KEY

_bearer = HTTPBearer(auto_error=False)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> None:
    if not GATEWAY_API_KEY:
        # Auth disabled — dev mode
        return

    if credentials is None or credentials.credentials.strip() != GATEWAY_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
