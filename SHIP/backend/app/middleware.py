from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from app.config import settings

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    """
    Dependency to verify API Key header.
    Validates X-API-Key header matches the configured key.
    """
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"API Key is missing. Please provide it in the '{API_KEY_NAME}' header."
        )
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key."
        )
    return api_key
