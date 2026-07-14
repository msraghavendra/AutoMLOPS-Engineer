import sys
sys.path.insert(0, '.')
from app.config import settings
key = settings.GEMINI_API_KEY
print("GEMINI_KEY_SET:", bool(key))
print("Key prefix:", key[:12] if key else "EMPTY")
