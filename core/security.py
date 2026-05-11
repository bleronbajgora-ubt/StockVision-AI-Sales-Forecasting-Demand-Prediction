"""
=============================================================================
 app/core/security.py — Authentication & Cryptography
=============================================================================

import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.settings import settings

# sha256_crypt is a well-regarded, POSIX-standard slow hash available on every
# platform without compiled C extensions. In production, swap to bcrypt/argon2.
pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")


def hash_password(plain_password: str) -> str:
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(
    subject: Any,
    roles: list[str] | None = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    now    = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    payload: Dict[str, Any] = {
        "sub":   str(subject),
        "roles": roles or [],
        "exp":   expire,
        "iat":   now,
        "type":  "access",
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


def decode_access_token(token: str) -> Dict[str, Any]:
    return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])


def generate_refresh_token() -> str:
    return secrets.token_urlsafe(64)


def hash_refresh_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_refresh_token_pair() -> tuple[str, str]:
    raw    = generate_refresh_token()
    hashed = hash_refresh_token(raw)
    return raw, hashed