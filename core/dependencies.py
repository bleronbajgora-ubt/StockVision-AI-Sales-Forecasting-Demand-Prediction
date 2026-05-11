"""
=============================================================================
 app/core/dependencies.py — FastAPI Dependency Injection
=============================================================================

from typing import List, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import decode_access_token
from app.db.session import get_db
from app.models.orm import User
from app.repositories.repositories import UserRepository

# OAuth2PasswordBearer tells FastAPI where to find the token.
# tokenUrl is the login endpoint (used by Swagger UI's "Authorize" button).
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


# ─────────────────────────────────────────────────────────────────────────────
# CORE AUTH DEPENDENCY
# ─────────────────────────────────────────────────────────────────────────────
async def get_current_user(
    token: str              = Depends(oauth2_scheme),
    db:    AsyncSession     = Depends(get_db),
) -> User:
    """
    Validate JWT and return the authenticated User.

    Flow:
    1. Extract Bearer token from Authorization header.
    2. Decode and verify JWT signature + expiry.
    3. Extract user_id from 'sub' claim.
    4. Load User from DB (ensures user still exists and is_active).
    5. Return User or raise HTTP 401.

    WHY DB lookup on every request:
      The token's 'sub' claim only tells us the user_id at token-issue time.
      A deactivated user would still have a valid token until expiry.
      The DB lookup catches this in ~1ms (indexed primary key lookup).
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload  = decode_access_token(token)
        user_id  = int(payload.get("sub", 0))
        if not user_id:
            raise credentials_exception
    except (JWTError, ValueError):
        raise credentials_exception

    repo = UserRepository(db)
    user = await repo.get_by_id(user_id)

    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated.",
        )
    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Alias — explicitly documents that the endpoint requires an active user."""
    return current_user


# ─────────────────────────────────────────────────────────────────────────────
# ROLE-BASED ACCESS CONTROL FACTORY
# ─────────────────────────────────────────────────────────────────────────────
def require_roles(*required_roles: str):
    """
    Dependency factory that checks the current user has at least one of the
    required roles embedded in the JWT.

    WHY read roles from JWT (not DB):
      For authorization (what you can do), we read from the token's 'roles'
      claim — a single in-memory list check with zero DB queries.
      For authentication (who you are), we always check the DB (see above).

    Usage:
        @router.delete("/users/{id}")
        async def delete_user(
            user = Depends(require_roles("admin"))
        ):

    Multi-role (any of):
        @router.post("/products")
        async def create(
            user = Depends(require_roles("admin", "moderator"))
        ):
    """
    async def checker(
        token: str          = Depends(oauth2_scheme),
        db:    AsyncSession = Depends(get_db),
    ) -> User:
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
        try:
            payload     = decode_access_token(token)
            user_id     = int(payload.get("sub", 0))
            token_roles = payload.get("roles", [])
        except (JWTError, ValueError):
            raise credentials_exception

        # Check role membership
        if not any(r in token_roles for r in required_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access requires one of these roles: {list(required_roles)}",
            )

        repo = UserRepository(db)
        user = await repo.get_by_id(user_id)
        if not user or not user.is_active:
            raise credentials_exception
        return user

    return checker


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE ROLE DEPENDENCIES
# ─────────────────────────────────────────────────────────────────────────────
require_admin    = require_roles("admin")
require_analyst  = require_roles("admin", "analyst")
require_moderator = require_roles("admin", "moderator")


# ─────────────────────────────────────────────────────────────────────────────
# OPTIONAL AUTH (for endpoints that work for both guests and logged-in users)
# ─────────────────────────────────────────────────────────────────────────────
async def get_optional_user(
    request: Request,
    db:      AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Returns the current user if authenticated, None if not.
    Used for endpoints like /predict that work for guests but log user_id if available.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header[7:]
    try:
        payload = decode_access_token(token)
        user_id = int(payload.get("sub", 0))
        repo    = UserRepository(db)
        return await repo.get_by_id(user_id)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PAGINATION DEPENDENCY
# ─────────────────────────────────────────────────────────────────────────────
class PaginationParams:
    """
    Reusable pagination dependency.
    Caps limit at 100 to prevent "SELECT * FROM reviews LIMIT 999999" attacks.
    """
    def __init__(self, skip: int = 0, limit: int = 20):
        self.skip  = max(0, skip)
        self.limit = min(limit, 100)