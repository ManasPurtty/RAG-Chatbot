"""
auth.py - JWT Authentication and Role-Based Access Control (RBAC)

Provides:
- User management (stored in users.json)
- Password hashing with bcrypt
- JWT token creation and verification
- FastAPI dependency for current user extraction
- Admin-only role guard
"""

import json
import os
import secrets
from datetime import datetime, timedelta
from typing import Optional

from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from models import UserInDB, TokenData, UserRole


# ── Config ─────────────────────────────────────────────────────────────────────
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_hex(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60
USERS_FILE = "users.json"

# pbkdf2_sha256 is built into passlib (no bcrypt version conflicts)
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# ── User Store ─────────────────────────────────────────────────────────────────

def load_users() -> dict:
    """Load users from JSON file. Creates default admin on first run."""
    if not os.path.exists(USERS_FILE):
        default_admin_password = os.getenv("ADMIN_PASSWORD", "admin123")
        default_users = {
            "admin": {
                "username": "admin",
                "hashed_password": pwd_context.hash(default_admin_password),
                "role": "admin",
            }
        }
        with open(USERS_FILE, "w") as f:
            json.dump(default_users, f, indent=2)
        return default_users

    with open(USERS_FILE, "r") as f:
        return json.load(f)


def save_users(users: dict) -> None:
    """Persist user store to disk."""
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


# ── Password Helpers ───────────────────────────────────────────────────────────

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


# ── Auth Logic ─────────────────────────────────────────────────────────────────

def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    """Return UserInDB if credentials are valid, else None."""
    users = load_users()
    if username not in users:
        # Constant-time check to prevent username enumeration
        pwd_context.verify(password, pwd_context.hash("dummy"))
        return None
    user_data = users[username]
    if not verify_password(password, user_data["hashed_password"]):
        return None
    return UserInDB(**user_data)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ── FastAPI Dependencies ───────────────────────────────────────────────────────

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    """Dependency: extract and validate JWT, return current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    users = load_users()
    if token_data.username not in users:
        raise credentials_exception

    return UserInDB(**users[token_data.username])


def require_admin(current_user: UserInDB = Depends(get_current_user)) -> UserInDB:
    """Dependency: raises 403 if user is not an admin."""
    if current_user.role != UserRole.admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required for this action.",
        )
    return current_user
