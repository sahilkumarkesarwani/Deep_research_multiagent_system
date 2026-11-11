"""
auth.py — Authentication helpers: password hashing, registration, login,
           and Streamlit session-state helpers.
"""

import re

import bcrypt
import streamlit as st
from sqlalchemy.exc import IntegrityError

from database import create_user, get_user_by_email, get_user_by_username, init_db

init_db()


def hash_password(plain: str) -> str:
    """Return a bcrypt hash of the given plaintext password."""
    return bcrypt.hashpw(plain.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    """Return True if plain matches the stored bcrypt hash."""
    return bcrypt.checkpw(plain.encode(), hashed.encode())

USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,30}$")
EMAIL_RE    = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def validate_registration(username: str, email: str, password: str, confirm: str):
    """Return a list of error strings, empty list means valid."""
    errors = []
    if not USERNAME_RE.match(username):
        errors.append("Username must be 3–30 characters and contain only letters, digits, or underscores.")
    if not EMAIL_RE.match(email):
        errors.append("Please enter a valid email address.")
    if len(password) < 8:
        errors.append("Password must be at least 8 characters.")
    if password != confirm:
        errors.append("Passwords do not match.")
    return errors

def register_user(username: str, email: str, password: str, confirm: str) -> tuple[bool, str]:
    """
    Validate inputs, hash password, and persist user.
    Returns (success: bool, message: str).
    """
    errors = validate_registration(username.strip(), email.strip(), password, confirm)
    if errors:
        return False, "\n".join(f"• {e}" for e in errors)

    if get_user_by_username(username.strip()):
        return False, "• Username is already taken."
    if get_user_by_email(email.strip()):
        return False, "• An account with that email already exists."

    try:
        create_user(
            username=username.strip(),
            email=email.strip().lower(),
            password_hash=hash_password(password),
        )
        return True, "Account created successfully! You can now log in."
    except IntegrityError:
        return False, "• That username or email is already in use."
    except Exception as e:
        return False, f"• Unexpected error: {e}"


def login_user(username_or_email: str, password: str) -> tuple[bool, str]:
    """
    Authenticate a user by username or email + password.
    On success, writes user info into st.session_state and returns (True, "").
    On failure, returns (False, error_message).
    """
    identifier = username_or_email.strip()
    user = (
        get_user_by_username(identifier)
        or get_user_by_email(identifier.lower())
    )
    if user is None:
        return False, "No account found with that username or email."
    if not verify_password(password, user.password_hash):
        return False, "Incorrect password."

    st.session_state["auth_user_id"]   = user.id
    st.session_state["auth_username"]  = user.username
    st.session_state["auth_email"]     = user.email
    st.session_state["auth_logged_in"] = True
    return True, ""

def is_logged_in() -> bool:
    return st.session_state.get("auth_logged_in", False)


def current_user_id() -> int | None:
    return st.session_state.get("auth_user_id")


def current_username() -> str:
    return st.session_state.get("auth_username", "")


def logout():
    """Clear all auth and research state from the session."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()
