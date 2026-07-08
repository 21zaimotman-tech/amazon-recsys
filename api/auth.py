"""Password hashing for the demo accounts system. PBKDF2-HMAC-SHA256 (stdlib
only, no new dependency) with a random per-user salt and 200k iterations --
plenty for a course-project login, nowhere near what a real production
auth system would additionally need (rate limiting, breach checks, etc.)."""
import hashlib
import hmac
import os

_ITERATIONS = 200_000


def hash_password(password: str) -> tuple[str, str]:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return digest.hex(), salt.hex()


def verify_password(password: str, password_hash: str, password_salt: str) -> bool:
    salt = bytes.fromhex(password_salt)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _ITERATIONS)
    return hmac.compare_digest(digest.hex(), password_hash)
