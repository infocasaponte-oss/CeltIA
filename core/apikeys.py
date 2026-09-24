# Copyright (c) 2026 Luis Manuel Cousido Hermida. All rights reserved.
import hashlib
import secrets

PREFIX = "ck-live-"


def generate_key() -> tuple[str, str, str]:
    """Returns (raw_key, key_hash, display_prefix)."""
    raw = PREFIX + secrets.token_hex(24)
    key_hash = hash_key(raw)
    display_prefix = raw[: len(PREFIX) + 6] + "…"
    return raw, key_hash, display_prefix


def generate_client_key() -> tuple[str, str, str]:
    """Secondary keys for external clients. Returns (raw_key, key_hash, display_prefix)."""
    raw = "sk_live_" + secrets.token_urlsafe(32)
    return raw, hash_key(raw), raw[:12] + "…"


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def hash_password(raw: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", raw.encode("utf-8"), salt.encode("utf-8"), 200_000)
    return f"{salt}${digest.hex()}"


def verify_password(raw: str, stored: str) -> bool:
    try:
        salt, _ = stored.split("$", 1)
    except ValueError:
        return False
    return secrets.compare_digest(hash_password(raw, salt), stored)
