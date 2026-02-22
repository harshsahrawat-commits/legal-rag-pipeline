from __future__ import annotations

import hashlib


def content_hash(data: str | bytes) -> str:
    """Return the SHA-256 hex digest of the given content."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()
