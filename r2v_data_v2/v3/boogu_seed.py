from __future__ import annotations

import secrets


def new_boogu_seed() -> int:
    """Return a fresh seed for one independent Boogu generation request."""

    return secrets.randbelow(2**31)
