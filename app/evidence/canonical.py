"""
app/evidence/canonical.py — RFC 8785 JSON Canonicalization Scheme (JCS) encoding.

Ensures deterministic serialization so that equivalent dictionaries produce the exact
same byte sequence across different platforms and runtimes.
"""
from __future__ import annotations

import json
from typing import Any


def canonical_json_bytes(data: dict[str, Any]) -> bytes:
    """
    Serialize *data* deterministically to UTF-8 bytes using RFC 8785 principles:
      1. Recursively sort all object keys in lexicographical order.
      2. Use minimal separators `(",", ":")` with zero unnecessary whitespace.
      3. UTF-8 character encoding with standard JSON escaping.
    """
    return json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


def canonical_json_str(data: dict[str, Any]) -> str:
    """Return the deterministic canonical JSON string."""
    return canonical_json_bytes(data).decode("utf-8")
