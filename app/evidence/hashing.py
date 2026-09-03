"""
app/evidence/hashing.py — Cryptographic SHA-256 fingerprinting for canonical evidence records.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional, Union

from app.blockchain.hasher import (
    calculate_fingerprint,
    create_canonical_record,
    fingerprint_from_data,
    fingerprint_from_match,
)
from app.evidence.canonical import canonical_json_bytes
from app.models.schemas import EvidenceRecord, Fingerprint, MatchResult

__all__ = [
    "canonical_json_bytes",
    "create_canonical_record",
    "calculate_fingerprint",
    "fingerprint_from_match",
    "fingerprint_from_data",
]
