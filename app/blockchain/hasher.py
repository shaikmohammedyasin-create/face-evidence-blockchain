"""
app/blockchain/hasher.py — Deterministic SHA-256 fingerprinting & canonical provenance.

The same input data always produces the exact same byte sequence and fingerprint,
which enables transparent, independent on-chain verification.

Biometric privacy guarantee:
  - Personal biometric vectors (embeddings) are NEVER included in the fingerprint.
  - No personal names or raw facial images are stored on-chain.
  - The record is a cryptographic proof of the public post metadata and match parameters.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Union

from app.models.schemas import EvidenceRecord, Fingerprint, MatchResult
from app.utils.file_utils import sha256_file
from app.utils.logger import get_logger

log = get_logger(__name__)


def canonical_json_bytes(data: dict[str, Any]) -> bytes:
    """
    Serialise *data* deterministically to UTF-8 bytes.

    Canonical encoding rules (RFC 8785 JSON Canonicalization Scheme principles):
      1. Keys are recursively sorted in lexicographical order.
      2. No extraneous whitespace between elements (separators `(",", ":")`).
      3. UTF-8 character encoding with standard JSON escaping.
    """
    return json.dumps(data, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")


# Alias for backwards compatibility with tests
_canonical = canonical_json_bytes


def create_canonical_record(
    match: MatchResult,
    input_image_path: Optional[Union[Path, str]] = None,
    input_image_sha256: Optional[str] = None,
    timestamp_iso: Optional[str] = None,
) -> EvidenceRecord:
    """
    Construct a formal EvidenceRecord from a match result.

    Args:
        match: The MatchResult object containing candidate metadata and confidence.
        input_image_path: Optional path to the input image to calculate its SHA-256 digest.
        input_image_sha256: Optional precomputed SHA-256 digest of the input image.
        timestamp_iso: Optional fixed timestamp (defaults to current UTC ISO 8601).
    """
    candidate = match.candidate
    img_hash = input_image_sha256 or ""
    if not img_hash and input_image_path:
        try:
            img_hash = sha256_file(input_image_path)
        except Exception as exc:
            log.debug("Could not compute image SHA-256: %s", exc)

    ts = timestamp_iso or datetime.now(timezone.utc).isoformat()

    match_metadata = {
        "url": candidate.url,
        "platform": candidate.platform,
        "title": candidate.title,
        "text": candidate.snippet,
        "image_url": candidate.image_url,
    }

    return EvidenceRecord(
        schema_version="1.0",
        timestamp_iso=ts,
        input_image_sha256=img_hash,
        match_metadata=match_metadata,
        match_confidence=float(match.confidence),
        face_detection_model="YuNet-2023mar",
        face_recognition_model="SFace-2021dec",
        raw_metadata=getattr(candidate, "raw_metadata", {}),
    )


def calculate_fingerprint(record_or_data: Union[EvidenceRecord, dict[str, Any]]) -> Fingerprint:
    """
    Calculate a deterministic SHA-256 fingerprint from an EvidenceRecord or dictionary.

    Returns a :class:`Fingerprint` object containing the digest, canonical JSON, and source data.
    """
    if isinstance(record_or_data, EvidenceRecord):
        source_dict = record_or_data.to_canonical_dict()
        evidence_rec = record_or_data
    else:
        source_dict = record_or_data
        evidence_rec = None

    raw_bytes = canonical_json_bytes(source_dict)
    digest = hashlib.sha256(raw_bytes).hexdigest()
    canonical_str = raw_bytes.decode("utf-8")

    log.debug("Computed fingerprint: %s (canonical len=%d bytes)", digest, len(raw_bytes))

    return Fingerprint(
        algorithm="SHA-256",
        fingerprint=digest,
        source_data=source_dict,
        canonical_json=canonical_str,
        evidence_record=evidence_rec,
    )


def fingerprint_from_match(
    result: MatchResult,
    input_image_path: Optional[Union[Path, str]] = None,
) -> Fingerprint:
    """
    Build a SHA-256 fingerprint from the matched post's public metadata.

    Preserved for backwards compatibility. Uses standard post metadata fields.
    """
    candidate = result.candidate
    source_data = {
        "url": candidate.url,
        "platform": candidate.platform,
        "title": candidate.title,
        "text": candidate.snippet,
        "image_url": candidate.image_url,
    }

    raw_bytes = canonical_json_bytes(source_data)
    digest = hashlib.sha256(raw_bytes).hexdigest()

    log.info("Generated fingerprint: %s…  (source keys: %s)", digest[:16], sorted(source_data.keys()))
    return Fingerprint(
        algorithm="SHA-256",
        fingerprint=digest,
        source_data=source_data,
        canonical_json=raw_bytes.decode("utf-8"),
    )


def fingerprint_from_data(source_data: dict[str, Any]) -> Fingerprint:
    """
    Build a fingerprint directly from a dict of source data.

    Useful for re-verification (re-hashing the exact same dictionary).
    """
    return calculate_fingerprint(source_data)
