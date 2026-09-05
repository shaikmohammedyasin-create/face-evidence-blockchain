"""
app/models/schemas.py — Shared data models (plain dataclasses, no ORM).

Defines structured entities for:
  - Face detection & embedding outputs
  - Visual search candidates & match results
  - Canonical evidence & cryptographic provenance records
  - Blockchain transaction receipts & independent verification proofs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FaceResult:
    """Output of the face-detection / encoding stage."""
    faces_detected: int
    selected_face: int
    embedding_generated: bool
    embedding_dimension: int
    embedding: list[float] = field(default_factory=list, repr=False)
    landmarks: list[tuple[float, float]] = field(default_factory=list, repr=False)
    bounding_box: dict[str, int] = field(default_factory=dict)
    detection_confidence: float = 1.0


@dataclass
class SearchCandidate:
    """A single URL / post returned by the search provider."""
    url: str
    title: str = ""
    snippet: str = ""
    image_url: str = ""
    platform: str = "unknown"
    raw_metadata: dict = field(default_factory=dict, repr=False)


@dataclass
class IdentityDecision:
    """
    Forensic three-tier decision output separating raw ML cosine similarity
    from final application-level identity determinations.
    """
    verdict: str                  # "HIGH CONFIDENCE MATCH" | "REVIEW / UNCERTAIN" | "NO MATCH"
    tier: str                     # "HIGH_MATCH" | "REVIEW" | "NO_MATCH"
    is_match: bool                # True ONLY for HIGH CONFIDENCE MATCH
    confidence: float             # Primary cosine similarity score
    margin: float                 # Delta over runner-up from distinct cluster
    runner_up_confidence: float   # Similarity of runner-up candidate
    matched_face_index: int       # 0-indexed face chosen in candidate image
    candidate_faces_count: int    # Total faces detected in candidate image
    quality_passed: bool          # Whether candidate face met forensic quality thresholds
    quality_score: float          # Combined quality score 0.0 - 1.0
    quality_details: dict[str, Any] = field(default_factory=dict)
    decision_reason: str = ""     # Human-readable forensic rationale
    warnings: list[str] = field(default_factory=list)
    match_factors: dict[str, Any] = field(default_factory=dict)
    z_score: float = 0.0          # Per-query statistical z-score above candidate pool mean
    pool_mean: float = 0.0        # Mean similarity score of search candidate pool
    pool_std: float = 0.0         # Standard deviation of candidate pool scores
    is_statistical_outlier: bool = True  # True if score exceeds pool mean + k*std


@dataclass
class MatchResult:
    """Result of comparing a candidate image against the input face."""
    candidate: SearchCandidate
    matched: bool
    confidence: float          # cosine similarity 0–1
    local_image_path: str = ""
    candidate_embedding_dimension: int = 128
    candidate_embedding: list[float] = field(default_factory=list, repr=False)
    matched_face_index: int = 0
    candidate_faces_count: int = 1
    matched_face_box: dict[str, int] = field(default_factory=dict)
    matched_face_landmarks: list[tuple[float, float]] = field(default_factory=list)
    quality_score: float = 1.0
    quality_details: dict[str, Any] = field(default_factory=dict)
    decision_tier: str = "NO_MATCH"
    decision_reason: str = ""
    margin_from_runner_up: float = 0.0
    image_dhash: str = ""
    image_relationship: str = ""
    warnings: list[str] = field(default_factory=list)
    all_face_scores: list[float] = field(default_factory=list)
    identity_decision: Optional[IdentityDecision] = None


@dataclass
class EvidenceRecord:
    """
    Formal canonical provenance record linking the search match and biometric match
    to a verifiable cryptographic representation.

    Schema Version: 1.0
    Biometric privacy guarantee: zero raw biometric vectors or personal identity
    labels are stored in this record.
    """
    schema_version: str = "1.0"
    timestamp_iso: str = ""
    input_image_sha256: str = ""
    match_metadata: dict[str, Any] = field(default_factory=dict)
    match_confidence: float = 0.0
    face_detection_model: str = "YuNet-2023mar"
    face_recognition_model: str = "SFace-2021dec"
    raw_metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return a clean dictionary representation suitable for canonical serialisation."""
        return {
            "schema_version": self.schema_version,
            "timestamp_iso": self.timestamp_iso,
            "input_image_sha256": self.input_image_sha256,
            "match_metadata": {
                "url": self.match_metadata.get("url", ""),
                "platform": self.match_metadata.get("platform", ""),
                "title": self.match_metadata.get("title", ""),
                "text": self.match_metadata.get("text", "") or self.match_metadata.get("snippet", ""),
                "image_url": self.match_metadata.get("image_url", ""),
            },
            "match_confidence": round(float(self.match_confidence), 4),
            "face_detection_model": self.face_detection_model,
            "face_recognition_model": self.face_recognition_model,
        }


@dataclass
class Fingerprint:
    """
    Deterministic SHA-256 fingerprint of a matched post and evidence record.
    """
    algorithm: str
    fingerprint: str
    source_data: dict = field(default_factory=dict, repr=False)
    canonical_json: str = ""
    evidence_record: Optional[EvidenceRecord] = None


@dataclass
class BlockchainRecord:
    """On-chain record returned by the blockchain uploader."""
    network: str
    transaction_hash: str
    fingerprint: str
    status: str                # "confirmed" | "pending" | "failed" | "simulated"
    block_number: Optional[int] = None
    timestamp: Optional[int] = None
    gas_used: Optional[int] = None
    explorer_url: Optional[str] = None
    contract_address: Optional[str] = None


@dataclass
class VerificationResult:
    """Outcome of re-verifying a fingerprint against the blockchain."""
    transaction_hash: str
    stored_fingerprint: str
    current_fingerprint: str
    verified: bool
    note: str = ""
    network: str = ""
    block_number: Optional[int] = None
    timestamp: Optional[int] = None
