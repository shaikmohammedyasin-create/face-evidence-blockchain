"""
app/search/provenance.py — Candidate search provenance tracking and forensic audit records.

Records the full discovery lineage of every candidate examined by the pipeline:
  - Discovery provider (e.g. SerpAPI Google Lens)
  - Search rank
  - Canonical and raw page URLs
  - Direct image thumbnail URLs
  - Image cryptographic hash (SHA-256)
  - Detected face count & matched face index
  - Forensic similarity & decision tier
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class CandidateProvenance:
    source: str
    search_rank: int
    page_url: str
    image_url: str
    title: str
    domain: str
    platform: str
    retrieved_at: str
    candidate_image_sha256: str = ""
    detected_faces: int = 0
    selected_face_index: int = 0
    face_similarity: float = 0.0
    runner_up_margin: float = 0.0
    decision: str = "NO_RELIABLE_MATCH"
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def create_candidate_provenance(
    match_result: Any,
    search_rank: int = 1,
    provider_name: str = "Google Lens via SerpAPI",
    image_sha256: str = "",
) -> CandidateProvenance:
    """Build a structured provenance record from a MatchResult."""
    cand = match_result.candidate
    return CandidateProvenance(
        source=provider_name,
        search_rank=search_rank,
        page_url=cand.url,
        image_url=cand.image_url,
        title=cand.title,
        domain=cand.platform,
        platform=cand.platform,
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        candidate_image_sha256=image_sha256,
        detected_faces=getattr(match_result, "candidate_faces_count", 1),
        selected_face_index=getattr(match_result, "matched_face_index", 0),
        face_similarity=float(match_result.confidence),
        runner_up_margin=float(getattr(match_result, "margin_from_runner_up", 0.0)),
        decision=getattr(match_result, "decision_tier", "NO_MATCH"),
        raw_metadata=getattr(cand, "raw_metadata", {}),
    )
