"""
app/face/matcher.py — Forensic cosine-similarity face matching & multi-face scanning.

Compares an input target face embedding against candidate images found via visual search.
Key capabilities:
  1. Full multi-face scanning: Detects and aligns every face in candidate images,
     evaluating cosine similarity for each and tracking the best matched face index.
  2. Candidate Deduplication: Normalizes URLs and image SHA-256 hashes to prevent
     redundant evaluations.
  3. Quality Assessment: Flags degraded, blurry, or low-resolution face crops.
  4. Decision Engine Integration: Assigns 3-tier forensic verdicts (HIGH_MATCH, REVIEW, NO_MATCH)
     with margin analysis.

Statistical Disclaimer:
  Match results represent algorithmic cosine-similarity estimates on 128-d
  feature vectors, not legal or infallible proof of human identity.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import cv2
import numpy as np

from app.config import FACE_MATCH_THRESHOLD
from app.face.identity_decision import (
    BASELINE_MATCH_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    evaluate_identity_decision,
)
from app.face.quality import FaceQualityAssessment, assess_face_quality
from app.models.schemas import FaceResult, MatchResult, SearchCandidate
from app.utils.file_utils import compute_dhash, hamming_distance
from app.utils.logger import get_logger

log = get_logger(__name__)


def normalize_url(url: str) -> str:
    """
    Normalize URL by removing tracking query params, fragments, and standardizing host/scheme.
    """
    if not url:
        return ""
    try:
        parsed = urlparse(url.strip())
        # Strip common tracking query params
        tracking_keys = {
            "utm_source", "utm_medium", "utm_campaign", "utm_term",
            "utm_content", "fbclid", "gclid", "ref", "source", "igshid",
        }
        qs = parse_qs(parsed.query, keep_blank_values=False)
        clean_qs = {k: v for k, v in qs.items() if k.lower() not in tracking_keys}
        clean_query = urlencode(clean_qs, doseq=True)

        clean_path = parsed.path.rstrip("/")
        normalized = urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            clean_path,
            "",
            clean_query,
            "",
        ))
        return normalized
    except Exception:
        return url.strip()


def deduplicate_candidates(candidates: list[SearchCandidate]) -> list[SearchCandidate]:
    """
    Remove duplicate search candidates based on normalized URL and image URL.
    Preserves original discovery order.
    """
    seen_urls: set[str] = set()
    seen_img_urls: set[str] = set()
    deduped: list[SearchCandidate] = []

    for c in candidates:
        norm_url = normalize_url(c.url)
        norm_img = normalize_url(c.image_url)

        if norm_url and norm_url in seen_urls:
            continue
        if norm_img and norm_img in seen_img_urls:
            continue

        if norm_url:
            seen_urls.add(norm_url)
        if norm_img:
            seen_img_urls.add(norm_img)

        deduped.append(c)

    return deduped


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Return the cosine similarity between two 128-d embedding vectors (range: -1.0 to 1.0).

    For L2-normalised vectors, cosine similarity equals the dot product.
    """
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    norm_a = np.linalg.norm(va)
    norm_b = np.linalg.norm(vb)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    sim = float(np.dot(va, vb) / (norm_a * norm_b))
    return max(-1.0, min(1.0, sim))


def match_candidate(
    input_face: FaceResult,
    candidate: SearchCandidate,
    candidate_image_path: Path | str,
) -> MatchResult:
    """
    Compare *input_face* embedding against every face found in *candidate_image_path*.

    Scans all candidate faces, measures face quality, performs 5-point affine alignment,
    generates 128-d SFace embeddings, and records full forensic diagnostics.
    """
    from app.face.detector import detect_faces
    from app.face.encoder import generate_embedding

    path_str = str(candidate_image_path)
    img = cv2.imread(path_str)
    if img is None:
        log.warning("Could not read candidate image file: %s", path_str)
        return MatchResult(
            candidate=candidate,
            matched=False,
            confidence=0.0,
            local_image_path=path_str,
            decision_tier="NO_MATCH",
            decision_reason=f"Failed to read candidate image at {path_str}",
        )

    try:
        cand_faces = detect_faces(candidate_image_path)
    except (ValueError, FileNotFoundError, Exception) as exc:
        log.warning("No faces detected in candidate image %s: %s", candidate_image_path, exc)
        return MatchResult(
            candidate=candidate,
            matched=False,
            confidence=0.0,
            local_image_path=path_str,
            candidate_faces_count=0,
            decision_tier="NO_MATCH",
            decision_reason="No human face detected in candidate image",
        )

    cand_dhash = compute_dhash(candidate_image_path)

    best_similarity = -1.0
    best_face_idx = 0
    best_quality: Optional[FaceQualityAssessment] = None
    best_cand_embedding: list[float] = []
    all_scores: list[float] = []

    for idx, face_meta in enumerate(cand_faces):
        quality = assess_face_quality(img, face_meta)
        try:
            cand_face = generate_embedding(candidate_image_path, face_index=idx)
            sim = cosine_similarity(input_face.embedding, cand_face.embedding)
            cand_emb = cand_face.embedding
        except Exception as exc:
            log.debug("Could not encode candidate face %d in %s: %s", idx, path_str, exc)
            sim = 0.0
            cand_emb = []

        all_scores.append(round(sim, 4))
        if sim > best_similarity:
            best_similarity = sim
            best_face_idx = idx
            best_quality = quality
            best_cand_embedding = cand_emb

    similarity = max(0.0, best_similarity)
    selected_meta = cand_faces[best_face_idx]
    fa = selected_meta.get("facial_area", {})
    landmarks = selected_meta.get("landmarks", [])

    matched = similarity >= BASELINE_MATCH_THRESHOLD

    quality_score = best_quality.score if best_quality else 1.0
    quality_details = best_quality.to_dict() if best_quality else {}
    warnings = list(best_quality.reasons) if best_quality else []

    if len(cand_faces) > 1:
        warnings.append(
            f"Multi-face image: Evaluated {len(cand_faces)} faces; highest similarity on Face #{best_face_idx + 1}"
        )

    rel_type = "NEAR_DUPLICATE_PHOTO" if (matched and cand_dhash) else ("HIGH_SIMILARITY_FACE" if matched else "NO_RELATION")

    return MatchResult(
        candidate=candidate,
        matched=matched,
        confidence=similarity,
        local_image_path=path_str,
        candidate_embedding_dimension=len(input_face.embedding),
        candidate_embedding=best_cand_embedding,
        matched_face_index=best_face_idx,
        candidate_faces_count=len(cand_faces),
        matched_face_box=fa,
        matched_face_landmarks=landmarks,
        quality_score=quality_score,
        quality_details=quality_details,
        decision_tier="HIGH_MATCH" if similarity >= HIGH_CONFIDENCE_THRESHOLD else ("REVIEW" if matched else "NO_MATCH"),
        decision_reason="",
        margin_from_runner_up=0.0,
        image_dhash=cand_dhash,
        image_relationship=rel_type,
        warnings=warnings,
        all_face_scores=all_scores,
    )


def is_near_duplicate(r1: MatchResult, r2: MatchResult, threshold: float = 0.90) -> bool:
    """
    Return True if candidates r1 and r2 represent near-duplicate face images / same identity cluster.
    Compares perceptual image dhash hamming distance and embedding cosine similarity.
    """
    if r1.image_dhash and r2.image_dhash:
        dist = hamming_distance(r1.image_dhash, r2.image_dhash)
        if dist <= 10:
            return True
    if r1.candidate_embedding and r2.candidate_embedding:
        return cosine_similarity(r1.candidate_embedding, r2.candidate_embedding) >= threshold
    return abs(r1.confidence - r2.confidence) < 0.005



def rank_candidates(results: list[MatchResult]) -> list[MatchResult]:
    """
    Sort match results descending by cosine similarity confidence score,
    perform near-duplicate candidate clustering, compute non-cluster runner-up margins,
    and apply 3-tier forensic identity decisions with pool z-score evaluation.
    """
    if not results:
        return []

    sorted_results = sorted(results, key=lambda r: r.confidence, reverse=True)
    pool_scores = [r.confidence for r in sorted_results]

    # Cluster candidates by embedding similarity (near-duplicate grouping)
    clusters: list[list[MatchResult]] = []
    for res in sorted_results:
        assigned = False
        for cluster in clusters:
            if is_near_duplicate(cluster[0], res, threshold=0.90):
                cluster.append(res)
                assigned = True
                break
        if not assigned:
            clusters.append([res])

    # Map candidate instance to its assigned cluster
    candidate_to_cluster: dict[int, list[MatchResult]] = {}
    for cluster in clusters:
        for item in cluster:
            candidate_to_cluster[id(item)] = cluster

    # Clustered pool scores (highest score per distinct candidate cluster)
    clustered_pool_scores = [c[0].confidence for c in clusters]

    # Calculate runner-up margin against candidates NOT in the top candidate's cluster
    for rank_idx, match_res in enumerate(sorted_results):
        my_cluster = candidate_to_cluster.get(id(match_res), [match_res])
        current_domain = urlparse(match_res.candidate.url).netloc.lower()

        # Find highest-scoring candidate belonging to a DIFFERENT cluster
        runner_up_sim = 0.0
        runner_up_domain = ""

        for other in sorted_results:
            if other is match_res or other in my_cluster:
                continue
            if other.confidence > 0.0:
                runner_up_sim = other.confidence
                runner_up_domain = urlparse(other.candidate.url).netloc.lower()
                break

        margin = max(0.0, match_res.confidence - runner_up_sim) if runner_up_sim > 0 else match_res.confidence
        match_res.margin_from_runner_up = margin

        # Re-evaluate identity decision with margin and statistical pool context
        quality_obj = None
        if match_res.quality_details:
            quality_obj = FaceQualityAssessment(
                is_acceptable=match_res.quality_details.get("is_acceptable", True),
                score=match_res.quality_details.get("score", 1.0),
                width=match_res.quality_details.get("width", 50),
                height=match_res.quality_details.get("height", 50),
                laplacian_variance=match_res.quality_details.get("laplacian_variance", 50.0),
                mean_brightness=match_res.quality_details.get("mean_brightness", 128.0),
                contrast_std=match_res.quality_details.get("contrast_std", 30.0),
                interocular_dist_ratio=match_res.quality_details.get("interocular_dist_ratio", 0.35),
                detection_confidence=match_res.quality_details.get("detection_confidence", 1.0),
                reasons=match_res.quality_details.get("reasons", []),
                quality_label=match_res.quality_details.get("quality_label", "Good"),
            )

        decision = evaluate_identity_decision(
            best_similarity=match_res.confidence,
            runner_up_similarity=runner_up_sim if rank_idx == 0 else 0.0,
            quality=quality_obj,
            matched_face_index=match_res.matched_face_index,
            candidate_faces_count=match_res.candidate_faces_count,
            candidate_domain=current_domain,
            runner_up_domain=runner_up_domain,
            candidate_pool_scores=clustered_pool_scores,
        )

        match_res.identity_decision = decision
        match_res.decision_tier = decision.tier
        match_res.decision_reason = decision.decision_reason
        match_res.matched = decision.is_match or (decision.tier == "REVIEW" and match_res.confidence >= BASELINE_MATCH_THRESHOLD)

    return sorted_results
