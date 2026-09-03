"""
app/face/similarity.py — Authoritative mathematical cosine similarity & distance metrics.

Ensures strict vector arithmetic guarantees:
  - Identical normalized unit vectors: dot(u, u) = 1.0 (100% self-match)
  - Orthogonal vectors: dot(u, v) ≈ 0.0
  - Opposite vectors: dot(u, -u) = -1.0
  - Commutativity / Symmetry: sim(u, v) == sim(v, u)
  - Safe handling of zero norms (returns 0.0, no NaN or ZeroDivisionError)
"""
from __future__ import annotations

import math
from typing import Sequence, Union

import numpy as np


def cosine_similarity(
    a: Union[Sequence[float], np.ndarray],
    b: Union[Sequence[float], np.ndarray],
) -> float:
    """
    Compute the cosine similarity between two feature vectors:
      cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)

    For 128-d L2-normalized vectors (||a|| = 1, ||b|| = 1), this reduces to the dot product.

    Returns:
        float in the closed interval [-1.0, 1.0].
    """
    va = np.asarray(a, dtype=np.float64).flatten()
    vb = np.asarray(b, dtype=np.float64).flatten()

    if va.size == 0 or vb.size == 0 or va.shape != vb.shape:
        return 0.0

    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    dot_product = float(np.dot(va, vb))
    raw_sim = dot_product / (norm_a * norm_b)

    # Clamp to avoid floating point precision overshoot
    clamped = max(-1.0, min(1.0, raw_sim))
    return float(clamped)


def cosine_distance(
    a: Union[Sequence[float], np.ndarray],
    b: Union[Sequence[float], np.ndarray],
) -> float:
    """
    Compute cosine distance: 1.0 - cosine_similarity(a, b).
    Ranges from 0.0 (identical) to 2.0 (opposite).
    """
    return 1.0 - cosine_similarity(a, b)


def l2_normalize(v: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """
    Project a vector onto the unit hypersphere so that ||v||_2 = 1.0.
    """
    arr = np.asarray(v, dtype=np.float64).flatten()
    norm = float(np.linalg.norm(arr))
    if norm > 0.0:
        return arr / norm
    return arr


def euclidean_distance(
    a: Union[Sequence[float], np.ndarray],
    b: Union[Sequence[float], np.ndarray],
) -> float:
    """
    Compute standard Euclidean distance (L2 norm) between two embedding vectors.
    For unit vectors: dist = sqrt(2 * (1 - cosine_similarity)).
    """
    va = np.asarray(a, dtype=np.float64).flatten()
    vb = np.asarray(b, dtype=np.float64).flatten()
    if va.shape != vb.shape:
        return float("inf")
    return float(np.linalg.norm(va - vb))


def is_valid_embedding(
    v: Union[Sequence[float], np.ndarray],
    expected_dim: int = 128,
) -> bool:
    """
    Check if vector is non-empty, contains finite numbers, and has expected dimensionality.
    """
    arr = np.asarray(v, dtype=np.float64).flatten()
    if arr.size != expected_dim:
        return False
    if not np.all(np.isfinite(arr)):
        return False
    if float(np.linalg.norm(arr)) == 0.0:
        return False
    return True
