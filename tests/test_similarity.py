"""
tests/test_similarity.py — Unit tests for the authoritative cosine similarity implementation.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from app.face.similarity import cosine_distance, cosine_similarity, is_valid_embedding, l2_normalize


class TestAuthoritativeCosineSimilarity:
    def test_self_identity(self):
        v = np.random.randn(128).tolist()
        sim = cosine_similarity(v, v)
        assert abs(sim - 1.0) < 1e-9, f"Self similarity must be exactly 1.0, got {sim}"

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0, 0.0]
        sim = cosine_similarity(a, b)
        assert abs(sim - 0.0) < 1e-9

    def test_opposite_vectors(self):
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        sim = cosine_similarity(a, b)
        assert abs(sim - (-1.0)) < 1e-9

    def test_zero_vector_safety(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0
        assert cosine_similarity(b, a) == 0.0

    def test_empty_vector_safety(self):
        assert cosine_similarity([], [1.0]) == 0.0

    def test_dimension_mismatch(self):
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

    def test_commutative_symmetry(self):
        a = np.random.randn(128)
        b = np.random.randn(128)
        sim_ab = cosine_similarity(a, b)
        sim_ba = cosine_similarity(b, a)
        assert abs(sim_ab - sim_ba) < 1e-12

    def test_l2_normalize(self):
        v = np.array([3.0, 4.0])
        normed = l2_normalize(v)
        assert abs(np.linalg.norm(normed) - 1.0) < 1e-9
        assert abs(normed[0] - 0.6) < 1e-6
        assert abs(normed[1] - 0.8) < 1e-6

    def test_cosine_distance(self):
        v = [1.0, 0.0]
        assert abs(cosine_distance(v, v) - 0.0) < 1e-9
        assert abs(cosine_distance([1.0, 0.0], [-1.0, 0.0]) - 2.0) < 1e-9

    def test_is_valid_embedding(self):
        valid = np.random.randn(128).tolist()
        assert is_valid_embedding(valid, expected_dim=128) is True
        assert is_valid_embedding([0.0] * 128, expected_dim=128) is False
        assert is_valid_embedding(valid[:64], expected_dim=128) is False
        assert is_valid_embedding([float("nan")] * 128, expected_dim=128) is False
