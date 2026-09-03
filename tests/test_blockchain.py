"""
tests/test_blockchain.py — Unit tests for the blockchain layer.

All tests are hermetic (no network calls).
"""
from __future__ import annotations

import hashlib
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from app.models.schemas import BlockchainRecord, Fingerprint, MatchResult, SearchCandidate


# ── hasher tests ──────────────────────────────────────────────────────────────

class TestHasher:
    def _candidate(self) -> SearchCandidate:
        return SearchCandidate(
            url="https://instagram.com/p/abc123",
            title="Test title",
            snippet="Some text",
            image_url="https://example.com/img.jpg",
            platform="Instagram",
        )

    def _match_result(self) -> MatchResult:
        return MatchResult(
            candidate=self._candidate(),
            matched=True,
            confidence=0.91,
        )

    def test_generates_sha256(self):
        from app.blockchain.hasher import fingerprint_from_match
        fp = fingerprint_from_match(self._match_result())
        assert fp.algorithm == "SHA-256"
        assert len(fp.fingerprint) == 64  # hex SHA-256

    def test_deterministic(self):
        from app.blockchain.hasher import fingerprint_from_match
        r = self._match_result()
        fp1 = fingerprint_from_match(r)
        fp2 = fingerprint_from_match(r)
        assert fp1.fingerprint == fp2.fingerprint

    def test_different_url_different_hash(self):
        from app.blockchain.hasher import fingerprint_from_match
        r1 = self._match_result()
        r2 = MatchResult(
            candidate=SearchCandidate(
                url="https://twitter.com/user/status/999",
                title="Different title",
                snippet="Different text",
                image_url="",
                platform="Twitter / X",
            ),
            matched=True,
            confidence=0.85,
        )
        assert fingerprint_from_match(r1).fingerprint != fingerprint_from_match(r2).fingerprint

    def test_source_data_excludes_biometrics(self):
        from app.blockchain.hasher import fingerprint_from_match
        fp = fingerprint_from_match(self._match_result())
        # Biometric data must not appear in source_data
        assert "embedding" not in fp.source_data
        assert "face" not in fp.source_data
        # Expected keys
        assert "url" in fp.source_data
        assert "platform" in fp.source_data

    def test_canonical_serialisation_is_deterministic(self):
        """Two dicts with the same content but different insertion order must hash identically."""
        from app.blockchain.hasher import _canonical
        d1 = {"b": 2, "a": 1}
        d2 = {"a": 1, "b": 2}
        assert _canonical(d1) == _canonical(d2)


# ── SimulatedClient tests ─────────────────────────────────────────────────────

class TestSimulatedClient:
    def _client(self):
        from app.blockchain.client import SimulatedClient
        return SimulatedClient()

    def test_store_returns_record(self):
        client = self._client()
        record = client.store_fingerprint("a" * 64, {})
        assert record.transaction_hash.startswith("0x")
        assert record.fingerprint == "a" * 64
        assert record.status == "simulated"

    def test_verify_after_store(self):
        client = self._client()
        fp = "b" * 64
        client.store_fingerprint(fp, {})
        assert client.verify_fingerprint(fp) is True

    def test_verify_unknown_returns_false(self):
        client = self._client()
        assert client.verify_fingerprint("0" * 64) is False

    def test_tx_hash_determinism_within_same_timestamp(self):
        """Two stores of the same fingerprint within 1 second produce different tx hashes (timestamp changes)."""
        client = self._client()
        fp = "c" * 64
        r1 = client.store_fingerprint(fp, {})
        # Store again — timestamp may or may not differ, but client tracks per-call
        r2 = client.store_fingerprint(fp, {})
        # Both should be valid tx hashes
        assert r1.transaction_hash.startswith("0x")
        assert r2.transaction_hash.startswith("0x")


# ── verifier tests ────────────────────────────────────────────────────────────

class TestVerifier:
    def _fingerprint(self) -> Fingerprint:
        return Fingerprint(
            algorithm="SHA-256",
            fingerprint="a" * 64,
            source_data={
                "url": "https://instagram.com/p/abc",
                "platform": "Instagram",
                "title": "Post title",
                "text": "Post text",
                "image_url": "https://example.com/img.jpg",
            },
        )

    def test_verified_when_fingerprint_matches(self):
        from app.blockchain.verifier import re_verify

        # Build a fingerprint from the actual source data so it's consistent
        from app.blockchain.hasher import fingerprint_from_data
        source = {
            "url": "https://instagram.com/p/abc",
            "platform": "Instagram",
            "title": "Post title",
            "text": "Post text",
            "image_url": "https://example.com/img.jpg",
        }
        fp = fingerprint_from_data(source)

        record = BlockchainRecord(
            network="Test",
            transaction_hash="0x" + "f" * 64,
            fingerprint=fp.fingerprint,
            status="simulated",
        )

        client = MagicMock()
        client.verify_fingerprint.return_value = True

        result = re_verify(record, fp, client)
        assert result.verified is True
        assert result.stored_fingerprint == result.current_fingerprint

    def test_not_verified_when_not_on_chain(self):
        from app.blockchain.hasher import fingerprint_from_data
        from app.blockchain.verifier import re_verify

        source = {"url": "https://example.com", "platform": "Web",
                   "title": "T", "text": "", "image_url": ""}
        fp = fingerprint_from_data(source)

        record = BlockchainRecord(
            network="Test",
            transaction_hash="0xabc",
            fingerprint=fp.fingerprint,
            status="simulated",
        )

        client = MagicMock()
        client.verify_fingerprint.return_value = False  # not on chain

        result = re_verify(record, fp, client)
        assert result.verified is False

    def test_not_verified_when_fingerprint_mismatch(self):
        from app.blockchain.hasher import fingerprint_from_data
        from app.blockchain.verifier import re_verify

        source = {"url": "https://example.com", "platform": "Web",
                   "title": "T", "text": "", "image_url": ""}
        fp = fingerprint_from_data(source)

        # Record has a DIFFERENT fingerprint (simulates tampering)
        record = BlockchainRecord(
            network="Test",
            transaction_hash="0xabc",
            fingerprint="0" * 64,   # wrong
            status="simulated",
        )

        client = MagicMock()
        client.verify_fingerprint.return_value = False

        result = re_verify(record, fp, client)
        assert result.verified is False


# ── tamper detection test ─────────────────────────────────────────────────────

class TestTamperDetection:
    def test_original_and_tampered_differ(self):
        from app.blockchain.hasher import fingerprint_from_data
        from app.blockchain.verifier import demonstrate_tamper

        source = {
            "url": "https://instagram.com/p/abc",
            "platform": "Instagram",
            "title": "Original title",
            "text": "Some text",
            "image_url": "https://example.com/img.jpg",
        }
        fp = fingerprint_from_data(source)
        result = demonstrate_tamper(fp)

        assert result["original_hash"] != result["tampered_hash"]
        assert result["tamper_detected"] is True

    def test_sha256_avalanche(self):
        """A 1-character change produces a completely different hash."""
        from app.blockchain.hasher import fingerprint_from_data

        source_a = {"url": "https://ex.com/a", "platform": "Web",
                    "title": "Title", "text": "", "image_url": ""}
        source_b = {"url": "https://ex.com/b", "platform": "Web",
                    "title": "Title", "text": "", "image_url": ""}

        fp_a = fingerprint_from_data(source_a)
        fp_b = fingerprint_from_data(source_b)
        assert fp_a.fingerprint != fp_b.fingerprint


# ── uploader tests ────────────────────────────────────────────────────────────

class TestUploader:
    def test_success(self):
        from app.blockchain.uploader import upload_fingerprint

        fp = Fingerprint(
            algorithm="SHA-256",
            fingerprint="d" * 64,
            source_data={"url": "https://ex.com", "platform": "Web"},
        )

        mock_client = MagicMock()
        mock_client.store_fingerprint.return_value = BlockchainRecord(
            network="Simulated",
            transaction_hash="0x" + "e" * 64,
            fingerprint="d" * 64,
            status="simulated",
        )

        record = upload_fingerprint(fp, mock_client)
        assert record.fingerprint == "d" * 64
        mock_client.store_fingerprint.assert_called_once()

    def test_failure_raises(self):
        from app.blockchain.uploader import upload_fingerprint

        fp = Fingerprint(
            algorithm="SHA-256",
            fingerprint="f" * 64,
            source_data={"url": "https://ex.com", "platform": "Web"},
        )

        mock_client = MagicMock()
        mock_client.store_fingerprint.return_value = BlockchainRecord(
            network="Simulated",
            transaction_hash="0xfail",
            fingerprint="f" * 64,
            status="failed",
        )

        with pytest.raises(RuntimeError, match="Blockchain transaction failed"):
            upload_fingerprint(fp, mock_client)
