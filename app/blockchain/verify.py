"""
app/blockchain/verify.py — Independent blockchain re-verification module.
"""
from __future__ import annotations

from app.blockchain.verifier import re_verify
from app.models.schemas import BlockchainRecord, EvidenceRecord, Fingerprint, VerificationResult

__all__ = ["re_verify", "VerificationResult"]
