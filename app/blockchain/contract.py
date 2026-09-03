"""
app/blockchain/contract.py — Smart contract interaction interface for FingerprintRegistry.

Interacts with the Ethereum Sepolia / EVM smart contract or in-memory simulation client.
"""
from __future__ import annotations

from typing import Any, Optional

from app.blockchain.client import BlockchainClient, build_blockchain_client
from app.blockchain.uploader import upload_fingerprint
from app.models.schemas import BlockchainRecord, Fingerprint

__all__ = [
    "BlockchainClient",
    "build_blockchain_client",
    "upload_fingerprint",
]
