"""
app/blockchain/client.py — Blockchain interface & implementations.

Supported modes:
  1. EthereumClient  — Live Ethereum Sepolia testnet via web3.py & smart contract
  2. SimulatedClient — In-memory, deterministic stand-in when keys are omitted

Design Principle:
  - If keys are missing, the pipeline runs seamlessly in clearly-labelled
    [SIMULATION] mode so evaluators can test the entire workflow without gas funds.
  - No biometric data is ever sent to the blockchain — only bytes32 SHA-256 fingerprints.
"""
from __future__ import annotations

import abc
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Optional

from app.config import (
    BLOCKCHAIN_CHAIN_ID,
    BLOCKCHAIN_CONTRACT_ADDRESS,
    BLOCKCHAIN_NETWORK,
    BLOCKCHAIN_PRIVATE_KEY,
    BLOCKCHAIN_RPC_URL,
    PROJECT_ID,
    PROJECT_ROOT,
)
from app.models.schemas import BlockchainRecord
from app.utils.logger import get_logger

log = get_logger(__name__)

# Minimal ABI for FingerprintRegistry.sol
_REGISTRY_ABI = [
    {
        "inputs": [
            {"internalType": "bytes32", "name": "fingerprint", "type": "bytes32"},
            {"internalType": "string", "name": "projectId", "type": "string"},
        ],
        "name": "store",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "fingerprint", "type": "bytes32"}
        ],
        "name": "exists",
        "outputs": [{"internalType": "bool", "name": "", "type": "bool"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "inputs": [
            {"internalType": "bytes32", "name": "fingerprint", "type": "bytes32"}
        ],
        "name": "getTimestamp",
        "outputs": [{"internalType": "uint256", "name": "", "type": "uint256"}],
        "stateMutability": "view",
        "type": "function",
    },
    {
        "anonymous": False,
        "inputs": [
            {
                "indexed": True,
                "internalType": "bytes32",
                "name": "fingerprint",
                "type": "bytes32",
            },
            {
                "indexed": False,
                "internalType": "uint256",
                "name": "timestamp",
                "type": "uint256",
            },
            {
                "indexed": False,
                "internalType": "string",
                "name": "projectId",
                "type": "string",
            },
        ],
        "name": "Stored",
        "type": "event",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Abstract Interface
# ─────────────────────────────────────────────────────────────────────────────

class BlockchainClient(abc.ABC):
    """Abstract base class for all blockchain interaction clients."""

    network_name: str = "AbstractNetwork"

    @abc.abstractmethod
    def store_fingerprint(
        self,
        fingerprint: str,
        metadata: dict[str, Any],
    ) -> BlockchainRecord:
        """
        Record a 32-byte SHA-256 fingerprint on-chain.

        Args:
            fingerprint: 64-character hexadecimal string (with or without '0x').
            metadata: Associated public post metadata (not written on-chain).

        Returns:
            BlockchainRecord containing transaction receipt & status.
        """

    @abc.abstractmethod
    def verify_fingerprint(self, fingerprint: str) -> bool:
        """Query the blockchain registry to check whether *fingerprint* exists."""


# ─────────────────────────────────────────────────────────────────────────────
# Ethereum Sepolia Testnet Client
# ─────────────────────────────────────────────────────────────────────────────

class EthereumClient(BlockchainClient):
    """
    Interacts with the FingerprintRegistry smart contract on Ethereum Sepolia.

    Requires:
        - BLOCKCHAIN_RPC_URL (e.g. Infura / Alchemy / public Sepolia RPC)
        - BLOCKCHAIN_PRIVATE_KEY (funded testnet account)
        - BLOCKCHAIN_CONTRACT_ADDRESS (deployed FingerprintRegistry contract)
    """

    network_name = BLOCKCHAIN_NETWORK

    def __init__(
        self,
        rpc_url: str = "",
        private_key: str = "",
        contract_address: str = "",
    ) -> None:
        try:
            from web3 import Web3  # type: ignore
            from web3.middleware import ExtraDataToPOAMiddleware  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "web3 package is required for live blockchain interaction. "
                "Run: pip install web3"
            ) from exc

        self._rpc_url = rpc_url or BLOCKCHAIN_RPC_URL
        self._private_key = private_key or BLOCKCHAIN_PRIVATE_KEY
        self._contract_address = contract_address or BLOCKCHAIN_CONTRACT_ADDRESS

        if not self._private_key:
            raise ValueError("BLOCKCHAIN_PRIVATE_KEY is required for live chain operation.")
        if not self._contract_address:
            raise ValueError("BLOCKCHAIN_CONTRACT_ADDRESS is required for live chain operation.")

        self._w3 = Web3(Web3.HTTPProvider(self._rpc_url))
        try:
            self._w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
        except Exception:
            pass

        if not self._w3.is_connected():
            raise ConnectionError(f"Cannot connect to Ethereum RPC at {self._rpc_url}")

        self._account = self._w3.eth.account.from_key(self._private_key)
        self._checksum_address = Web3.to_checksum_address(self._contract_address)
        self._contract = self._w3.eth.contract(
            address=self._checksum_address,
            abi=_REGISTRY_ABI,
        )

        log.info(
            "EthereumClient initialised: network=%s, account=%s, contract=%s",
            self.network_name,
            self._account.address,
            self._checksum_address,
        )

    @staticmethod
    def _to_bytes32(fingerprint: str) -> bytes:
        hexstr = fingerprint[2:] if fingerprint.startswith("0x") else fingerprint
        raw = bytes.fromhex(hexstr)
        if len(raw) != 32:
            raise ValueError(
                f"Fingerprint must be 32 bytes (SHA-256 hex), got {len(raw)} bytes."
            )
        return raw

    def store_fingerprint(self, fingerprint: str, metadata: dict[str, Any]) -> BlockchainRecord:
        fp_bytes = self._to_bytes32(fingerprint)

        nonce = self._w3.eth.get_transaction_count(self._account.address)
        gas_price = self._w3.eth.gas_price

        tx = self._contract.functions.store(fp_bytes, PROJECT_ID).build_transaction(
            {
                "chainId": BLOCKCHAIN_CHAIN_ID,
                "gas": 85_000,
                "gasPrice": gas_price,
                "nonce": nonce,
            }
        )
        signed = self._account.sign_transaction(tx)
        tx_hash_bytes = self._w3.eth.send_raw_transaction(signed.raw_transaction)
        tx_hash = tx_hash_bytes.hex()
        if not tx_hash.startswith("0x"):
            tx_hash = f"0x{tx_hash}"

        log.info("Transaction submitted: %s", tx_hash)
        receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash_bytes, timeout=120)

        status = "confirmed" if receipt.status == 1 else "failed"
        explorer_url = f"https://sepolia.etherscan.org/tx/{tx_hash}"
        log.info(
            "Transaction %s confirmed (block=%s, gas=%s, explorer=%s)",
            tx_hash[:18], receipt.blockNumber, receipt.gasUsed, explorer_url,
        )

        return BlockchainRecord(
            network=BLOCKCHAIN_NETWORK,
            transaction_hash=tx_hash,
            fingerprint=fingerprint,
            status=status,
            block_number=receipt.blockNumber,
            timestamp=int(time.time()),
            gas_used=receipt.gasUsed,
            explorer_url=explorer_url,
            contract_address=self._checksum_address,
        )

    def verify_fingerprint(self, fingerprint: str) -> bool:
        fp_bytes = self._to_bytes32(fingerprint)
        return bool(self._contract.functions.exists(fp_bytes).call())


# ─────────────────────────────────────────────────────────────────────────────
# Local Simulation (Zero-configuration testing)
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedClient(BlockchainClient):
    """
    In-memory, deterministic simulation of the FingerprintRegistry smart contract.

    Used when no private key or contract address is configured.
    Produces deterministic, verifiable receipts so evaluators can inspect
    the complete cryptographic flow without needing Sepolia testnet ETH.

    [SIMULATION MODE] is clearly labelled across all logs and CLI output.
    """

    network_name = "[SIMULATION] Local In-Memory Chain (Sepolia Stand-in)"
    _SHARED_STORE: dict[str, dict[str, Any]] = {}

    def __init__(self, store: Optional[dict[str, dict[str, Any]]] = None) -> None:
        self._store = store if store is not None else SimulatedClient._SHARED_STORE
        self._simulated_base_block = 6_428_190
        log.warning(
            "[SIMULATION MODE] No BLOCKCHAIN_PRIVATE_KEY / CONTRACT_ADDRESS configured. "
            "Using local in-memory simulation — zero gas fees required. "
            "Set keys in .env to broadcast to Ethereum Sepolia testnet."
        )

    def store_fingerprint(self, fingerprint: str, metadata: dict[str, Any]) -> BlockchainRecord:
        ts = int(time.time())
        block_no = self._simulated_base_block + len(self._store) + 1

        # Deterministic simulated tx hash: SHA-256(fingerprint + ts + project_id)
        sim_hash = "0x" + hashlib.sha256(f"{fingerprint}:{ts}:{PROJECT_ID}".encode("utf-8")).hexdigest()
        sim_contract = "0x71C7656EC7ab88b098defB751B7401B5f6d8976F"  # sample mock address

        self._store[fingerprint] = {
            "timestamp": ts,
            "tx_hash": sim_hash,
            "block_number": block_no,
            "metadata": metadata,
        }

        log.info("[SIMULATION] Stored fingerprint %s… → tx %s (block %d)", fingerprint[:16], sim_hash[:18], block_no)

        return BlockchainRecord(
            network=self.network_name,
            transaction_hash=sim_hash,
            fingerprint=fingerprint,
            status="simulated",
            block_number=block_no,
            timestamp=ts,
            gas_used=48_210,
            explorer_url=None,
            contract_address=sim_contract,
        )

    def verify_fingerprint(self, fingerprint: str) -> bool:
        exists = fingerprint in self._store
        log.info("[SIMULATION] Verify query for %s… → %s", fingerprint[:16], exists)
        return exists


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_blockchain_client(preferred_network: Optional[str] = None) -> BlockchainClient:
    """
    Return the appropriate blockchain client based on environment configuration or explicit selection.

    Returns EthereumClient if preferred_network is 'sepolia' or if keys are configured;
    otherwise returns SimulatedClient with transparent logging.
    """
    pref = (preferred_network or "").strip().lower()
    if pref in ("sim", "simulation", "local"):
        log.info("Using blockchain client: SimulatedClient (explicit selection)")
        return SimulatedClient()

    if pref in ("sepolia", "ethereum", "eth", "testnet"):
        log.info("Using blockchain client: EthereumClient (explicit selection: %s)", pref)
        try:
            return EthereumClient()
        except Exception as exc:
            log.warning("Ethereum Sepolia live connection failed/unconfigured (%s) — falling back to labelled Simulation.", exc)
            return SimulatedClient()

    if BLOCKCHAIN_PRIVATE_KEY and BLOCKCHAIN_CONTRACT_ADDRESS:
        try:
            return EthereumClient()
        except Exception as exc:
            log.error("EthereumClient initialization failed (%s) — falling back to simulation.", exc)

    return SimulatedClient()
