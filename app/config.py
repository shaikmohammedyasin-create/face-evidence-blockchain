"""
app/config.py — Centralised configuration from environment variables.

All modules import from here instead of reading os.environ directly.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Ensure system CA certificates are injected on Windows/macOS if truststore is available
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

# Load .env from project root (parent of app/)
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env", override=False)


def _require(name: str) -> str:
    """Return the value of *name* or raise a clear error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


def _optional(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


# ── Face ─────────────────────────────────────────────────────
# OpenCV Zoo models: YuNet (detector) + SFace (128-d recogniser)
# Official SFace cosine-similarity threshold is 0.363
# (two faces are the same person if cosine_similarity >= threshold).
# Increase to reduce false positives; decrease to reduce false negatives.
FACE_MATCH_THRESHOLD: float = float(_optional("FACE_MATCH_THRESHOLD", "0.363"))

# ── Search ───────────────────────────────────────────────────
SERPAPI_KEY: str = _optional("SERPAPI_KEY", "")
BING_SEARCH_API_KEY: str = _optional("BING_SEARCH_API_KEY", "")
BING_SEARCH_ENDPOINT: str = _optional(
    "BING_SEARCH_ENDPOINT",
    "https://api.bing.microsoft.com/v7.0/images/visualsearch",
)

# ── Blockchain ───────────────────────────────────────────────
BLOCKCHAIN_RPC_URL: str = _optional("BLOCKCHAIN_RPC_URL", "https://rpc.sepolia.org")
BLOCKCHAIN_PRIVATE_KEY: str = _optional("BLOCKCHAIN_PRIVATE_KEY", "")
BLOCKCHAIN_CONTRACT_ADDRESS: str = _optional("BLOCKCHAIN_CONTRACT_ADDRESS", "")
BLOCKCHAIN_NETWORK: str = _optional("BLOCKCHAIN_NETWORK", "Ethereum Sepolia Testnet")
BLOCKCHAIN_CHAIN_ID: int = int(_optional("BLOCKCHAIN_CHAIN_ID", "11155111"))

# ── Application ──────────────────────────────────────────────
LOG_LEVEL: str = _optional("LOG_LEVEL", "INFO")
PROJECT_ID: str = _optional("PROJECT_ID", "HH-GOA-2026-TASK3")

# ── Paths ────────────────────────────────────────────────────
PROJECT_ROOT: Path = _project_root
DATA_INPUT: Path = _project_root / "data" / "input"
DATA_RESULTS: Path = _project_root / "data" / "results"
DATA_MODELS: Path = _project_root / "data" / "models"
