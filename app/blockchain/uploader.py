"""
app/blockchain/uploader.py — Upload a fingerprint to the blockchain.

Thin wrapper that calls the client and enriches the returned record
with the source metadata for later verification reference.
"""
from __future__ import annotations

from app.blockchain.client import BlockchainClient
from app.models.schemas import BlockchainRecord, Fingerprint
from app.utils.logger import get_logger

log = get_logger(__name__)


def upload_fingerprint(
    fp: Fingerprint,
    client: BlockchainClient,
) -> BlockchainRecord:
    """
    Store *fp* on the blockchain via *client*.

    Returns:
        BlockchainRecord with network / transaction_hash / status.

    Raises:
        RuntimeError  – on chain-level failure (propagated from client).
    """
    log.info(
        "Uploading fingerprint %s… to %s",
        fp.fingerprint[:16],
        client.__class__.__name__,
    )

    metadata = {
        "source_url": fp.source_data.get("url", ""),
        "platform": fp.source_data.get("platform", ""),
    }

    record = client.store_fingerprint(fp.fingerprint, metadata)

    if record.status not in ("confirmed", "simulated"):
        raise RuntimeError(
            f"Blockchain transaction failed: status={record.status}, "
            f"tx={record.transaction_hash}"
        )

    log.info(
        "Upload complete — tx=%s, status=%s",
        record.transaction_hash[:20], record.status,
    )
    return record
