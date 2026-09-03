"""
app/blockchain/verifier.py — Independent blockchain re-verification & tamper demonstration.

Key operations:
  1. re_verify()         — Recomputes canonical hash from source data and queries
                           the smart contract / client to confirm on-chain integrity.
  2. demonstrate_tamper() — Systematically mutates fields to illustrate cryptographic
                           avalanche effect and how tampering is immediately detected.
"""
from __future__ import annotations

from typing import Any, Union

from app.blockchain.client import BlockchainClient
from app.blockchain.hasher import calculate_fingerprint, fingerprint_from_data
from app.models.schemas import BlockchainRecord, EvidenceRecord, Fingerprint, VerificationResult
from app.utils.logger import get_logger

log = get_logger(__name__)


def re_verify(
    record: BlockchainRecord,
    original_fingerprint: Union[Fingerprint, EvidenceRecord, dict[str, Any]],
    client: BlockchainClient,
) -> VerificationResult:
    """
    Independently recalculate the fingerprint from the original data
    and query the smart contract via *client* to prove on-chain existence.

    Proves two properties:
      (a) Data Integrity: The source data has not changed since initial registration.
      (b) On-Chain Proof: The exact hash exists in the smart contract registry.

    Returns:
        VerificationResult with status, stored/recomputed hashes, and diagnostic notes.
    """
    if isinstance(original_fingerprint, Fingerprint):
        if original_fingerprint.evidence_record:
            recomputed = calculate_fingerprint(original_fingerprint.evidence_record)
        else:
            recomputed = calculate_fingerprint(original_fingerprint.source_data)
    elif isinstance(original_fingerprint, EvidenceRecord):
        recomputed = calculate_fingerprint(original_fingerprint)
    else:
        recomputed = calculate_fingerprint(original_fingerprint)

    on_chain_exists = client.verify_fingerprint(recomputed.fingerprint)
    fingerprints_match = record.fingerprint == recomputed.fingerprint

    verified = on_chain_exists and fingerprints_match

    if verified:
        note = (
            "Cryptographic proof confirmed. The recomputed canonical SHA-256 hash "
            "matches the on-chain registry record exactly."
        )
    elif not fingerprints_match:
        note = (
            "INTEGRITY VIOLATION — The recomputed fingerprint diverges from the uploaded record. "
            "Source evidence metadata has been altered."
        )
    else:
        note = (
            "REGISTRY LOOKUP FAILED — The fingerprint was not found in the on-chain contract. "
            "The transaction may still be pending confirmation or an invalid network is selected."
        )

    log.info(
        "Verification: verified=%s, on_chain=%s, hash_match=%s",
        verified, on_chain_exists, fingerprints_match,
    )

    return VerificationResult(
        transaction_hash=record.transaction_hash,
        stored_fingerprint=record.fingerprint,
        current_fingerprint=recomputed.fingerprint,
        verified=verified,
        note=note,
        network=record.network,
        block_number=record.block_number,
        timestamp=record.timestamp,
    )


def demonstrate_tamper(
    original_fingerprint: Union[Fingerprint, EvidenceRecord, dict[str, Any]],
    mutation_field: str = "title",
    mutation_value: str = " [TAMPERED METADATA]",
) -> dict[str, Any]:
    """
    Demonstrate cryptographic tamper detection by mutating a field in the evidence record.

    Illustrates the cryptographic avalanche effect: altering even a single byte in
    the input metadata produces a completely different 256-bit hash, causing verification
    to fail definitively.

    Returns:
        A rich comparison dictionary detailing the original vs tampered data and hashes.
    """
    if isinstance(original_fingerprint, Fingerprint):
        orig_fp = original_fingerprint
        source_dict = dict(original_fingerprint.source_data)
    elif isinstance(original_fingerprint, EvidenceRecord):
        orig_fp = calculate_fingerprint(original_fingerprint)
        source_dict = original_fingerprint.to_canonical_dict()
    else:
        orig_fp = calculate_fingerprint(original_fingerprint)
        source_dict = dict(original_fingerprint)

    original_hash = orig_fp.fingerprint

    # Apply mutation
    tampered_data = json_deep_copy(source_dict)
    if "match_metadata" in tampered_data and isinstance(tampered_data["match_metadata"], dict):
        current_val = tampered_data["match_metadata"].get(mutation_field, "")
        tampered_data["match_metadata"][mutation_field] = str(current_val) + mutation_value
    else:
        current_val = tampered_data.get(mutation_field, "")
        tampered_data[mutation_field] = str(current_val) + mutation_value

    tampered_fp = fingerprint_from_data(tampered_data)
    tampered_hash = tampered_fp.fingerprint

    tamper_detected = original_hash != tampered_hash

    # Calculate bit differences (hamming distance) to demonstrate avalanche effect
    bit_diff = 0
    try:
        orig_int = int(original_hash, 16)
        tamp_int = int(tampered_hash, 16)
        bit_diff = bin(orig_int ^ tamp_int).count("1")
    except Exception:
        pass

    log.info(
        "Tamper demonstration: original=%s…, tampered=%s…, bit_diff=%d/256 bits, detected=%s",
        original_hash[:16], tampered_hash[:16], bit_diff, tamper_detected,
    )

    return {
        "original_data": source_dict,
        "tampered_data": tampered_data,
        "original_hash": original_hash,
        "tampered_hash": tampered_hash,
        "tamper_detected": tamper_detected,
        "bit_difference": f"{bit_diff}/256 bits ({bit_diff/256*100:.1f}%)",
        "mutated_field": mutation_field,
        "algorithm": "SHA-256 (Canonical JSON)",
    }


def json_deep_copy(data: dict[str, Any]) -> dict[str, Any]:
    """Helper to deep-copy dictionary."""
    import copy
    return copy.deepcopy(data)
