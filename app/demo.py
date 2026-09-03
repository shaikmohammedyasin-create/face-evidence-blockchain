"""
app/demo.py — Dedicated Unified Demo & Judge Evaluation Entry Point.

Usage:
    # Standard Rich Terminal Run (Uses default target image ./data/input/target.jpg)
    python -m app.demo

    # Explicit Target Image Run
    python -m app.demo --image ./data/input/target.jpg

    # Judge Mode with Explicit Blockchain Target
    python -m app.demo --image ./data/input/target.jpg --judge-mode --blockchain sepolia

    # Machine-Readable JSON Output (For automated testing/grading harnesses)
    python -m app.demo --image ./data/input/target.jpg --json
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
import logging
from pathlib import Path
import sys
from typing import Any

# Ensure UTF-8 console output on Windows platforms
if sys.platform.startswith("win"):
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel

from app.pipeline.orchestrator import run_pipeline
from app.utils.file_utils import find_default_target_image
from app.utils.logger import get_logger

console = Console(legacy_windows=False)
log = get_logger(__name__)

_HEADER = """
+------------------------------------------------------------------+
|   FACE IDENTIFICATION & BLOCKCHAIN VERIFICATION                  |
|   Competition Submission -- HH Goa 2026 (Task 3)                  |
+------------------------------------------------------------------+
"""


def _serialize_for_json(obj: Any) -> Any:
    """Helper to cleanly serialize dataclasses and custom objects to JSON."""
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    if isinstance(obj, bytes):
        return obj.hex()
    return str(obj)


def run_demo(
    image_path: str | Path | None = None,
    face_index: int | None = None,
    max_candidates: int = 10,
    skip_blockchain: bool = False,
    blockchain_network: str | None = None,
    json_output: bool = False,
    judge_mode: bool = False,
) -> int:
    """Execute the pipeline demo and render output in the requested format."""
    resolved_path: Path | None = None

    if image_path:
        resolved_path = Path(image_path)
    else:
        resolved_path = find_default_target_image()
        if not resolved_path:
            console.print("[bold red]Error: No target image found in ./data/input/. Please provide --image path.[/bold red]")
            return 1

    if not json_output:
        console.print(_HEADER, style="bold cyan")
        console.print(f"Target Image: [bold]{resolved_path}[/bold]")
        if judge_mode:
            console.print("[bold magenta]Mode: JUDGE AUDIT & SYSTEM DIAGNOSTICS ACTIVE[/bold magenta]\n")
        else:
            console.print("[dim]Mode: Standard Forensic Pipeline Evaluation[/dim]\n")

    try:
        result = run_pipeline(
            image_path=resolved_path,
            face_index=face_index,
            max_candidates=max_candidates,
            skip_blockchain=skip_blockchain,
            blockchain_network=blockchain_network,
            judge_mode=judge_mode,
            quiet=json_output,
        )
    except Exception as exc:
        if json_output:
            err_json = {
                "status": "error",
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
            }
            print(json.dumps(err_json, indent=2))
        else:
            console.print(f"\n[bold red]Fatal Pipeline Error:[/bold red] {exc}")
            log.debug("Pipeline execution halted: %s", exc, exc_info=True)
        return 1

    if json_output:
        # Build clean serializable dictionary
        safe_result: dict[str, Any] = {
            "status": "success",
            "input": result.get("input", {
                "filename": Path(result.get("image_path", "")).name,
                "sha256": result.get("input_image_sha256"),
                "resolution": result.get("quality_info", {}).get("resolution") if result.get("quality_info") else None,
                "faces_detected": len(result.get("faces", [])),
            }),
            "image_path": result.get("image_path"),
            "input_image_sha256": result.get("input_image_sha256"),
            "faces_detected": len(result.get("faces", [])),
            "selected_face_index": result.get("face_index"),
            "embedding_dimension": result.get("face_result").embedding_dimension if result.get("face_result") else None,
            "self_match_calibration": result.get("self_match_calibration"),
            "candidates_count": len(result.get("candidates", [])),
            "best_match": {
                "matched": result["best_match"].matched if result.get("best_match") else False,
                "confidence": result["best_match"].confidence if result.get("best_match") else 0.0,
                "decision_tier": result.get("decision_tier", "NO_MATCH"),
                "platform": result["best_match"].candidate.platform if result.get("best_match") else None,
                "url": result["best_match"].candidate.url if result.get("best_match") else None,
                "title": result["best_match"].candidate.title if result.get("best_match") else None,
                "margin_from_runner_up": result.get("margin_from_runner_up", 0.0),
                "matched_face_index": result["best_match"].matched_face_index if result.get("best_match") else 0,
                "candidate_faces_count": result["best_match"].candidate_faces_count if result.get("best_match") else 1,
            } if result.get("best_match") else None,
            "fingerprint": {
                "algorithm": result["fingerprint"].algorithm if result.get("fingerprint") else None,
                "digest": result["fingerprint"].fingerprint if result.get("fingerprint") else None,
                "canonical_json": result["fingerprint"].canonical_json if result.get("fingerprint") else None,
            } if result.get("fingerprint") else None,
            "blockchain_record": {
                "network": result["blockchain_record"].network if result.get("blockchain_record") else None,
                "transaction_hash": result["blockchain_record"].transaction_hash if result.get("blockchain_record") else None,
                "status": result["blockchain_record"].status if result.get("blockchain_record") else None,
                "block_number": result["blockchain_record"].block_number if result.get("blockchain_record") else None,
                "gas_used": result["blockchain_record"].gas_used if result.get("blockchain_record") else None,
                "explorer_url": result["blockchain_record"].explorer_url if result.get("blockchain_record") else None,
                "contract_address": result["blockchain_record"].contract_address if result.get("blockchain_record") else None,
            } if result.get("blockchain_record") else None,
            "verification": {
                "verified": result["verification"].verified if result.get("verification") else None,
                "stored_fingerprint": result["verification"].stored_fingerprint if result.get("verification") else None,
                "current_fingerprint": result["verification"].current_fingerprint if result.get("verification") else None,
                "note": result["verification"].note if result.get("verification") else None,
            } if result.get("verification") else None,
            "tamper_demo": {
                "tamper_detected": result["tamper_demo"].get("tamper_detected") if result.get("tamper_demo") else None,
                "original_hash": result["tamper_demo"].get("original_hash") if result.get("tamper_demo") else None,
                "tampered_hash": result["tamper_demo"].get("tampered_hash") if result.get("tamper_demo") else None,
                "bit_difference": result["tamper_demo"].get("bit_difference") if result.get("tamper_demo") else None,
            } if result.get("tamper_demo") else None,
            "stage_timings": result.get("stage_timings", {}),
            "total_duration_seconds": result.get("total_pipeline_duration_seconds", 0.0),
        }
        print(json.dumps(safe_result, indent=2, default=_serialize_for_json))
    else:
        _print_final_summary(result)

    return 0


def _print_final_summary(result: dict[str, Any]) -> None:
    """Print the final executive summary panel."""
    face_result = result.get("face_result")
    matches = result.get("matches", [])
    best = result.get("best_match")
    decision_tier = result.get("decision_tier", "NO_MATCH")
    fp = result.get("fingerprint")
    record = result.get("blockchain_record")
    verification = result.get("verification")
    tamper = result.get("tamper_demo")
    total_time = result.get("total_pipeline_duration_seconds", 0.0)

    def _status(ok: bool | None) -> str:
        if ok is None:
            return "[dim]SKIPPED[/dim]"
        return "[bold green]PASS [OK][/bold green]" if ok else "[bold red]FAIL [ERR][/bold red]"

    face_pass = face_result is not None and face_result.embedding_generated
    search_pass = len(matches) > 0
    candidate_pass = len(matches) > 0
    decision_pass = decision_tier in ("HIGH_MATCH", "REVIEW", "NO_MATCH")
    hash_pass = fp is not None
    upload_pass = record is not None and record.status in ("confirmed", "simulated")
    verify_pass = verification.verified if verification else None
    tamper_pass = tamper is not None and tamper.get("tamper_detected", False)

    summary_text = (
        f"[1] Face Detection              : {_status(face_pass)}\n"
        f"[2] Face Embedding              : {_status(face_pass)}\n"
        f"[3] Live Web Search             : {_status(search_pass)}\n"
        f"[4] Candidate Face Verification : {_status(candidate_pass)}\n"
        f"[5] Identity Decision           : {_status(decision_pass)} ({decision_tier})\n"
        f"[6] Evidence Hash               : {_status(hash_pass)}\n"
        f"[7] Blockchain Commitment       : {_status(upload_pass)}\n"
        f"[8] Independent Verification    : {_status(verify_pass)}\n"
        f"[9] Tamper Detection            : {_status(tamper_pass)}\n\n"
        f"[bold cyan]TOTAL PIPELINE EXECUTION TIME: {total_time:.3f}s[/bold cyan]"
    )

    console.print(
        Panel(
            summary_text,
            title="[bold]FACE IDENTIFICATION & BLOCKCHAIN VERIFICATION SUMMARY[/bold]",
            border_style="cyan",
        )
    )

    if record:
        console.print(f"[bold]Blockchain Network:[/bold] {record.network}")
        console.print(f"Transaction Hash  : [cyan]{record.transaction_hash}[/cyan]")
        if record.contract_address:
            console.print(f"Contract Address  : [dim]{record.contract_address}[/dim]")
        if record.explorer_url:
            console.print(f"Etherscan Explorer: [link={record.explorer_url}]{record.explorer_url}[/link]")
    if fp:
        console.print(f"SHA-256 Digest    : [green]{fp.fingerprint}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="face-blockchain-demo",
        description="Face Identification & Blockchain Verification — Competition Submission",
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the input face portrait (JPG/PNG/WEBP).",
    )
    parser.add_argument(
        "--face-index", "-f",
        type=int, default=None,
        help="Face index to select when multiple faces are detected (default: 0).",
    )
    parser.add_argument(
        "--max-candidates", "-m",
        type=int, default=10,
        help="Maximum search candidates to process (default: 10).",
    )
    parser.add_argument(
        "--skip-blockchain",
        action="store_true",
        help="Skip blockchain upload and verification (face detection + visual search only).",
    )
    parser.add_argument(
        "--blockchain", "-b",
        type=str, default=None,
        help="Blockchain target network ('sepolia' for Ethereum Sepolia testnet or 'simulation' for local simulation).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output pure, machine-readable JSON to stdout.",
    )
    parser.add_argument(
        "--judge-mode",
        action="store_true",
        help="Enable full technical audit mode (model checksums, canonical bytes, gas metrics, timers).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging to stderr.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    sys.exit(
        run_demo(
            image_path=args.image,
            face_index=args.face_index,
            max_candidates=args.max_candidates,
            skip_blockchain=args.skip_blockchain,
            blockchain_network=args.blockchain,
            json_output=args.json,
            judge_mode=args.judge_mode,
        )
    )


if __name__ == "__main__":
    main()
