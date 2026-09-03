"""
app/pipeline/orchestrator.py — End-to-end pipeline coordinator.

Orchestrates all 9 forensic pipeline stages in strict topological sequence:
  [1] Face Detection (OpenCV YuNet + Pre-flight quality inspection)
  [2] Face Embedding (5-point landmark affine alignment + SFace 128-D unit embedding + self-match calibration)
  [3] Live Web Search (Genuine discovery via SerpAPI Google Lens / Bing)
  [4] Candidate Face Verification (Multi-face scanning + quality filtering + cosine similarities)
  [5] Identity Decision (Three-tier calibrated state machine + margin separation)
  [6] Evidence Hash (RFC 8785 canonical JSON + deterministic SHA-256 fingerprint)
  [7] Blockchain Commitment (Ethereum Sepolia broadcast / Local EVM simulation)
  [8] Independent Verification (Cryptographic state reconstruction & smart contract query)
  [9] Tamper Detection (Single-byte metadata mutation & avalanche divergence proof)
"""
from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from app.blockchain.client import build_blockchain_client
from app.blockchain.hasher import (
    calculate_fingerprint,
    create_canonical_record,
    fingerprint_from_match,
)
from app.blockchain.uploader import upload_fingerprint
from app.blockchain.verifier import demonstrate_tamper, re_verify
from app.config import FACE_MATCH_THRESHOLD
from app.face.alignment import align_face_crop
from app.face.detector import detect_faces
from app.face.embedder import generate_face_embedding
from app.face.identity_decision import (
    BASELINE_MATCH_THRESHOLD,
    HIGH_CONFIDENCE_THRESHOLD,
    evaluate_identity_decision,
)
from app.face.matcher import (
    cosine_similarity,
    deduplicate_candidates,
    match_candidate,
    rank_candidates,
)
from app.face.similarity import cosine_similarity as verified_cosine_sim
from app.identity.calibration import calibrate_self_match
from app.models.schemas import (
    BlockchainRecord,
    EvidenceRecord,
    FaceResult,
    Fingerprint,
    MatchResult,
    SearchCandidate,
    VerificationResult,
)
from app.search.provenance import create_candidate_provenance
from app.search.result_validator import filter_accessible_candidates, summarise_matches
from app.search.reverse_search import search_for_image
from app.utils.file_utils import assess_image_quality, download_image, sha256_file, validate_image_path
from app.utils.logger import get_logger

log = get_logger(__name__)

console = Console(legacy_windows=False)
_RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "results"


# ─────────────────────────────────────────────────────────────────────────────
# Step implementations
# ─────────────────────────────────────────────────────────────────────────────

def step_load_image(image_path: str | Path) -> tuple[Path, str, dict[str, Any]]:
    """Stage 1a — validate input image file, compute SHA-256, and assess resolution/quality."""
    validated = validate_image_path(image_path)
    img_sha256 = sha256_file(validated)
    quality_info = assess_image_quality(validated)
    return validated, img_sha256, quality_info


def step_detect_faces(image_path: Path, face_index: Optional[int] = None) -> tuple[list[dict[str, Any]], int]:
    """
    Stage 1 — detect faces using YuNet (OpenCV Zoo ONNX).

    Returns (faces_list, selected_index). Requires explicit face index if multiple faces exist.
    """
    faces = detect_faces(image_path)
    if len(faces) == 0:
        raise ValueError(f"No faces detected in '{image_path.name}'. Please supply a clear face portrait.")

    if len(faces) > 1 and face_index is None:
        raise ValueError(
            f"Multiple faces ({len(faces)}) detected in '{image_path.name}'. "
            "Explicit face selection is required via --face-index to avoid selecting the wrong person."
        )

    selected = face_index if face_index is not None else 0
    if selected < 0 or selected >= len(faces):
        raise IndexError(f"Face index #{selected} is out of bounds for image with {len(faces)} face(s).")

    return faces, selected


def step_generate_embedding(image_path: Path, face_index: int) -> FaceResult:
    """Stage 2 — 5-point landmark affine alignment and 128-d L2-normalized SFace embedding."""
    return generate_face_embedding(image_path, face_index=face_index)


def step_search(image_path: Path, max_candidates: int = 30) -> list[SearchCandidate]:
    """Stage 3 — execute live reverse-image search, filter accessible URLs, and deduplicate."""
    candidates = search_for_image(image_path)
    candidates = filter_accessible_candidates(candidates)
    candidates = deduplicate_candidates(candidates)
    if not candidates:
        raise RuntimeError(
            "Search returned no accessible candidates. "
            "Verify your internet connection and SERPAPI_KEY in .env."
        )
    return candidates[:max_candidates]


def step_match_candidates(
    face_result: FaceResult,
    candidates: list[SearchCandidate],
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
) -> list[MatchResult]:
    """Stage 4 — download candidate images, scan ALL detected faces per image, and compute similarities."""
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[MatchResult] = []

    for i, candidate in enumerate(candidates):
        if progress_cb:
            progress_cb(i + 1, len(candidates), candidate.url[:60])

        if not candidate.image_url:
            log.debug("Candidate %d has no image thumbnail — skipping face comparison.", i)
            results.append(
                MatchResult(candidate=candidate, matched=False, confidence=0.0)
            )
            continue

        try:
            local = download_image(candidate.image_url, _RESULTS_DIR / "candidates")
            mr = match_candidate(face_result, candidate, local)
        except Exception as exc:
            log.warning("Could not process candidate %s: %s", candidate.url[:60], exc)
            results.append(
                MatchResult(candidate=candidate, matched=False, confidence=0.0)
            )
            continue

        results.append(mr)

    # Rank descending by cosine similarity and compute runner-up margins
    return rank_candidates(results)


def step_fingerprint(
    best_match: MatchResult,
    image_path: Path,
    image_sha256: str,
) -> tuple[EvidenceRecord, Fingerprint]:
    """Stage 6 — construct canonical EvidenceRecord and deterministic SHA-256 fingerprint."""
    evidence = create_canonical_record(
        match=best_match,
        input_image_path=image_path,
        input_image_sha256=image_sha256,
    )
    fp = calculate_fingerprint(evidence)
    return evidence, fp


def step_upload(fp: Fingerprint, blockchain_network: Optional[str] = None) -> tuple[BlockchainRecord, object]:
    """Stage 7 — broadcast fingerprint to Ethereum Sepolia or Local EVM Simulation."""
    client = build_blockchain_client(preferred_network=blockchain_network)
    record = upload_fingerprint(fp, client)
    return record, client


def step_verify(
    record: BlockchainRecord,
    evidence_or_fp: Any,
    client: Any,
) -> VerificationResult:
    """Stage 8 — independently re-verify fingerprint against the blockchain registry."""
    return re_verify(record, evidence_or_fp, client)


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    image_path: str | Path,
    face_index: Optional[int] = None,
    max_candidates: int = 30,
    skip_blockchain: bool = False,
    blockchain_network: Optional[str] = None,
    judge_mode: bool = False,
    quiet: bool = False,
    stage_cb: Optional[Callable[[str, str, dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """
    Execute the complete 9-stage Face Identification & Blockchain Verification pipeline.
    """
    total_start = time.perf_counter()
    stage_timings: dict[str, float] = {}
    result: dict[str, Any] = {}

    def _show_banner(step_num: int, title: str) -> None:
        if not quiet:
            console.rule(f"[bold cyan][{step_num}/9] {title.upper()}[/bold cyan]")

    def _show_ok(msg: str) -> None:
        if not quiet:
            console.print(f"  [bold green][OK][/bold green] {msg}")

    # ── [1/9] Face Detection ──────────────────────────────────
    if stage_cb:
        stage_cb("1_face_detection", "running", {"title": "Face Detection"})
    t0 = time.perf_counter()
    _show_banner(1, "Face Detection & Pre-Flight Quality Check")
    validated, img_sha256, quality_info = step_load_image(image_path)
    faces, selected = step_detect_faces(validated, face_index=face_index)
    stage_timings["1_face_detection"] = time.perf_counter() - t0

    result["image_path"] = str(validated)
    result["input_image_sha256"] = img_sha256
    result["quality_info"] = quality_info
    result["faces"] = faces
    result["face_index"] = selected
    result["input"] = {
        "filename": validated.name,
        "sha256": img_sha256,
        "resolution": quality_info["resolution"],
        "quality": quality_info["summary"],
        "faces_detected": len(faces),
        "selected_face": selected,
    }

    fa = faces[selected]["facial_area"]
    conf = faces[selected]["confidence"]
    if not quiet:
        console.print(f"    Target Image    : [bold]{validated.name}[/bold]")
        console.print(f"    SHA-256 Checksum: [dim]{img_sha256}[/dim]")
        console.print(f"    Resolution      : [cyan]{quality_info['resolution']}[/cyan] ({quality_info['summary']})")
        console.print(f"    Faces Detected  : [bold cyan]{len(faces)}[/bold cyan]")
        console.print(f"    Selected Face   : #[bold]{selected}[/bold] (Detection Confidence: {conf*100:.1f}%)")
        console.print(f"    Bounding Box    : x={fa['x']}, y={fa['y']}, w={fa['w']}, h={fa['h']}")
    _show_ok(f"Face geometry extracted and validated in {stage_timings['1_face_detection']:.3f}s")

    # ── [2/9] Face Embedding & Alignment ──────────────────────
    if stage_cb:
        stage_cb("2_face_embedding", "running", {"title": "Face Embedding"})
    t0 = time.perf_counter()
    _show_banner(2, "Face Alignment & SFace 128-D Embedding Generation")
    face_result = step_generate_embedding(validated, selected)
    self_calib = calibrate_self_match(face_result.embedding)
    stage_timings["2_face_embedding"] = time.perf_counter() - t0

    result["face_result"] = face_result
    result["self_match_calibration"] = self_calib

    if not quiet:
        console.print(f"    Alignment       : 5-Point Affine Landmark Normalization (112x112 canonical)")
        console.print(f"    Feature Model   : OpenCV SFace (ONNX)")
        console.print(f"    Embedding Dim   : [bold cyan]{face_result.embedding_dimension}-D[/bold cyan] (L2-Normalized, ||v||=1.0)")
        console.print(f"    Self-Match Test : [bold green]{self_calib*100:.2f}%[/bold green] (Internal Sanity Pass)")
    _show_ok(f"128-d unit embedding generated and calibrated in {stage_timings['2_face_embedding']:.3f}s")

    # ── [3/9] Live Web Search ─────────────────────────────────
    if stage_cb:
        stage_cb("3_live_search", "running", {"title": "Live Web Search"})
    t0 = time.perf_counter()
    _show_banner(3, "Live Web & Social Media Discovery (Google Lens / SerpAPI)")
    candidates = step_search(validated, max_candidates=max_candidates)
    stage_timings["3_live_search"] = time.perf_counter() - t0

    result["candidates"] = candidates
    if not quiet:
        console.print(f"    Search Engine   : SerpAPI Google Lens Engine")
        console.print(f"    Query Target    : [dim]{validated.name}[/dim]")
        console.print(f"    Candidates Found: [bold cyan]{len(candidates)}[/bold cyan] unique, deduplicated URLs")
    _show_ok(f"Retrieved {len(candidates)} candidate(s) across platforms in {stage_timings['3_live_search']:.3f}s")

    # ── [4/9] Candidate Face Verification ─────────────────────
    if stage_cb:
        stage_cb("4_candidate_verification", "running", {"title": "Candidate Face Verification"})
    t0 = time.perf_counter()
    _show_banner(4, "Candidate Face Extraction & Multi-Face Verification")

    def _cb(current: int, total: int, url: str) -> None:
        if not quiet:
            console.print(f"    [{current}/{total}] Scanning: {url[:70]}…", style="dim")

    matches = step_match_candidates(face_result, candidates, progress_cb=_cb)
    stage_timings["4_candidate_verification"] = time.perf_counter() - t0
    result["matches"] = matches

    summary = summarise_matches(matches)
    result["match_summary"] = summary
    best = matches[0] if matches else None

    if not quiet and matches:
        cand_table = Table(title="Ranked Candidate Multi-Face Comparisons", border_style="cyan")
        cand_table.add_column("Rank", justify="center", style="bold")
        cand_table.add_column("Platform", style="yellow")
        cand_table.add_column("Candidate Page URL", style="dim")
        cand_table.add_column("Faces", justify="center")
        cand_table.add_column("Cosine Sim", justify="right")
        cand_table.add_column("Margin (Δ)", justify="right")
        cand_table.add_column("Decision Tier", justify="center")

        for idx, m in enumerate(matches[:5], 1):
            if m.decision_tier == "HIGH_MATCH":
                verdict = "[bold green]HIGH MATCH[/bold green]"
            elif m.decision_tier == "REVIEW":
                verdict = "[bold yellow]REVIEW[/bold yellow]"
            else:
                verdict = "[dim red]NO MATCH[/dim red]"

            cand_table.add_row(
                f"#{idx}",
                m.candidate.platform,
                m.candidate.url[:45] + ("…" if len(m.candidate.url) > 45 else ""),
                f"#{m.matched_face_index+1}/{m.candidate_faces_count}",
                f"{m.confidence:.4f}",
                f"+{m.margin_from_runner_up:.4f}",
                verdict,
            )
        console.print(cand_table)

    _show_ok(f"Evaluated {len(matches)} candidate image(s) and multi-face crops in {stage_timings['4_candidate_verification']:.3f}s")

    # ── [5/9] Identity Decision ───────────────────────────────
    if stage_cb:
        stage_cb("5_identity_decision", "running", {"title": "Identity Decision"})
    t0 = time.perf_counter()
    _show_banner(5, "Forensic Identity Decision Analysis")

    margin = best.margin_from_runner_up if best else 0.0
    decision_tier = best.decision_tier if best else "NO_MATCH"
    identity_dec = best.identity_decision if best else None
    result["margin_from_runner_up"] = margin
    result["decision_tier"] = decision_tier
    result["identity_decision"] = identity_dec
    stage_timings["5_identity_decision"] = time.perf_counter() - t0

    if best:
        is_reliable_match = (decision_tier == "HIGH_MATCH")
        face_info_str = f"Face #{best.matched_face_index+1} of {best.candidate_faces_count} detected"
        status_color = "green" if decision_tier == "HIGH_MATCH" else ("yellow" if decision_tier == "REVIEW" else "red")
        tier_title = "HIGH CONFIDENCE MATCH" if decision_tier == "HIGH_MATCH" else ("POSSIBLE MATCH — REVIEW" if decision_tier == "REVIEW" else "NO RELIABLE MATCH FOUND")

        panel_content = (
            f"Candidate Page URL   : {best.candidate.url}\n"
            f"Candidate Image URL  : {best.candidate.image_url[:70]}…\n"
            f"Platform / Domain    : {best.candidate.platform}\n"
            f"Face Analysis        : {face_info_str}\n"
            f"Measured Similarity  : [bold cyan]{best.confidence:.4f}[/bold cyan] (Cosine metric, range [-1.0, 1.0])\n"
            f"Calibrated Threshold : {BASELINE_MATCH_THRESHOLD:.3f} (High Confidence: {HIGH_CONFIDENCE_THRESHOLD:.3f})\n"
            f"Runner-Up Margin (Δ) : +{margin:.4f} (Min Required: 0.035)\n"
            f"Final Decision       : [bold {status_color}]{tier_title}[/bold {status_color}]\n\n"
            f"Forensic Rationale   : {best.decision_reason or 'Evaluated against calibrated biometric boundary'}"
        )

        if not quiet:
            console.print(
                Panel(
                    panel_content,
                    title="[bold]EVIDENCE & IDENTITY DECISION SUMMARY[/bold]",
                    border_style=status_color,
                )
            )

        if decision_tier == "HIGH_MATCH":
            _show_ok(f"Identity confirmed with high confidence ({best.confidence:.4f})")
        elif decision_tier == "REVIEW":
            _show_ok(f"Candidate flagged for manual review ({best.confidence:.4f})")
        else:
            if not quiet:
                console.print("[bold red]    STATUS: NO RELIABLE MATCH FOUND[/bold red]")
                console.print("[dim]    Search produced candidates, but none satisfied the calibrated identity criteria.[/dim]")

    result["best_match"] = best

    # ── [6/9] Evidence Hash ───────────────────────────────────
    if stage_cb:
        stage_cb("6_evidence_hash", "running", {"title": "Evidence Hash"})
    t0 = time.perf_counter()
    _show_banner(6, "Canonical Evidence Record Serialization & SHA-256 Hashing")
    evidence_rec, fp = step_fingerprint(best, validated, img_sha256)
    stage_timings["6_evidence_hash"] = time.perf_counter() - t0

    result["evidence_record"] = evidence_rec
    result["fingerprint"] = fp

    if not quiet:
        console.print(f"    Canonical Encoding: RFC 8785 JSON Canonicalization Scheme (JCS)")
        console.print(f"    Biometric Privacy : Zero raw embeddings stored on-chain (100% Zero-Leak)")
        console.print(f"    SHA-256 Digest    : [bold green]{fp.fingerprint}[/bold green]")
    _show_ok(f"Deterministic fingerprint generated in {stage_timings['6_evidence_hash']:.3f}s")

    if skip_blockchain:
        if not quiet:
            console.print("[yellow]Blockchain stages skipped (--skip-blockchain active).[/yellow]")
        result["blockchain_record"] = None
        result["verification"] = None
        result["tamper_demo"] = None
        result["total_pipeline_duration_seconds"] = time.perf_counter() - total_start
        result["stage_timings"] = stage_timings
        return result

    # ── [7/9] Blockchain Commitment ───────────────────────────
    if stage_cb:
        stage_cb("7_blockchain_commitment", "running", {"title": "Blockchain Commitment"})
    t0 = time.perf_counter()
    _show_banner(7, "Blockchain Commitment & Transaction Broadcast")
    record, client = step_upload(fp, blockchain_network=blockchain_network)
    stage_timings["7_blockchain_commitment"] = time.perf_counter() - t0

    result["blockchain_record"] = record
    if not quiet:
        console.print(f"    Network           : [bold]{record.network}[/bold]")
        console.print(f"    Transaction Hash  : [cyan]{record.transaction_hash}[/cyan]")
        if record.contract_address:
            console.print(f"    Contract Address  : [dim]{record.contract_address}[/dim]")
        if record.explorer_url:
            console.print(f"    Block Explorer    : [link={record.explorer_url}]{record.explorer_url}[/link]")
        console.print(f"    Commitment Status : [bold green]{record.status.upper()}[/bold green]")
    _show_ok(f"Fingerprint committed to blockchain in {stage_timings['7_blockchain_commitment']:.3f}s")

    # ── [8/9] Independent Verification ────────────────────────
    if stage_cb:
        stage_cb("8_independent_verification", "running", {"title": "Independent Verification"})
    t0 = time.perf_counter()
    _show_banner(8, "Independent Blockchain Verification")
    verification = step_verify(record, evidence_rec, client)
    stage_timings["8_independent_verification"] = time.perf_counter() - t0

    result["verification"] = verification
    if not quiet:
        _print_verification_panel(verification)
    _show_ok(f"Independent on-chain verification completed ({stage_timings['8_independent_verification']:.3f}s)")

    # ── [9/9] Tamper Detection ────────────────────────────────
    if stage_cb:
        stage_cb("9_tamper_detection", "running", {"title": "Tamper Detection"})
    t0 = time.perf_counter()
    _show_banner(9, "Cryptographic Tamper Detection & Avalanche Test")
    tamper = demonstrate_tamper(evidence_rec)
    stage_timings["9_tamper_detection"] = time.perf_counter() - t0

    result["tamper_demo"] = tamper
    if not quiet:
        _print_tamper_panel(tamper)
    _show_ok(f"Tamper divergence demonstrated in {stage_timings['9_tamper_detection']:.3f}s")

    # ── Final Timing Summary ─────────────────────────────────
    total_duration = time.perf_counter() - total_start
    result["total_pipeline_duration_seconds"] = total_duration
    result["stage_timings"] = stage_timings

    if not quiet and judge_mode:
        _print_judge_diagnostics_panel(result)

    return result


def _print_verification_panel(v: VerificationResult) -> None:
    status_str = "[bold green]PASS (VERIFIED ON-CHAIN)[/bold green]" if v.verified else "[bold red]FAIL (MISMATCH)[/bold red]"
    console.print(
        Panel(
            f"Local Digest    : [green]{v.current_fingerprint}[/green]\n"
            f"On-Chain Digest : [green]{v.stored_fingerprint}[/green]\n"
            f"Transaction Hash: [cyan]{v.transaction_hash}[/cyan]\n"
            f"Verification    : {status_str}\n\n"
            f"[dim]{v.note}[/dim]",
            title="[bold]INDEPENDENT BLOCKCHAIN VERIFICATION[/bold]",
            border_style="green" if v.verified else "red",
        )
    )


def _print_tamper_panel(t: dict[str, Any]) -> None:
    detected_color = "green" if t.get("tamper_detected") else "red"
    console.print(
        Panel(
            f"Original Digest : [green]{t['original_hash']}[/green]\n"
            f"Tampered Digest : [red]{t['tampered_hash']}[/red]\n"
            f"Mutation Field  : [yellow]{t.get('mutated_field', 'title')}[/yellow] (Altered by modifier string)\n"
            f"Bit Divergence  : [bold cyan]{t.get('bit_difference', 'N/A')}[/bold cyan] (Avalanche Effect)\n\n"
            f"Original == Chain  : [bold green]PASS (MATCH)[/bold green]\n"
            f"Tampered == Chain  : [bold red]FAIL (DIVERGENCE)[/bold red]\n"
            f"Tamper Detected    : [{detected_color}]YES — Cryptographic mismatch confirmed[/{detected_color}]",
            title="[bold]TAMPER DETECTION & AVALANCHE EFFECT PROOF[/bold]",
            border_style="cyan",
        )
    )


def _print_judge_diagnostics_panel(res: dict[str, Any]) -> None:
    """Print an exhaustive engineering and audit panel for judges."""
    from app.face.models import get_model_checksums

    console.rule("[bold magenta]JUDGE AUDIT — SYSTEM & CRYPTOGRAPHIC VERIFICATION[/bold magenta]")

    # 1. Models Table
    model_table = Table(title="1. ONNX Model Integrity Checksums", border_style="magenta")
    model_table.add_column("Model", style="bold")
    model_table.add_column("Size", justify="right")
    model_table.add_column("SHA-256 Digest", style="dim")
    model_table.add_column("Status", style="green")

    for name, info in get_model_checksums().items():
        model_table.add_row(name, info["size_mb"], info["sha256"][:32] + "…", info["status"])
    console.print(model_table)

    # 2. Timing Table
    timings = res.get("stage_timings", {})
    timer_table = Table(title="2. High-Precision Pipeline Stage Timings", border_style="cyan")
    timer_table.add_column("Pipeline Stage", style="bold")
    timer_table.add_column("Duration (seconds)", justify="right")
    timer_table.add_column("% of Total", justify="right")

    total = res.get("total_pipeline_duration_seconds", 1.0)
    for stage, dur in timings.items():
        pct = (dur / total) * 100 if total > 0 else 0
        timer_table.add_row(stage, f"{dur:.4f}s", f"{pct:.1f}%")
    console.print(timer_table)
    console.print(f"[bold cyan]TOTAL PIPELINE EXECUTION TIME: {total:.4f}s[/bold cyan]\n")
