"""
app/main.py — CLI entry point for the Face Identification & Blockchain Verification pipeline.

Usage (interactive):
    python -m app.main

Usage (non-interactive):
    python -m app.main --image ./data/input/person.jpg [--face-index 0]
                       [--max-candidates 10] [--skip-blockchain]
                       [--judge-mode] [--json]

Output is written to stdout. Logs go to stderr.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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
from rich.prompt import IntPrompt, Prompt

from app.demo import run_demo
from app.pipeline.orchestrator import (
    run_pipeline,
    step_detect_faces,
    step_load_image,
)

console = Console(legacy_windows=False)

_LOGO = """
+----------------------------------------------------------+
|   FACE IDENTIFICATION & BLOCKCHAIN VERIFICATION          |
|   HH Goa 2026 -- Task 3                                  |
+----------------------------------------------------------+
"""


def _print_logo() -> None:
    console.print(_LOGO, style="bold cyan")


def _print_menu() -> None:
    console.print(
        "\n[bold]Pipeline steps:[/bold]\n"
        "  [1] Load & validate image\n"
        "  [2] Detect face(s) (OpenCV YuNet)\n"
        "  [3] Generate face embedding (SFace 128-d)\n"
        "  [4] Search web / social media (SerpAPI / Bing)\n"
        "  [5] Compare candidate faces (Cosine Similarity)\n"
        "  [6] Generate canonical SHA-256 fingerprint\n"
        "  [7] Upload fingerprint to blockchain (Sepolia / Sim)\n"
        "  [8] Verify on-chain record & tamper demonstration\n"
        "  [R] Run full pipeline end-to-end\n"
        "  [J] Run full pipeline in JUDGE AUDIT mode\n"
        "  [Q] Quit\n"
    )


def interactive_mode() -> None:
    """Interactive menu-driven pipeline."""
    _print_logo()

    while True:
        _print_menu()
        choice = Prompt.ask("[bold]Choice[/bold]", default="R").strip().upper()

        if choice == "Q":
            console.print("Goodbye.")
            sys.exit(0)

        if choice in ("R", "J"):
            judge_mode = (choice == "J")
            image_path = Prompt.ask("[bold]Enter path to face image[/bold]")
            try:
                validated, _ = step_load_image(image_path)
            except (FileNotFoundError, ValueError) as exc:
                console.print(f"[red]Error: {exc}[/red]")
                continue

            # Face detection preview to allow user to select a face
            try:
                faces, _ = step_detect_faces(validated)
            except ValueError as exc:
                console.print(f"[red]Error: {exc}[/red]")
                continue

            face_index = 0
            if len(faces) > 1:
                console.print(f"[yellow]{len(faces)} faces detected.[/yellow]")
                for f in faces:
                    fa = f["facial_area"]
                    console.print(
                        f"  Face {f['index']}: x={fa['x']} y={fa['y']} "
                        f"w={fa['w']} h={fa['h']} "
                        f"confidence={f['confidence']:.2f}"
                    )
                face_index = IntPrompt.ask(
                    "Select face index",
                    default=0,
                    choices=[str(i) for i in range(len(faces))],
                )

            skip_bc = (
                Prompt.ask(
                    "Skip blockchain step?",
                    choices=["y", "n"],
                    default="n",
                ) == "y"
            )

            try:
                run_demo(
                    image_path=str(validated),
                    face_index=face_index,
                    skip_blockchain=skip_bc,
                    judge_mode=judge_mode,
                )
            except Exception as exc:
                console.print(f"\n[bold red]Pipeline error:[/bold red] {exc}")
                console.print("[dim]Check the log above for details.[/dim]")
        else:
            console.print(
                "[yellow]Please choose R to run the pipeline, J for Judge Mode, or Q to quit.[/yellow]"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="face-blockchain-verifier",
        description="Face Identification & Blockchain Verification Pipeline",
    )
    parser.add_argument(
        "--image", "-i",
        help="Path to input face image (JPG/PNG). Omit for interactive mode.",
    )
    parser.add_argument(
        "--face-index", "-f",
        type=int, default=None,
        help="Which face to use when multiple are detected (default: 0).",
    )
    parser.add_argument(
        "--max-candidates", "-m",
        type=int, default=10,
        help="Maximum number of search candidates to process (default: 10).",
    )
    parser.add_argument(
        "--skip-blockchain",
        action="store_true",
        help="Skip blockchain upload and verification (face+search only).",
    )
    parser.add_argument(
        "--judge-mode",
        action="store_true",
        help="Display detailed judge diagnostics, model checksums, and execution timers.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output pure, machine-readable JSON to stdout.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging.",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.image:
        sys.exit(
            run_demo(
                image_path=args.image,
                face_index=args.face_index,
                max_candidates=args.max_candidates,
                skip_blockchain=args.skip_blockchain,
                json_output=args.json,
                judge_mode=args.judge_mode,
            )
        )
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
