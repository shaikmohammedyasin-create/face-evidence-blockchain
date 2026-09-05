"""
app/evidence/certificate.py — HTML Verification Certificate Generator.

Generates a standalone, visually clean HTML verification certificate artifact
in ./output/ after successful on-chain independent verification (Stage 8).
"""
from __future__ import annotations

import base64
from datetime import datetime
import io
from pathlib import Path
from typing import Optional

import qrcode

from app.utils.logger import get_logger

log = get_logger(__name__)

_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"


def generate_qr_code_base64(url: str) -> str:
    """Generate a base64-encoded PNG Data URI for the given URL string using qrcode."""
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=6,
        border=2,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    raw_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{raw_b64}"


def generate_verification_certificate(
    input_image_sha256: str,
    matched_candidate_url: str,
    matched_platform: str,
    decision_tier: str,
    cosine_similarity: float,
    margin: float,
    transaction_hash: str,
    contract_address: str,
    network_name: str = "Ethereum Sepolia",
    output_dir: Optional[Path | str] = None,
) -> Path:
    """
    Generate a single-file HTML verification certificate containing cryptographic and
    biometric evidence, on-chain transaction metadata, and an Etherscan QR code.

    Saves to *output_dir* (default: ./output/verification_certificate_<timestamp>.html).
    """
    out_dir = Path(output_dir) if output_dir else _DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamp_iso = datetime.now().isoformat()
    filename = f"verification_certificate_{timestamp_str}.html"
    filepath = out_dir / filename

    tx_hash = transaction_hash or "0x0000000000000000000000000000000000000000"
    etherscan_url = f"https://sepolia.etherscan.io/tx/{tx_hash}"
    qr_data_uri = generate_qr_code_base64(etherscan_url)

    tier_color = "#10B981" if decision_tier == "HIGH_MATCH" else ("#F59E0B" if decision_tier == "REVIEW" else "#EF4444")
    tier_label = "HIGH CONFIDENCE MATCH" if decision_tier == "HIGH_MATCH" else ("POSSIBLE MATCH REVIEW" if decision_tier == "REVIEW" else "NO RELIABLE MATCH FOUND")

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Biometric & Blockchain Verification Certificate</title>
    <style>
        :root {{
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --border-color: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-cyan: #06b6d4;
            --accent-green: #10b981;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-primary);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }}
        .certificate-card {{
            background-color: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5);
            max-width: 750px;
            width: 100%;
            padding: 32px;
            box-sizing: border-box;
        }}
        .header {{
            border-bottom: 2px solid var(--border-color);
            padding-bottom: 20px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .header h1 {{
            font-size: 20px;
            margin: 0;
            color: var(--accent-cyan);
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}
        .badge {{
            background-color: rgba(16, 185, 129, 0.15);
            color: var(--accent-green);
            border: 1px solid var(--accent-green);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 200px;
            gap: 24px;
        }}
        .field-group {{
            margin-bottom: 16px;
        }}
        .field-label {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-secondary);
            margin-bottom: 4px;
        }}
        .field-value {{
            font-size: 14px;
            font-weight: 500;
            color: var(--text-primary);
            word-break: break-all;
        }}
        .field-value.mono {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 12px;
            color: var(--accent-cyan);
        }}
        .field-value.highlight {{
            color: {tier_color};
            font-weight: 700;
        }}
        .qr-container {{
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: #ffffff;
            padding: 12px;
            border-radius: 8px;
            text-align: center;
        }}
        .qr-container img {{
            width: 140px;
            height: 140px;
        }}
        .qr-caption {{
            font-size: 10px;
            color: #475569;
            margin-top: 6px;
            font-weight: 600;
        }}
        .footer {{
            margin-top: 24px;
            border-top: 1px solid var(--border-color);
            padding-top: 16px;
            display: flex;
            justify-content: space-between;
            font-size: 11px;
            color: var(--text-secondary);
        }}
        a {{
            color: var(--accent-cyan);
            text-decoration: none;
        }}
        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <div class="certificate-card">
        <div class="header">
            <div>
                <h1>Biometric & Blockchain Verification Certificate</h1>
                <div style="font-size: 12px; color: var(--text-secondary); margin-top: 4px;">HH Goa 2026 — Task 3 Verification Artifact</div>
            </div>
            <div class="badge">ON-CHAIN VERIFIED</div>
        </div>

        <div class="grid">
            <div class="details">
                <div class="field-group">
                    <div class="field-label">Input Image SHA-256 Checksum</div>
                    <div class="field-value mono">{input_image_sha256}</div>
                </div>

                <div class="field-group">
                    <div class="field-label">Matched Candidate Web URL</div>
                    <div class="field-value"><a href="{matched_candidate_url}" target="_blank">{matched_candidate_url}</a></div>
                </div>

                <div class="field-group">
                    <div class="field-label">Platform & Decision Tier</div>
                    <div class="field-value">
                        <span>{matched_platform}</span> &bull; <span class="highlight">{tier_label} ({decision_tier})</span>
                    </div>
                </div>

                <div class="field-group">
                    <div class="field-label">Cosine Similarity & Statistical Runner-Up Margin</div>
                    <div class="field-value">
                        Similarity: <strong>{cosine_similarity * 100:.1f}%</strong> ({cosine_similarity:.4f}) &bull;
                        Margin (Δ): <strong>+{margin * 100:.1f}%</strong> (+{margin:.4f})
                    </div>
                </div>

                <div class="field-group">
                    <div class="field-label">Blockchain Network</div>
                    <div class="field-value">{network_name}</div>
                </div>

                <div class="field-group">
                    <div class="field-label">On-Chain Transaction Hash</div>
                    <div class="field-value mono"><a href="{etherscan_url}" target="_blank">{tx_hash}</a></div>
                </div>

                <div class="field-group">
                    <div class="field-label">Contract Address</div>
                    <div class="field-value mono">{contract_address or "0x0000000000000000000000000000000000000000"}</div>
                </div>
            </div>

            <div class="qr-container">
                <img src="{qr_data_uri}" alt="Etherscan QR Code">
                <div class="qr-caption">Scan to Verify on Etherscan</div>
            </div>
        </div>

        <div class="footer">
            <div>Timestamp: {timestamp_iso}</div>
            <div>Biometric Privacy: Zero Raw Vectors Stored On-Chain</div>
        </div>
    </div>
</body>
</html>
"""

    filepath.write_text(html_content, encoding="utf-8")
    log.info("Verification certificate generated: %s", filepath)
    return filepath
