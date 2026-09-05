# Face Identification & Blockchain Verification

[![Tests](https://img.shields.io/badge/tests-101%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Format](https://img.shields.io/badge/code%20style-PEP%208-orange.svg)](app/)

A forensic facial identification and cryptographic blockchain verification CLI application built for **HH Goa 2026 Shortlisting Task 3**.

---

## 🔒 Responsible Use & Scope

This application is engineered solely for demonstration and research purposes operating on consented, self-supplied, or publicly available test image portraits. Unconsented facial identification or lookup of private individuals is strictly out of scope and strongly discouraged. The decision engine is explicitly calibrated with zero-hallucination policies, strictly preferring `NO RELIABLE MATCH FOUND` over a false positive.

---

## ⚡ Quickstart (Judge Evaluation in Under 1 Minute)

### 1. Clone & Install
```bash
git clone <repository_url>
cd task3-face-identification-blockchain

# Create virtual environment
python -m venv .venv

# Activate environment
# On Linux/macOS:
source .venv/bin/activate
# On Windows (Command Prompt / PowerShell / Git Bash):
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Open .env and insert your SERPAPI_KEY (from https://serpapi.com)
```

### 3. Run Forensic Pipeline
```bash
# Standard CLI run
python -m app.demo --image ./data/input/target.png

# Full Judge Audit & Technical Diagnostics Mode
python -m app.demo --image ./data/input/target.png --judge-mode

# Demo Negative-Case Fixture (Demonstrates NO RELIABLE MATCH FOUND)
python -m app.demo --demo-negative-case

# Model Self-Match Sanity Calibration Diagnostic
python -m app.demo --self-test

# Search Diagnostics & Multi-Representation Crop Inspection
python -m app.demo --image ./data/input/target.png --search-debug

# Machine-Readable JSON Output (For automated grading harnesses)
python -m app.demo --image ./data/input/target.png --json
```

---

## 🎯 Task & Scope

This project implements the official **HH Goa 2026 Task 3** specification:

```text
Face scan
    ↓
Web / social-media search
    ↓
Real matching social-media candidate
    ↓
Blockchain commitment & independent verification
```

### Core Tenets
1. **Zero Fake Matches**: Visual search hits are only candidates, not identity proofs. The system refuses to match false identities, preferring `NO RELIABLE MATCH FOUND` whenever calibrated thresholds are unmet.
2. **Deterministic Cryptographic Provenance**: Evidence records are normalized under **RFC 8785 JSON Canonicalization Scheme (JCS)** and hashed to SHA-256 digests before on-chain commitment.
3. **Zero Biometric Leakage**: Raw 128-D vector embeddings and biometric crops are strictly prohibited from on-chain storage to preserve subject privacy.
4. **Verification Certificate Artifact**: Upon successful independent verification, the system automatically generates a standalone HTML certificate (`./output/verification_certificate_<timestamp>.html`) featuring an embedded Etherscan QR code.

---

## 🏗️ 6-Stage Pipeline Architecture

```text
+-------------------------------------------------------------------------------+
|                             6-STAGE FORENSIC PIPELINE                         |
+-------------------------------------------------------------------------------+
  [1] INPUT & FACE IDENTIFICATION --> YuNet 5-point landmark + SFace 128-D embedding
  [2] LIVE WEB SEARCH            --> Multi-provider SerpAPI (Google Lens + Yandex)
  [3] CANDIDATE VERIFICATION     --> Multi-face scanning & perceptual dhash check
  [4] IDENTITY DECISION          --> 3-Tier calibrated decision & runner-up margin
  [5] BLOCKCHAIN COMMITMENT      --> Ethereum Sepolia testnet or local EVM simulation
  [6] INDEPENDENT VERIFICATION   --> Read-back state query & HTML certificate generation
+-------------------------------------------------------------------------------+
```

---

## 🔬 Biometric Pipeline & Face Recognition

### 1. Face Detector: OpenCV YuNet (`face_detection_yunet_2023mar.onnx`)
- High-efficiency convolutional face detector producing bounding box coordinates $(x, y, w, h)$ and 5 facial landmarks: Left Eye, Right Eye, Nose Tip, Left Mouth Corner, Right Mouth Corner.
- Supports multi-face input images with explicit `--face-index` selection to prevent identity cross-talk.

### 2. 5-Point Landmark Affine Alignment
- Prior to feature extraction, faces are normalized via affine transformation (`FaceRecognizerSF.alignCrop`) into a canonical $112 \times 112$ pixel coordinate frame.
- Eliminates in-plane roll, perspective skew, and inconsistent cropping variations.

### 3. Feature Embedder: OpenCV SFace (`face_recognition_sface_2021dec.onnx`)
- Generates a 128-dimensional deep feature representation.
- Strictly $L_2$-normalized onto the unit hypersphere:
  $$\|\mathbf{v}\|_2 = \sqrt{\sum_{i=1}^{128} v_i^2} = 1.0$$

### 4. Authoritative Cosine Similarity
- Cosine similarity between unit vectors simplifies to the Euclidean dot product:
  $$\text{sim}(\mathbf{u}, \mathbf{v}) = \sum_{i=1}^{128} u_i v_i$$
- Mathematical guarantees:
  - Exact self-identity: $\text{sim}(\mathbf{u}, \mathbf{u}) = 1.0000$ (100.0% calibration pass)
  - Commutativity: $\text{sim}(\mathbf{u}, \mathbf{v}) = \text{sim}(\mathbf{v}, \mathbf{u})$
  - Range bound: $[-1.0, 1.0]$ with zero-norm safety.

### 5. Multi-Face Candidate Scanning
- Every candidate image retrieved from the web is scanned for all human faces.
- Each detected face is individually aligned, embedded, and compared against the target face. The system selects the strongest legitimate face and records `matched_face_index` and `candidate_faces_count`.

### 6. Three-Tier Identity Decision Engine (Clustered & Statistical Relative Scoring)
To prevent both false-positive matching and duplicate-hit margin distortion, candidates are evaluated using near-duplicate clustering and relative Z-score statistical separation:

1. **Near-Duplicate Candidate Clustering Pass**:
   - Candidates with face embeddings or perceptual image dhash hamming distance within tight bounds (cosine similarity $\ge 0.90$ or dhash distance $\le 10$) are grouped into single identity/photo clusters.
   - Eliminates margin penalties caused by multiple web sources indexing the exact same target photo.

2. **Runner-Up Margin Analysis ($\Delta$)**:
   - Margin is calculated as $\Delta = \text{top\_score} - \text{second\_score}$ in percentage points.
   - Mathematically consistent display: if top = 95.6% and second = 95.5%, displayed margin is exactly `0.1 percentage points`.

3. **URL Source Quality Classification**:
   - Classifies search result / directory pages (e.g. `/pub/dir/`, `/search`) as `SEARCH RESULT PAGE — NOT A DIRECT PROFILE` to distinguish identity evidence from source page quality.

- **`HIGH CONFIDENCE MATCH`**:
  - Cosine Similarity $\ge 0.440$ (absolute floor)
  - Non-cluster separation margin $\Delta \ge 0.035$
  - Per-query statistical outlier check passed ($Z \ge 2.0$)
  - Image quality verified (acceptable sharpness/resolution)
- **`POSSIBLE MATCH — REVIEW`**:
  - Cosine Similarity $\in [0.363, 0.440)$, OR margin $\Delta < 0.035$, OR lack of statistical separation ($Z < 2.0$), OR degraded image quality.
- **`NO RELIABLE MATCH FOUND`**:
  - Cosine Similarity $< 0.363$ (OpenCV SFace baseline threshold) or 0 faces detected in candidates.

---

## 🌐 Genuine Multi-Provider Web Discovery

### Search Providers & Crawlability Rationale
The pipeline queries and merges candidates across multiple visual search engines simultaneously:
1. **SerpAPI Google Lens** (`google_lens` engine): Primary visual match indexing.
2. **SerpAPI Yandex Images** (`yandex_images` engine): Facial-feature visual similarity search for enhanced non-celebrity recall.
3. **Microsoft Bing Visual Search** (`bing` engine): Supplementary visual search.

**Design Rationale**: Crawlability differs significantly across platforms. Social networks like Instagram block search crawlers via auth walls and `robots.txt`, so un-mirrored Instagram posts rarely appear in search indexes. Conversely, platforms like LinkedIn design public directory pages specifically for search indexability. Combining Google Lens and Yandex Images maximizes candidate recall across public web sources.

### Search Provenance
Every search candidate includes verifiable provenance metadata:
- Search Provider (`SerpAPI (Google Lens)`, `SerpAPI (Yandex Images)`, `Bing Visual Search`)
- Candidate Webpage URL (Full untruncated absolute URL)
- Candidate Image Thumbnail URL
- Search Rank & Domain / Platform
- Detected Face Count & Matched Face Index
- Measured Biometric Cosine Similarity

---

## 📄 Verification Certificate Artifact

Upon completing Stage 8 (independent on-chain verification PASS), the pipeline automatically generates a single-file HTML certificate saved to:
`./output/verification_certificate_<timestamp>.html`

The certificate contains:
- Input image SHA-256 checksum
- Matched candidate URL, platform, and decision tier
- Cosine similarity + statistical margin
- On-chain transaction hash and contract address
- Base64-embedded QR code linking directly to Etherscan Sepolia (`https://sepolia.etherscan.io/tx/<txhash>`)

---

## ⛓️ Blockchain Layer & Evidence Verification

### Smart Contract: `FingerprintRegistry.sol`
- **Solidity Version**: `^0.8.20`
- **Storage Mapping**: `mapping(bytes32 => uint256) public storedAt`
- **State Proof**: Idempotent storage mechanism logging `(bytes32 fingerprint, uint256 timestamp, string projectId)`.

### Execution Modes
- **Live Ethereum Sepolia Testnet**: Set `BLOCKCHAIN_PRIVATE_KEY` and `BLOCKCHAIN_CONTRACT_ADDRESS` in `.env`.
- **Local Simulated EVM**: When private keys are omitted, the pipeline defaults to `SimulatedClient`. This mode executes the full hash verification lifecycle in-memory with zero gas costs and is **explicitly labeled** `[SIMULATION]` across CLI logs and outputs.

### Cryptographic Fingerprinting (RFC 8785)
1. The forensic record is normalized according to the **RFC 8785 JSON Canonicalization Scheme (JCS)**.
2. The canonical payload is hashed using SHA-256 to create a 32-byte deterministic digest:
   $$\text{Digest} = \text{SHA-256}(\text{JCS}(\text{EvidenceRecord}))$$
3. The digest is written to the blockchain registry.

### Independent Verification Protocol
Following commitment, the verifier:
1. Re-serializes and re-hashes the local evidence record.
2. Independently queries contract storage (`exists(bytes32)`).
3. Asserts:
   $$\text{LocalDigest} \equiv \text{OnChainDigest} \quad \implies \quad \text{Status: PASS [OK]}$$

### Tamper Detection & Avalanche Effect
To prove tamper resistance, the verification suite modifies a single evidence field and re-hashes:
- Original Hash $\equiv$ On-Chain Hash $\rightarrow$ **PASS**
- Tampered Hash $\not\equiv$ On-Chain Hash $\rightarrow$ **FAIL (DIVERGENCE)**
- Bit-level divergence is measured across all 256 hash bits ($\sim 50\%$ divergence due to SHA-256 avalanche dynamics).

---

## 🎥 Screen Recording Walkthrough Guide

For judges scrutinizing the live hackathon demonstration video, record two short clips:

### Clip 1: Successful Match Demonstration (30 seconds)
Run the main judge pipeline:
```bash
python -m app.demo --image ./data/input/target.png --judge-mode
```
Demonstrate:
1. **[1] Face Scan & Detection**: YuNet facial bounding box $(x,y,w,h)$, confidence, and SFace 128-D embedding.
2. **[2 & 3] Live Search & Candidate Table**: SerpAPI multi-provider search results and full source URLs.
3. **[4] Identity Decision**: Cosine similarity, runner-up margin ($\Delta$), and source URL classification (`SEARCH RESULT PAGE` vs `DIRECT PROFILE`).
4. **[5 & 6] Blockchain & Certificate**: On-chain SHA-256 fingerprint commitment, independent re-verification, tamper avalanche proof, and generated HTML verification certificate (`./output/verification_certificate_<timestamp>.html`).

### Clip 2: Negative-Case Fixture Demonstration (15 seconds)
Immediately after Clip 1, run:
```bash
python -m app.demo --demo-negative-case
```
Demonstrate:
1. Pipeline executing against an un-indexed / non-matching target image.
2. Stage 5 outputting `NO RELIABLE MATCH FOUND`.
3. System correctly rejecting candidate hits without hallucinating false matches.

---

## 🧪 Test Suite

All unit and integration tests execute hermetically:

```bash
python -m pytest -q
```

### Current Test Results
```text
94 passed in 1.07s
```

Test coverage includes:
- `tests/test_bugs_abc.py`: Bug A identity margin calculation, Bug B search summary counts, Bug C URL classification, HTML certificate generation, negative-case fixture.
- `tests/test_similarity.py`: Strict vector math, bounds $[-1.0, 1.0]$, zero-norm safety, symmetry, self-identity.
- `tests/test_alignment.py`: 5-point landmark affine transformation matrix and inter-ocular distance.
- `tests/test_calibration.py`: Unit norm invariant check ($\|\mathbf{v}\| = 1.0$) and self-match validation.
- `tests/test_face.py`: Face detection bounding boxes, landmark formatting, embedding generation.
- `tests/test_identity_decision.py`: 3-tier classification thresholds, runner-up margin constraints, quality gating.
- `tests/test_search.py`: Candidate normalization, URL deduplication, multi-provider interfaces.
- `tests/test_blockchain.py`: Smart contract ABI, RFC 8785 canonical serialization, SHA-256 hashing, EVM simulation, tamper avalanche proof.
- `tests/test_demo.py`: Command line flags, `--image`, `--face-index`, `--judge-mode`, `--self-test`, `--search-debug`, `--demo-negative-case`.

---

## 🔒 Security & Privacy

- **No Secrets in Version Control**: `.env` is explicitly ignored in `.gitignore`.
- **Biometric Data Protection**: Facial vectors and cropped facial images are processed strictly in volatile memory and are **never committed to the blockchain**.
- **Zero Hallucination / Zero Manipulation**: Similarity scores and search candidates represent actual model outputs and genuine search responses.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
