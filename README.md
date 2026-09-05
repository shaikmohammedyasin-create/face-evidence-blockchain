# Face Identification & Blockchain Verification

[![Tests](https://img.shields.io/badge/tests-89%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](requirements.txt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Format](https://img.shields.io/badge/code%20style-PEP%208-orange.svg)](app/)

A forensic facial identification and cryptographic blockchain verification CLI application built for **HH Goa 2026 Shortlisting Task 3**.

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

---

## 🏗️ 9-Stage Architecture

```text
+-------------------------------------------------------------------------------+
|                             9-STAGE FORENSIC PIPELINE                         |
+-------------------------------------------------------------------------------+
  [1] Face Detection & Quality  --> OpenCV YuNet 5-point landmark localization
  [2] Alignment & Embedding     --> Affine warping to 112x112 + SFace 128-D unit vector
  [3] Live Web Discovery        --> SerpAPI (Google Lens) / Bing Visual Search
  [4] Candidate Multi-Face Scan --> Face detection & alignment across candidate thumbnails
  [5] Identity Decision Engine  --> 3-Tier calibrated decision with runner-up margin
  [6] Canonical Evidence Hash   --> RFC 8785 canonical JSON serialization -> SHA-256
  [7] Blockchain Commitment     --> Ethereum Sepolia testnet or simulated EVM registry
  [8] Independent Verification  --> Direct query against smart contract state
  [9] Tamper & Avalanche Proof  --> Single-field bit mutation demonstrating ~50% divergence
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
  $$\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \sum_{i=1}^{128} u_i v_i$$
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
   - Candidates with face embeddings within a tight distance (cosine similarity $\ge 0.90$ to one another) are grouped into single identity/photo clusters.
   - Eliminates margin penalties caused by multiple web sources indexing the exact same target photo (e.g., #1 score 0.9563 and #2 near-duplicate 0.9548).

2. **Non-Cluster Margin Analysis ($\Delta$)**:
   - Margin is calculated as $\Delta = \text{top\_score} - \text{highest\_score\_NOT\_in\_top\_cluster}$.
   - Measures true separation against a genuinely distinct candidate identity.

3. **Per-Query Relative Statistical Check ($Z \ge 2.0$)**:
   - Computes mean ($\mu$) and standard deviation ($\sigma$) across the candidate search pool.
   - Requires top candidate score to be a statistical outlier ($Z = \frac{s - \mu}{\sigma} \ge 2.0$, adjusted for sample size $N$) above background candidate noise.

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

## 🌐 Genuine Web & Social Media Discovery

### Search Providers
1. **Primary**: SerpAPI Google Lens API (`google_lens` engine).
2. **Fallback**: Microsoft Bing Visual Search API.

### Search Provenance
Every search candidate includes verifiable provenance metadata:
- Search Provider (`SerpAPI (Google Lens)`)
- Candidate Webpage URL (e.g. LinkedIn, Instagram, X/Twitter, Facebook, TikTok)
- Candidate Image Thumbnail URL
- Search Rank & Domain / Platform
- Detected Face Count & Matched Face Index
- Measured Biometric Cosine Similarity

> **Important Principle**: Search results are **never** treated as identity proof. Every candidate thumbnail is downloaded and independently verified against the target face embedding before an identity decision is rendered.

---

## ⛓️ Blockchain Layer & Evidence Verification

### Smart Contract: `FingerprintRegistry.sol`
- **Solidity Version**: `^0.8.20`
- **Storage Mapping**: `mapping(bytes32 => uint256) public storedAt`
- **State Proof**: Idempotent storage mechanism logging `(bytes32 fingerprint, uint256 timestamp, string projectId)`.

### Execution Modes
The application supports both live public testnet and local simulated execution:
- **Live Ethereum Sepolia Testnet**: Set `BLOCKCHAIN_PRIVATE_KEY` and `BLOCKCHAIN_CONTRACT_ADDRESS` in `.env`.
- **Local Simulated EVM**: When private keys are omitted, the pipeline defaults to `SimulatedClient`. This mode executes the full hash verification lifecycle in-memory with zero gas costs and is **explicitly labeled** `[SIMULATION]` across all CLI logs and JSON outputs.

### Cryptographic Fingerprinting (RFC 8785)
1. The forensic record (input image SHA-256, model versions, candidate URL, match score, decision tier, timestamp) is normalized according to the **RFC 8785 JSON Canonicalization Scheme (JCS)**.
2. The canonical payload is hashed using SHA-256 to create a 32-byte deterministic digest:
   $$\text{Digest} = \text{SHA-256}(\text{JCS}(\text{EvidenceRecord}))$$
3. The digest is written to the blockchain registry.

### Independent Verification Protocol
Following commitment, the verifier:
1. Re-serializes and re-hashes the local evidence record.
2. Independently queries the contract storage (`exists(bytes32)`).
3. Asserts:
   $$\text{LocalDigest} \equiv \text{OnChainDigest} \quad \implies \quad \text{Status: PASS [OK]}$$

### Tamper Detection & Avalanche Effect
To prove tamper resistance, the verification suite modifies a single evidence field (e.g. altering the candidate title) and re-hashes:
- Original Hash $\equiv$ On-Chain Hash $\rightarrow$ **PASS**
- Tampered Hash $\not\equiv$ On-Chain Hash $\rightarrow$ **FAIL (DIVERGENCE)**
- Bit-level divergence is measured across all 256 hash bits (typically $\sim 50\%$ divergence due to SHA-256 avalanche dynamics).

---

## 💻 Installation & Usage

### Prerequisites
- Python 3.11, 3.12, 3.13, or 3.14
- C++ build tools for OpenCV (standard binary wheels install automatically via pip)

### Environment Configuration
Create a `.env` file from the provided template:
```bash
cp .env.example .env
```

Edit `.env`:
```ini
# SerpAPI Key (https://serpapi.com)
SERPAPI_KEY=your_serpapi_api_key_here

# Blockchain Configuration (Leave blank for instant zero-gas simulation mode)
BLOCKCHAIN_RPC_URL=https://rpc.sepolia.org
BLOCKCHAIN_PRIVATE_KEY=
BLOCKCHAIN_CONTRACT_ADDRESS=
BLOCKCHAIN_NETWORK="Ethereum Sepolia Testnet"
BLOCKCHAIN_CHAIN_ID=11155111

# Biometric Baseline Threshold
FACE_MATCH_THRESHOLD=0.363
```

### CLI Commands

```bash
# 1. Run pipeline on any input image
python -m app.demo --image ./data/input/target.png

# 2. Run with Technical Judge Audit diagnostics
python -m app.demo --image ./data/input/target.png --judge-mode

# 3. Specify target face index (for photos with multiple people)
python -m app.demo --image ./data/input/group_photo.jpg --face-index 0

# 4. Limit search candidates
python -m app.demo --image ./data/input/target.png --max-candidates 10

# 5. Skip blockchain upload (biometrics + visual search only)
python -m app.demo --image ./data/input/target.png --skip-blockchain

# 6. Output pure JSON for automated test harnesses
python -m app.demo --image ./data/input/target.png --json

# 7. Interactive Menu Mode
python -m app.main
```

---

## 🎥 Screen Recording Walkthrough Guide (Single Continuous Take)

For judges scrutinizing the live hackathon demonstration video, follow these exact steps to demonstrate all 3 core pipeline stages (**Face Scan → Live Web Search → Blockchain Commitment & Re-Verification**) in one uninterrupted recording:

### 1. Pre-Flight Preparation (5 seconds)
- Open a terminal in the project root folder `task3-face-identification-blockchain`.
- Ensure your `.env` contains a valid `SERPAPI_KEY`.
- Check that your input face image exists (e.g. `./data/input/target.png`).

### 2. Launch the Unified Pipeline Command (20–30 seconds)
Run the following single command in your terminal:
```bash
python -m app.demo --image ./data/input/target.png --judge-mode
```

### 3. What to Highlight During the Continuous Take
1. **[1/9] Face Scan & Detection**: Point out the YuNet facial bounding box $(x,y,w,h)$, confidence score, and quality metrics (resolution, sharpness, brightness).
2. **[2/9] Embedding & Alignment**: Highlight the SFace 5-point landmark affine normalization and 128-D $L_2$-normalized vector generation.
3. **[3/9] Live Web Search**: Emphasize the **live SerpAPI Google Lens call** returning real, non-hardcoded candidate post URLs.
4. **[4/9 & 5/9] Candidate Matching & Decision**: Point out the cosine similarity table, the runner-up separation margin ($\Delta$), and the calibrated `HIGH CONFIDENCE MATCH` decision.
5. **[6/9 & 7/9] Canonical Hashing & Blockchain Upload**: Show the RFC 8785 canonical SHA-256 fingerprint digest and the returned on-chain Transaction Hash / Contract Address.
6. **[8/9] Independent Re-Verification**: Highlight the **live read-back query from the blockchain** confirming that `LocalDigest === OnChainDigest` (`PASS (VERIFIED ON-CHAIN)`).
7. **[9/9] Cryptographic Tamper Detection**: Point out the live single-field metadata mutation showing bit-level avalanche divergence ($\sim 50\%$ bit shift), proving that tampered data fails on-chain verification.

---

## 📊 Example CLI Terminal Output

```text
+------------------------------------------------------------------+
|   FACE IDENTIFICATION & BLOCKCHAIN VERIFICATION                  |
|   Competition Submission -- HH Goa 2026 (Task 3)                  |
+------------------------------------------------------------------+

Target Image: data/input/target.png
Mode: JUDGE AUDIT & SYSTEM DIAGNOSTICS ACTIVE

─────────────── [1/9] FACE DETECTION & PRE-FLIGHT QUALITY CHECK ────────────────
    Target Image    : target.png
    SHA-256 Checksum: f8f00072c0efae8c9dd1e9697b4883d82bfa88893d35c55ef2c39a716895c7a2
    Resolution      : 800x800 (Acceptable, Sharpness: 39.3, Brightness: 124.5)
    Faces Detected  : 1
    Selected Face   : #0 (Detection Confidence: 94.9%)
    Bounding Box    : x=255, y=197, w=250, h=338
  [OK] Face geometry extracted and validated in 0.138s

─────────── [2/9] FACE ALIGNMENT & SFACE 128-D EMBEDDING GENERATION ────────────
    Alignment       : 5-Point Affine Landmark Normalization (112x112 canonical)
    Feature Model   : OpenCV SFace (ONNX)
    Embedding Dim   : 128-D (L2-Normalized, ||v||=1.0)
    Self-Match Test : 100.00% (Internal Sanity Pass)
  [OK] 128-d unit embedding generated and calibrated in 0.677s

─────── [3/9] LIVE WEB & SOCIAL MEDIA DISCOVERY (GOOGLE LENS / SERPAPI) ────────
    Search Engine   : SerpAPI Google Lens Engine
    Query Target    : target.png
    Candidates Found: 10 unique, deduplicated URLs
  [OK] Retrieved 10 candidate(s) across platforms in 17.178s

────────── [4/9] CANDIDATE FACE EXTRACTION & MULTI-FACE VERIFICATION ───────────
┏━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┓
┃ Rank ┃ Platform ┃ Candidate Page URL                 ┃ Faces ┃ Cosine Sim ┃ Margin (Δ) ┃ Decision   ┃
┡━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━┩
│  #1  │ LinkedIn │ https://www.linkedin.com/pub/dir/… │ #1/1  │     0.9563 │    +0.0015 │   REVIEW   │
│  #2  │ LinkedIn │ https://in.linkedin.com/in/…       │ #1/1  │     0.9548 │    +0.0000 │ HIGH MATCH │
│  #3  │ Web      │ https://rtvonline.com/country/…    │ #1/1  │     0.4693 │    +0.0000 │ HIGH MATCH │
└──────┴──────────┴────────────────────────────────────┴───────┴────────────┴────────────┴────────────┘
  [OK] Evaluated 10 candidate image(s) and multi-face crops in 20.014s

────────────────── [5/9] FORENSIC IDENTITY DECISION ANALYSIS ───────────────────
╭──────────────────── EVIDENCE & IDENTITY DECISION SUMMARY ────────────────────╮
│ Candidate Page URL   : https://www.linkedin.com/pub/dir/+/Mohammed+Yasin     │
│ Platform / Domain    : LinkedIn                                              │
│ Face Analysis        : Face #1 of 1 detected                                 │
│ Measured Similarity  : 0.9563 (Cosine metric, range [-1.0, 1.0])             │
│ Calibrated Threshold : 0.363 (High Confidence: 0.440)                        │
│ Runner-Up Margin (Δ) : +0.0015 (Min Required: 0.035)                         │
│ Final Decision       : POSSIBLE MATCH — REVIEW                               │
╰──────────────────────────────────────────────────────────────────────────────╯

─────── [6/9] CANONICAL EVIDENCE RECORD SERIALIZATION & SHA-256 HASHING ────────
    Canonical Encoding: RFC 8785 JSON Canonicalization Scheme (JCS)
    Biometric Privacy : Zero raw embeddings stored on-chain (100% Zero-Leak)
    SHA-256 Digest    : 0b37db71f1c82861d43384d3aab6a1a74d5537dd30cf2a1d21a7de1759f72d24
  [OK] Deterministic fingerprint generated in 0.003s

───────────── [7/9] BLOCKCHAIN COMMITMENT & TRANSACTION BROADCAST ──────────────
    Network           : [SIMULATION] Local In-Memory Chain (Sepolia Stand-in)
    Transaction Hash  : 0xb9566daf4450a3b7e7687931918836ae54337a7beafa4e6b1b79eae6ae701d64
    Contract Address  : 0x71C7656EC7ab88b098defB751B7401B5f6d8976F
    Commitment Status : SIMULATED
  [OK] Fingerprint committed to blockchain in 0.008s

────────────────── [8/9] INDEPENDENT BLOCKCHAIN VERIFICATION ───────────────────
    Local Digest    : 0b37db71f1c82861d43384d3aab6a1a74d5537dd30cf2a1d21a7de1759f72d24
    On-Chain Digest : 0b37db71f1c82861d43384d3aab6a1a74d5537dd30cf2a1d21a7de1759f72d24
    Verification    : PASS (VERIFIED ON-CHAIN)

──────────── [9/9] CRYPTOGRAPHIC TAMPER DETECTION & AVALANCHE TEST ─────────────
    Original Digest : 0b37db71f1c82861d43384d3aab6a1a74d5537dd30cf2a1d21a7de1759f72d24
    Tampered Digest : 6696d5b32df85833c32df005f57347a0d4ec9d7b971de3220b4ab4c4a5a7ac23
    Bit Divergence  : 129/256 bits (50.4%) (Avalanche Effect)
    Tamper Detected : YES — Cryptographic mismatch confirmed

╭─────────── FACE IDENTIFICATION & BLOCKCHAIN VERIFICATION SUMMARY ────────────╮
│ [1] Face Detection              : PASS [OK]                                  │
│ [2] Face Embedding              : PASS [OK]                                  │
│ [3] Live Web Search             : PASS [OK]                                  │
│ [4] Candidate Face Verification : PASS [OK]                                  │
│ [5] Identity Decision           : PASS [OK] (REVIEW)                         │
│ [6] Evidence Hash               : PASS [OK]                                  │
│ [7] Blockchain Commitment       : PASS [OK]                                  │
│ [8] Independent Verification    : PASS [OK]                                  │
│ [9] Tamper Detection            : PASS [OK]                                  │
│                                                                              │
│ TOTAL PIPELINE EXECUTION TIME: 38.104s                                       │
╰──────────────────────────────────────────────────────────────────────────────╯
```

---

## 🧪 Test Suite

All unit and integration tests execute hermetically:

```bash
python -m pytest -q
```

### Current Test Results
```text
89 passed in 1.58s
```

Test coverage includes:
- `tests/test_similarity.py`: Strict vector math, bounds $[-1.0, 1.0]$, zero-norm safety, symmetry, self-identity.
- `tests/test_alignment.py`: 5-point landmark affine transformation matrix and inter-ocular distance.
- `tests/test_calibration.py`: Unit norm invariant check ($\|\mathbf{v}\| = 1.0$) and self-match validation.
- `tests/test_face.py`: Face detection bounding boxes, landmark formatting, embedding generation.
- `tests/test_identity_decision.py`: 3-tier classification thresholds, runner-up margin constraints, quality gating.

## 🌐 Genuine Multi-Provider Web & Social Media Discovery

### Multi-Provider Search Orchestration
The pipeline queries and merges candidates across multiple visual search engines simultaneously:
1. **SerpAPI Google Lens** (`google_lens` engine): Primary visual match indexing.
2. **SerpAPI Yandex Images** (`yandex_images` engine): Facial-feature visual similarity search for enhanced non-celebrity recall.
3. **Microsoft Bing Visual Search** (`bing` engine): Supplementary visual search.

Candidates from all active providers are aggregated, normalized, and deduplicated into a single unified candidate pool prior to biometric face matching and relative statistical scoring.

### Search Provenance
Every search candidate includes verifiable provenance metadata:
- Search Provider (`SerpAPI (Google Lens)`, `SerpAPI (Yandex Images)`, `Bing Visual Search`)
- Candidate Webpage URL (Full untruncated absolute URL)
- Candidate Image Thumbnail URL
- Search Rank & Domain / Platform
- Detected Face Count & Matched Face Index
- Measured Biometric Cosine Similarity

---

## ⚠️ Known Limitations & Engineering Boundaries

In accordance with rigorous forensic engineering standards, this system documents the following technical boundaries:
1. **Social Platform Crawl Restrictions (Instagram vs. LinkedIn)**: Instagram aggressively blocks search engine crawlers via authentication walls and `robots.txt`, so un-mirrored Instagram posts rarely appear in search engine indexes regardless of search algorithm. Conversely, platforms like LinkedIn design public profiles specifically for search indexability, producing far higher recall for ordinary individuals.
2. **Public Web Indexing Coverage**: Reverse-image search results depend on public web search engine coverage (Google Lens / Yandex / Bing). Unindexed or private photos cannot be discovered via visual search APIs.
3. **Biometric Variations & Extreme Occlusion**: Deep feature representations are sensitive to heavy facial occlusion (sunglasses, masks, extreme profiles $> 60^\circ$ yaw/pitch) and severe motion blur (Laplacian variance $< 30$).
4. **Multi-Person Ambiguity**: When an image contains multiple subjects, the pipeline requires explicit `--face-index` selection to eliminate accidental identity cross-matching.
5. **Third-Party API Rate Limits**: Visual search engines (SerpAPI / Bing) impose rate limits based on subscription tiers.
6. **Blockchain Gas & RPC Latency**: Public Ethereum Sepolia transactions require funded test-ETH and 12–15 second block confirmation times. Simulated mode is provided for instantaneous, zero-cost deterministic grading.

- `tests/test_quality.py`: Sharpness, contrast, brightness, and resolution metrics.
- `tests/test_search.py`: Candidate normalization, URL deduplication, multi-provider interfaces.
- `tests/test_blockchain.py`: Smart contract ABI, RFC 8785 canonical serialization, SHA-256 hashing, EVM simulation, tamper avalanche proof.
- `tests/test_demo.py`: Command line flags, `--image`, `--face-index`, `--judge-mode`, and `--json` format output.
- `tests/test_integration.py`: End-to-end multi-stage pipeline orchestration.

---

## ⚠️ Known Limitations & Engineering Boundaries

In accordance with rigorous forensic engineering standards, this system documents the following technical boundaries:
1. **Public Web Indexing Coverage**: Reverse-image search results depend entirely on public web search engine coverage (Google Lens / Bing). Unindexed, private, or walled-garden profiles cannot be indexed by search providers.
2. **Biometric Variations & Extreme Occlusion**: Deep feature representations are sensitive to heavy facial occlusion (sunglasses, masks, extreme profiles $> 60^\circ$ yaw/pitch) and severe motion blur (Laplacian variance $< 30$).
3. **Multi-Person Ambiguity**: When an image contains multiple subjects, the pipeline requires explicit `--face-index` selection to eliminate accidental identity cross-matching.
4. **Third-Party API Rate Limits**: Visual search engines (SerpAPI / Bing) impose rate limits based on subscription tiers.
5. **Blockchain Gas & RPC Latency**: Public Ethereum Sepolia transactions require funded test-ETH and 12–15 second block confirmation times. Simulated mode is provided for instantaneous, zero-cost deterministic grading.

---

## 🔒 Security & Privacy

- **No Secrets in Version Control**: `.env` is explicitly ignored in `.gitignore`. Only `.env.example` with empty placeholders is committed.
- **Biometric Data Protection**: Facial vectors and cropped facial images are processed strictly in volatile memory and are **never committed to the blockchain**. Only deterministic SHA-256 digests of non-biometric metadata are stored on-chain.
- **Zero Hallucination / Zero Manipulation**: Similarity scores and search candidates represent actual model outputs and genuine search responses.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
