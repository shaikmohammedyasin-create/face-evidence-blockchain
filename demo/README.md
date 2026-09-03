# Live Demo & Evaluation Guide

**HH Goa 2026 — Task 3: Face Identification & Blockchain Verification Pipeline**

This guide provides step-by-step instructions for testing and recording the complete pipeline.

---

## 🚀 Quick Start (Zero Setup)

The repository includes a ready-to-run synthetic test fixture at `data/input/sample_portrait.png`.

### 1. Standard Terminal Demo
```bash
python -m app.demo --image ./data/input/sample_portrait.png
```

### 2. Judge Audit & Diagnostic Mode
```bash
python -m app.demo --image ./data/input/sample_portrait.png --judge-mode
```
*Outputs ONNX model SHA-256 integrity checksums, 128-d L2 unit norm checks, cryptographic avalanche bit difference metrics, and microsecond stage timers.*

### 3. Machine-Readable JSON Output (For Grading Harnesses)
```bash
python -m app.demo --image ./data/input/sample_portrait.png --json
```

### 4. Interactive Menu Mode
```bash
python -m app.main
```

---

## 📋 12-Step Demo Recording Walkthrough

When creating a video submission, follow this 12-step sequence:

```
 1. Input Face Image Selection       — supply JPG/PNG portrait (e.g., ./data/input/person.jpg)
 2. Face Detection (OpenCV YuNet)    — extracts bounding box & 5 facial landmarks
 3. 128-d Embedding (OpenCV SFace)   — strictly L2-normalised unit vector
 4. Genuine Reverse-Image Search     — executes SerpAPI (Google Lens) / Bing Visual Search
 5. Candidate Retrieval              — downloads candidate thumbnails from public sources
 6. In-Process Face Comparison       — calculates cosine similarity (threshold 0.363)
 7. Best Match Assessment            — identifies highest-similarity candidate post
 8. Canonical Evidence Record        — RFC 8785 canonical JSON without biometric vectors
 9. SHA-256 Digest Generation        — deterministic cryptographic fingerprint
10. Blockchain Registration          — Ethereum Sepolia testnet or labelled simulation
11. Independent Re-Verification      — re-hashes source metadata & validates on-chain
12. Tamper Avalanche Demonstration   — mutates title field; proves hash divergence & failure
```

---

## 🌐 Network & API Configurations

### Visual Search Providers (configure in `.env`)
- **Primary:** `SERPAPI_KEY` ([SerpAPI Google Lens](https://serpapi.com))
- **Fallback:** `BING_SEARCH_API_KEY` ([Bing Visual Search](https://azure.microsoft.com/products/ai-services/bing-search))

### Blockchain Layer
- **Live Chain:** Set `BLOCKCHAIN_PRIVATE_KEY` and `BLOCKCHAIN_CONTRACT_ADDRESS` (Ethereum Sepolia).
- **Simulation Mode:** Leave keys blank. The pipeline automatically uses `SimulatedClient`, which runs in-memory and is clearly marked `[SIMULATION]` across all logs.

---

## 🧪 Running Hermetic Tests
```bash
python -m pytest
```
*All 56 unit and integration tests run offline without network or gas dependencies.*
