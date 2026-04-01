# 🎵 ONOTE: Benchmarking Omnimodal Notation Processing for Expert-level Music Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Arxiv_Link-red.svg)](#)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-blue.svg)](#)

> This is the official repository for the paper: **ONOTE: Benchmarking Omnimodal Notation Processing for Expert-level Music Intelligence**.

Omnimodal Large Language Models (LLMs) excel generally, but Symbolic Music Processing (SMP) remains a formidable challenge plagued by Western notation biases and subjective "LLM-as-a-judge" evaluation hallucinations. 

**ONOTE (Omnimodal Notation Objective Topology Examination)** is a comprehensive evaluation framework featuring a programmatic, deterministic grading pipeline across three heterogeneous notation systems. It provides a rigorous, objective testbed for diagnosing cross-modal reasoning vulnerabilities in state-of-the-art AI.

---

## 🌟 Key Features

- **Heterogeneous Notation Systems:** Extends beyond Western standard staff to include globally prevalent **Jianpu (Numbered Notation)** and instrument-specific **Guitar Tablature**.
- **Comprehensive Task Taxonomy:** Evaluates the full lifecycle of symbolic music cognition through four distinct tasks.
- **Deterministic Evaluation:** Completely eliminates subjective "LLM-as-a-judge" biases using Canonical Pitch Space Projection and Sequence Alignment.

---

## 📊 The 4 Core Tasks

ONOTE assesses models across the following orthogonal dimensions:

1. **👀 Visual Score Understanding (VSU):** Visual QA tasks requiring the model to locate and identify specific musical symbols directly from PNG score images without textual hints.
2. **🔀 Cross-Format Notation Conversion (CNC):** Translating one notation format (e.g., standard staff) into another (e.g., ASCII Guitar Tab) to test deep musicological mapping.
3. **🎧 Audio-to-Symbolic Transcription (AST):** Transcribing 10-second segmented audio chunks (MP3/WAV) into notation strings, evaluating acoustic-temporal alignment.
4. **🎹 Symbolic Music Generation (SMG):** Generating rendering-compliant symbolic codes (e.g., MuseScore, ABC) based on textual prompts, evaluated on Syntactic Renderability and Musical Aesthetics.

---

## 🚀 Deterministic Evaluation Pipeline

To combat systemic hallucinations in LLM self-evaluations, ONOTE utilizes a strict, programmatic pipeline for sequence matching. 

Generated symbols and ground-truth JSONs are projected into a flattened, chronologically ordered sequence of absolute scientific pitches:
$S_{gt} = \mathcal{F}(y^{(n)}), \quad S_{pred} = \mathcal{F}(\hat{y}^{(n)})$

Alignment accuracy heavily penalizes temporal drift and hallucinated notes via the Levenshtein Edit Distance (ED):
$$Acc(S_{gt}, S_{pred}) = \max \left(0, 1 - \frac{ED(S_{gt}, S_{pred})}{\max(|S_{gt}|, |S_{pred}|)} \right)$$

---

## 🏆 Leaderboard

Performance of state-of-the-art MLLMs evaluated on ONOTE. *(For detailed metric breakdowns, please refer to our paper).*

| Model | SMG (Score) | CNC (Acc. %) | VSU (Acc. %) | AST (Acc. %) |
| :--- | :---: | :---: | :---: | :---: |
| **Qwen3-Omni-flash** | 3.84 | 17.31 | 88.00 | 9.32 |
| **Gemini-2.5-pro** | 3.03 | 17.04 | 97.00 | 7.50 |
| **Gemini-3.1-flash-lite** | 4.47 | 17.29 | **99.00** | 7.61 |
| **Gemini-2.5-flash** | 1.31 | **46.08** | 36.00 | 2.17 |
| *...more in paper* | ... | ... | ... | ... |

*(Note: High VSU accuracy strongly contrasts with low CNC/AST scores, exposing a critical gap between visual perception and music-theoretic reasoning in current models.)*

---

## 📂 Dataset Usage

The ONOTE dataset comprises 1,120 high-quality test samples, meticulously cleaned and cross-modally aligned from sources like MusiXQA and GuitarSet.

You can download the dataset via [HuggingFace Datasets](#) or directly clone the data folder:

```bash
git clone [https://github.com/your-username/ONOTE.git](https://github.com/your-username/ONOTE.git)
cd ONOTE/dataset
