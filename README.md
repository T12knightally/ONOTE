# 🎵 ONOTE: Benchmarking Omnimodal Notation Processing for Expert-level Music Intelligence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Paper](https://img.shields.io/badge/Paper-Arxiv_Link-red.svg)](#)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-blue.svg)](#)

> This is the official repository for the paper: **ONOTE: Benchmarking Omnimodal Notation Processing for
Expert-level Music Intelligence**.

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

Performance of state-of-the-art **Omnimodal LLMs** evaluated on ONOTE. The table details the dual-axis generative scores and alignment accuracies across three heterogeneous notation systems. Bold values indicate the best performance in each metric.

<table>
  <thead>
    <tr>
      <th rowspan="2">Models</th>
      <th colspan="4" align="center">Standard Staff</th>
      <th colspan="4" align="center">Jianpu</th>
      <th colspan="4" align="center">Guitar Tablature</th>
    </tr>
    <tr>
      <th align="center">SMG<br><sub>(Score)</sub></th>
      <th align="center">CNC<br><sub>(Acc. %)</sub></th>
      <th align="center">VSU<br><sub>(Acc. %)</sub></th>
      <th align="center">AST<br><sub>(Acc. %)</sub></th>
      <th align="center">SMG<br><sub>(Score)</sub></th>
      <th align="center">CNC<br><sub>(Acc. %)</sub></th>
      <th align="center">VSU<br><sub>(Acc. %)</sub></th>
      <th align="center">AST<br><sub>(Acc. %)</sub></th>
      <th align="center">SMG<br><sub>(Score)</sub></th>
      <th align="center">CNC<br><sub>(Acc. %)</sub></th>
      <th align="center">VSU<br><sub>(Acc. %)</sub></th>
      <th align="center">AST<br><sub>(Acc. %)</sub></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Baichuan-Omni-1.5</td>
      <td align="center">1.24</td><td align="center"><b>18.54</b></td><td align="center">4.00</td><td align="center">3.96</td>
      <td align="center">1.39</td><td align="center">5.51</td><td align="center">19.8</td><td align="center">14.75</td>
      <td align="center">1.67</td><td align="center">6.42</td><td align="center">18.5</td><td align="center">1.53</td>
    </tr>
    <tr>
      <td>Qwen2.5-Omni-7b</td>
      <td align="center"><b>4.51</b></td><td align="center">14.27</td><td align="center">44.00</td><td align="center">3.79</td>
      <td align="center">1.07</td><td align="center">8.62</td><td align="center">65.30</td><td align="center">20.63</td>
      <td align="center">2.67</td><td align="center">7.05</td><td align="center">80.2</td><td align="center">3.30</td>
    </tr>
    <tr>
      <td>Qwen-Omni-turbo</td>
      <td align="center">2.07</td><td align="center">14.72</td><td align="center">48.00</td><td align="center">8.55</td>
      <td align="center">1.39</td><td align="center">8.86</td><td align="center">62.38</td><td align="center">14.78</td>
      <td align="center">2.79</td><td align="center">7.45</td><td align="center">60.49</td><td align="center"><b>4.32</b></td>
    </tr>
    <tr>
      <td>Qwen3-Omni-flash</td>
      <td align="center">3.84</td><td align="center">17.31</td><td align="center">88.00</td><td align="center"><b>9.32</b></td>
      <td align="center">1.86</td><td align="center">5.49</td><td align="center">82.10</td><td align="center">17.96</td>
      <td align="center">3.19</td><td align="center">4.07</td><td align="center">94.37</td><td align="center">2.55</td>
    </tr>
    <tr>
      <td>Gemini-2.5-flash</td>
      <td align="center">1.31</td><td align="center">12.98</td><td align="center">45.00</td><td align="center">4.11</td>
      <td align="center">1.52</td><td align="center">9.44</td><td align="center">46.07</td><td align="center">19.85</td>
      <td align="center">1.17</td><td align="center"><b>46.08</b></td><td align="center">36.00</td><td align="center">2.17</td>
    </tr>
    <tr>
      <td>Gemini-2.5-pro</td>
      <td align="center">3.03</td><td align="center">17.04</td><td align="center">97.00</td><td align="center">7.50</td>
      <td align="center">4.33</td><td align="center"><b>23.04</b></td><td align="center"><b>90.38</b></td><td align="center">15.67</td>
      <td align="center">3.71</td><td align="center">43.58</td><td align="center">82.72</td><td align="center">2.57</td>
    </tr>
    <tr>
      <td>Gemini-3.1-flash-lite-preview</td>
      <td align="center">4.47</td><td align="center">17.29</td><td align="center"><b>99.00</b></td><td align="center">7.61</td>
      <td align="center"><b>4.72</b></td><td align="center">13.06</td><td align="center">80.20</td><td align="center"><b>24.32</b></td>
      <td align="center"><b>3.68</b></td><td align="center">22.47</td><td align="center"><b>93.83</b></td><td align="center">1.64</td>
    </tr>
  </tbody>
</table>

*(Note: High VSU accuracy across notations strongly contrasts with overall low CNC/AST scores, exposing a critical gap between optical perception and structural music-theoretic reasoning in current architectures.)*

---

## 📂 Dataset Usage

The ONOTE dataset comprises 1,120 high-quality test samples, meticulously cleaned and cross-modally aligned from sources like MusiXQA and GuitarSet.

You can download the dataset via [HuggingFace Datasets](#) 
