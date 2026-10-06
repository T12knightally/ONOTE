# ONOTE Evaluation Code

This repository contains scripts for evaluating multimodal models on four music-notation tasks: Visual Score Understanding (VSU), Cross-Format Notation Conversion (CNC), Audio-to-Symbolic Transcription (AST), and Structurally Constrained Music Generation (SMG).

SMG scoring covers structural compliance only. It does not rate musicality, aesthetics, creativity, or artistic preference.

## Scripts

| Script | Task | Task names | Output |
| --- | --- | --- | --- |
| `metrics/evaluate_vsu.py` | Visual Score Understanding | `notation`, `guitar`, `jianpu` | Excel workbook |
| `metrics/evaluate_cnc.py` | Cross-Format Notation Conversion | `guitar_to_staff`, `jianpu_to_staff`, `staff_to_jianpu` | Console metrics |
| `metrics/evaluate_ast.py` | Audio-to-Symbolic Transcription | `abc`, `jian`, `guitar` | Excel workbook |
| `metrics/evaluate_smg.py` | Structurally Constrained Music Generation | `staff`, `guitar`, `jianpu` | CSV file |

## Evaluation Outputs

- **VSU:** exact answer accuracy.
- **CNC:** sequence similarity (SR) over the task's normalized output sequences.
- **AST:** matching-block precision (MP) over transcribed sequences.
- **SMG:** a GPT-5 1--5 structural-compliance score. Gemini applies the same rubric as a cross-family check; these scores are stored separately and are not averaged. Deterministic syntax, measure, meter, layout, and fret-span checks are recorded in separate fields.

The CNC/AST sequence metrics are implemented in `metrics/sequence_metrics.py`. SMG validation and score extraction are implemented in `metrics/smg_validation.py`.

## Setup

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Configure the API and data paths:

- `APIYI_API_KEY`: required API key. The scripts also check for `api_key_apiyi.txt` in the current working directory.
- `APIYI_BASE_URL`: optional API endpoint; defaults to `https://api.apiyi.com/v1`.
- `MUSICBENCH_ROOT`: MusicBench data directory; defaults to `D:\\MusicBench`. The scripts look for task data under `<MUSICBENCH_ROOT>/data`.
- `KNOWLEDGECLAW_ROOT`: optional import root for the external RAG integration; defaults to the repository's parent project directory.

The RAG integration must provide `integration.musicbench_rag_adapter` with the task-retrieval and system-message functions used by the scripts. The adapter and its configured knowledge-base files are external dependencies.

For SMG, the model names can be selected with `SMG_GENERATOR_MODEL`, `SMG_PRIMARY_JUDGE_MODEL` (default `gpt-5`), and `SMG_CROSS_FAMILY_JUDGE_MODEL` (default `gemini-2.5-pro`).

## Run

Run commands from the repository root:

```bash
python metrics/evaluate_vsu.py --task notation --output vsu.xlsx
python metrics/evaluate_cnc.py --task guitar_to_staff
python metrics/evaluate_ast.py --task abc --output ast.xlsx
python metrics/evaluate_smg.py --task staff --condition base --output smg_base.csv
python metrics/evaluate_smg.py --task staff --condition rag --output smg_rag.csv
```

Each script supports `--help` for its available task names. SMG separates Base and RAG outputs and checks model, judge, and condition metadata before resuming an existing CSV.

Run the local unit tests with:

```bash
python -m unittest discover -s tests -v
```
