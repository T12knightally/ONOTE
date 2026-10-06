import os
import sys
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from smg_validation import (
    extract_guitar_diagnostic_scores,
    extract_structural_score,
    validate_smg_output,
)

_KC_ROOT_ENV = os.environ.get("KNOWLEDGECLAW_ROOT")
KNOWLEDGECLAW_ROOT = Path(_KC_ROOT_ENV) if _KC_ROOT_ENV else Path(__file__).resolve().parents[2]
if str(KNOWLEDGECLAW_ROOT) not in sys.path:
    sys.path.insert(0, str(KNOWLEDGECLAW_ROOT))

DATA_ROOT = os.environ.get("MUSICBENCH_ROOT", r"D:\MusicBench")

from integration.musicbench_rag_adapter import rag_system_message, retrieve_task_context

API_KEY = os.environ.get("APIYI_API_KEY", "")
if not API_KEY and os.path.exists("api_key_apiyi.txt"):
    API_KEY = open("api_key_apiyi.txt", encoding="utf-8").read().strip()
if not API_KEY:
    raise RuntimeError("APIYI_API_KEY is not set. Export it or create api_key_apiyi.txt.")

API_URL = os.environ.get("APIYI_BASE_URL", "https://api.apiyi.com/v1")
GENERATOR_MODEL = os.environ.get("SMG_GENERATOR_MODEL", "gemini-2.5-pro")
PRIMARY_JUDGE_MODEL = os.environ.get("SMG_PRIMARY_JUDGE_MODEL", "gpt-5")
CROSS_FAMILY_JUDGE_MODEL = os.environ.get("SMG_CROSS_FAMILY_JUDGE_MODEL", "gemini-2.5-pro")
TOTAL_SONGS = 80
client = OpenAI(api_key=API_KEY, base_url=API_URL)

_RAG_CACHE = {}


def get_rag_context(task):
    if task not in _RAG_CACHE:
        _RAG_CACHE[task], _ = retrieve_task_context("smg", task)
    return _RAG_CACHE[task]


def generate_score(composer_prompt, condition="base", max_retries=3):
    rag_context = get_rag_context(_CURRENT_TASK) if condition == "rag" else ""
    messages = [rag_system_message(rag_context, "smg")] if rag_context else []
    messages.append({"role": "user", "content": composer_prompt})
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=GENERATOR_MODEL, messages=messages,
                temperature=0.8, max_tokens=2048)
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "limit" in str(e).lower() or "429" in str(e):
                time.sleep((attempt + 1) * 3)
            else:
                return f"[Generation Error] {e}"
    return "[Generation Error] Request failed after retries."


def evaluate_score(score_text, judge_prompt, model, max_tokens=4000, max_retries=3):
    for attempt in range(max_retries):
        try:
            request = {
                "model": model,
                "messages": [
                    {"role": "system", "content": judge_prompt},
                    {"role": "user", "content": f"Notation output to evaluate:\n{score_text}"},
                ],
            }
            # Newer GPT-5 endpoints use max_completion_tokens and default sampling.
            if model.lower().startswith("gpt-5"):
                request["max_completion_tokens"] = max_tokens
            else:
                request["max_tokens"] = max_tokens
                request["temperature"] = 0.1
            response = client.chat.completions.create(
                **request)
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "limit" in str(e).lower() or "429" in str(e):
                time.sleep((attempt + 1) * 3)
            else:
                return f"[Judge Error: {model}] {e}"
    return f"[Judge Error: {model}] Request failed after retries."


STAFF_COMPOSER_PROMPT = """
Generate a symbolic notation sequence that satisfies the declared constraints.

Key: C major
Meter: 4/4
Required length: exactly 8 measures
Output format: valid ABC notation only, including X:, T:, M:, L:, and K: headers.
Use M:4/4, L:1/4, and K:C. Separate measures with | and use standard ABC note, rest, and duration tokens.
Each measure must total exactly four quarter-note beats. Do not include prose outside the ABC notation.
"""

STAFF_CRITIC_PROMPT = """
Evaluate only the structural compliance of the supplied ABC notation. Do not assess or mention any non-structural property.

Check:
1. Syntax and parseability: required ABC headers, valid note and rest tokens, bar delimiters, and a complete parseable output.
2. Rhythmic validity: each measure in 4/4 totals exactly 4.0 quarter-note beats.
3. Required structure: exactly 8 complete measures and the required headers.

Use the disclosed 1--5 structural-compliance scale. Score 5 when all checks pass, 3 when about half pass, and 1 when none pass; use intermediate values for intermediate compliance. Do not average separate subscores.
Return only: Structural Compliance Score: [score]/5
"""

GUITAR_COMPOSER_PROMPT = """Generate a symbolic guitar-tablature sequence that satisfies the declared constraints.
Key: C Major
Time Signature: 4/4
Length: 4 measures.
Format: Please only output standard ASCII guitar tablature.

[Guitar Tablature Layout Syntax]
Please strictly adhere to the "monospace character grid" rules:
The staff consists of 6 lines, representing the strings from top to bottom: e, B, G, D, A, E.
Use | as bar lines.
Each measure is strictly divided into exactly 16 character positions. Every single number (fret) or hyphen (-) occupies one position.
Therefore, between two | bar lines, every single string must contain exactly 16 characters! Four characters represent one beat.
At the very top of the tablature, please add a beat indicator row, marking the exact positions of beats 1, 2, 3, and 4.

[Correct Format Example]
(Each displayed measure is exactly 16 characters wide.)
Beat|1 . . . 2 . . . 3 . . . 4 . . . |
e   |----------------|
B   |--------1-------|
G   |----0-----------|
D   |--2-------------|
A   |3---------------|
E   |----------------|

Structural constraints: keep six string rows aligned, use exactly 16 character positions per measure on every string, and separate measures with |. Use valid fret tokens. For notes in the same column, keep the disclosed maximum fret-span proxy at 7 frets. These are operational structural constraints, not a complete playability test.
"""

GUITAR_CRITIC_PROMPT = """Evaluate only the structural compliance of the supplied ASCII guitar tablature.

Check:
1. Layout and timing: six aligned string rows and exactly 16 character positions per measure between bar lines.
2. Syntax validity: valid string rows, fret tokens, separators, measure widths, and exactly 4 requested measures.
3. Fingering-constraint proxy: for simultaneous notes in one column, check the disclosed maximum 7-fret span. This is an operational proxy, not a complete playability measure.

Use the disclosed 1--5 scale. Score 5 when all relevant checks pass, 3 for partial compliance, and 1 when most checks fail; use intermediate values for intermediate compliance. Do not average subscores.
Return only these fields:
Layout Score: [score]/5
Fingering-Constraint Score: [score]/5
Structural Compliance Score: [score]/5
"""

JIANPU_COMPOSER_PROMPT = """
Generate a symbolic Jianpu notation sequence that satisfies the declared constraints.

Key Signature: C Major (1=C)
Time Signature: 4/4
Length: 8 measures.
Format: Please only output the structured text-based numbered musical notation (Jianpu) with precise fractional rhythmic values as provided below.

[Structured Jianpu Syntax]
You must strictly use the following format to generate each note or rest: [Pitch Modifier][Note Number]([Duration Fraction])
Note Number: Use 1-7 to represent pitch, and 0 for a rest.
Pitch Modifier: Add _ before the number to indicate one octave lower (e.g., _5), and __ for two octaves lower (e.g., __2). No prefix means the middle register (e.g., 3), and ^ indicates one octave higher.
Duration Fraction: Strictly represented by fractions enclosed in parentheses. A quarter note is written as (1/4), an eighth note as (1/8), a sixteenth note as (1/16), a dotted eighth note as (3/16), and a half note as (1/2).
Measures and Separation: Use | to separate measures. Use a single space to separate notes.

[Example of Correct Format] (Note: The fractions inside the parentheses within a single measure must sum up to exactly 1)
3(1/4) _5(1/4) 0(1/8) 3(1/16) 4(5/16) | _6(3/16) __2(1/16) 1(1/4) _6(1/2) |

Each measure must total exactly one whole note (four quarter-note beats). Output notation only, with no prose.
"""

JIANPU_CRITIC_PROMPT = """
Evaluate only the structural compliance of the supplied Jianpu notation.

Check:
1. Syntax and token validity: valid pitch, rest, duration, grouping, and measure-delimiter tokens.
2. Rhythmic validity: each 4/4 measure totals exactly four quarter-note beats.
3. Required structure: exactly 8 measures with complete measure delimiters.

Use the disclosed 1--5 structural-compliance scale. Score 5 when all checks pass, 3 when about half pass, and 1 when none pass; use intermediate values for intermediate compliance. Do not average separate subscores.
Return only: Structural Compliance Score: [score]/5
"""

TASK_CONFIG = {
    "staff": {
        "composer_prompt": STAFF_COMPOSER_PROMPT,
        "critic_prompt": STAFF_CRITIC_PROMPT,
        "expected_measures": 8,
        "output_csv": os.path.join(DATA_ROOT, "data", "ai_music_evaluation_results.csv"),
        "critic_max_tokens": 4000,
    },
    "guitar": {
        "composer_prompt": GUITAR_COMPOSER_PROMPT,
        "critic_prompt": GUITAR_CRITIC_PROMPT,
        "expected_measures": 4,
        "output_csv": os.path.join(DATA_ROOT, "data", "ai_music_evaluation_results_guitar.csv"),
        "critic_max_tokens": 8000,
    },
    "jianpu": {
        "composer_prompt": JIANPU_COMPOSER_PROMPT,
        "critic_prompt": JIANPU_CRITIC_PROMPT,
        "expected_measures": 8,
        "output_csv": os.path.join(DATA_ROOT, "data", "ai_music_evaluation_results_JIAN.csv"),
        "critic_max_tokens": 4000,
    },
}


def run_composer(task, output_csv=None):
    cfg = TASK_CONFIG[task]
    condition = _CURRENT_CONDITION
    if output_csv is None:
        output_path = Path(cfg["output_csv"])
        output_csv = str(output_path.with_name(f"{output_path.stem}_{condition}{output_path.suffix}"))
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting SMG pipeline for {TOTAL_SONGS} notation outputs.")

    results = []
    processed_count = 0
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        df_old = pd.read_csv(output_csv)
        required_columns = {
            "Item ID", "Task", "Retrieval Condition", "Generator Model",
            "Primary Judge Model", "Cross-Family Judge Model",
            "Primary Structural Compliance Score", "Cross-Family Structural Compliance Score",
        }
        if not required_columns.issubset(df_old.columns):
            raise RuntimeError(
                "The existing CSV uses a legacy schema. It was left untouched; "
                "provide a new --output path to avoid mixing rubric versions."
            )
        if not df_old.empty and set(df_old["Retrieval Condition"].dropna()) != {condition}:
            raise RuntimeError(
                "The existing CSV belongs to a different retrieval condition. "
                "Provide a separate --output path."
            )
        if not df_old.empty and set(df_old["Generator Model"].dropna()) != {GENERATOR_MODEL}:
            raise RuntimeError(
                "The existing CSV belongs to a different generator model. "
                "Provide a separate --output path."
            )
        if not df_old.empty and set(df_old["Primary Judge Model"].dropna()) != {PRIMARY_JUDGE_MODEL}:
            raise RuntimeError(
                "The existing CSV belongs to a different primary judge model. "
                "Provide a separate --output path."
            )
        if not df_old.empty and set(df_old["Cross-Family Judge Model"].dropna()) != {CROSS_FAMILY_JUDGE_MODEL}:
            raise RuntimeError(
                "The existing CSV belongs to a different cross-family judge model. "
                "Provide a separate --output path."
            )
        results = df_old.to_dict("records")
        processed_count = len(results)
        print(f"Found {processed_count} compatible rows; resuming from that point.")

    for i in tqdm(range(processed_count, TOTAL_SONGS), initial=processed_count, total=TOTAL_SONGS):
        tqdm.write(f"\nGenerating notation output {i + 1}...")
        notation_output = generate_score(cfg["composer_prompt"], condition=condition)

        if notation_output.startswith("[Generation Error]"):
            tqdm.write(f"Generation {i + 1} failed: {notation_output}")
            continue

        deterministic_results = validate_smg_output(
            task, notation_output, cfg["expected_measures"])
        tqdm.write(f"Primary structural review: {i + 1}...")
        primary_reply = evaluate_score(
            notation_output, cfg["critic_prompt"], PRIMARY_JUDGE_MODEL,
            max_tokens=cfg["critic_max_tokens"])
        cross_family_reply = evaluate_score(
            notation_output, cfg["critic_prompt"], CROSS_FAMILY_JUDGE_MODEL,
            max_tokens=cfg["critic_max_tokens"])

        record = {
            "Item ID": i + 1,
            "Task": task,
            "Retrieval Condition": condition,
            "Generator Model": GENERATOR_MODEL,
            "Primary Judge Model": PRIMARY_JUDGE_MODEL,
            "Cross-Family Judge Model": CROSS_FAMILY_JUDGE_MODEL,
            "Generated Notation": notation_output,
            **deterministic_results,
            "Primary Structural Compliance Score": extract_structural_score(primary_reply),
            "Primary Judge Output": primary_reply,
            "Cross-Family Structural Compliance Score": extract_structural_score(cross_family_reply),
            "Cross-Family Judge Output": cross_family_reply,
        }
        if task == "guitar":
            record.update({
                f"Primary {key.replace('_', ' ').title()}": value
                for key, value in extract_guitar_diagnostic_scores(primary_reply).items()
            })
        results.append(record)

        df_current = pd.DataFrame(results)
        df_current.to_csv(output_csv, index=False, encoding='utf-8-sig')
        time.sleep(3)

    print(f"\nRun complete. Results saved to: {output_csv}")


_CURRENT_TASK = "staff"
_CURRENT_CONDITION = "base"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Structurally constrained music generation evaluation")
    parser.add_argument("--task", choices=list(TASK_CONFIG), default="staff",
                        help="Task: staff / guitar / jianpu")
    parser.add_argument("--condition", choices=["base", "rag"], default="base",
                        help="Use no retrieved evidence (base) or include the retrieved evidence packet (rag).")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()
    global _CURRENT_TASK, _CURRENT_CONDITION
    _CURRENT_TASK = args.task
    _CURRENT_CONDITION = args.condition
    run_composer(args.task, output_csv=args.output)


if __name__ == "__main__":
    main()
