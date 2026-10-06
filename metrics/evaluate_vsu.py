import os
import sys
import base64
import mimetypes
import json
import time
import re
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

_KC_ROOT_ENV = os.environ.get("KNOWLEDGECLAW_ROOT")
KNOWLEDGECLAW_ROOT = Path(_KC_ROOT_ENV) if _KC_ROOT_ENV else Path(__file__).resolve().parents[2]
if str(KNOWLEDGECLAW_ROOT) not in sys.path:
    sys.path.insert(0, str(KNOWLEDGECLAW_ROOT))

DATA_ROOT = os.environ.get("MUSICBENCH_ROOT", r"D:\MusicBench")

from integration.musicbench_rag_adapter import retrieve_vsu_context, vsu_rag_system_message

API_KEY = os.environ.get("APIYI_API_KEY", "")
if not API_KEY and os.path.exists("api_key_apiyi.txt"):
    API_KEY = open("api_key_apiyi.txt", encoding="utf-8").read().strip()
if not API_KEY:
    raise RuntimeError("APIYI_API_KEY is not set. Export it or create api_key_apiyi.txt.")

BASE_URL = os.environ.get("APIYI_BASE_URL", "https://api.apiyi.com/v1")
VISION_MODEL = "gemini-2.5-pro"
client = OpenAI(api_key=API_KEY.strip(), base_url=BASE_URL, timeout=300.0, max_retries=2)


def extract_answer_letter(raw_response):
    """Extract a single answer option letter A/B/C/D from the model response."""
    text = (raw_response or "").strip().upper()
    m = re.search(r'^([A-D])$', text)
    if m:
        return m.group(1)
    m = re.search(r'\b([A-D])\b', text)
    if m:
        return m.group(1)
    m = re.search(r'([A-D])', text)
    return m.group(1) if m else "UNKNOWN"


def evaluate_mcq_question(image_path, prompt, description):
    """Send an image and question to the multimodal model; failures return an [Error] string."""
    if not os.path.exists(image_path):
        return "[Error] Image Not Found"

    mime_type = mimetypes.guess_type(image_path)[0] or "image/png"
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("ascii")
    image_url = f"data:{mime_type};base64,{image_data}"

    rag_context, _ = retrieve_vsu_context(prompt, [], description)
    rag_messages = [vsu_rag_system_message(rag_context)] if rag_context else []

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=rag_messages + [{
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_url}},
                        {"type": "text", "text": prompt},
                    ]
                }],
                temperature=0.01, max_tokens=512)
            content = response.choices[0].message.content
            if isinstance(content, list):
                raw_text = "".join(
                    item.get("text", "") if isinstance(item, dict) else getattr(item, "text", "")
                    for item in content)
            else:
                raw_text = str(content or "")
            return raw_text.strip()
        except Exception as e:
            err_chain = []
            cur = e
            while cur is not None:
                err_chain.append(f"{type(cur).__name__}: {cur}")
                cur = cur.__cause__ or cur.__context__
            print(f"[API Retry {attempt + 1}/3] " + " <- ".join(err_chain))
            time.sleep(2)
    return "[Error] API Call Failed after 3 attempts"


def build_prompt_base(question, options):
    prompt = f"{question}\n\nOptions:\n"
    for i, opt in enumerate(options):
        prompt += f"{chr(65 + i)}. {opt}\n"
    return prompt


NOTATION_PROMPT_TAIL = """
[CRITICAL REQUIREMENT]
You are an expert Optical Music Recognition (OMR) AI.
Carefully analyze the provided sheet music image and answer the multiple-choice question.

[CRITICAL: Notation Format Rules]
The musical notes in the options are represented using a specific symbolic text format. You MUST decode them using the following rules to match the visual image:

1. Durations (Note Values):
   - 'w' = whole note
   - 'h' = half note
   - 'q' = quarter note
   - 'e' = eighth note
   - 's' = sixteenth note
   - 't' = thirty-second note

2. Modifiers:
   - '.' = dotted note (e.g., 'e.' means dotted eighth note)
   - '~' = tie / slur (e.g., 'q~' means the note is tied to the next note)

3. Pitches:
   - Uses Standard Scientific Pitch Notation (e.g., C4 is middle C).
   - Accidentals: '#' for sharp, 'b' for flat (e.g., C#4, Bb3).

4. Combined Format (Duration + Pitch):
   - "sC#4" -> Sixteenth note, C sharp 4.
   - "qA2~" -> Quarter note, A 2, tied.
   - "e.A2" -> Dotted eighth note, A 2.

5. Important Terminology Mapping in Questions:
   - If asked for "pitch and duration", expect the combined format (e.g., "sC#4 eA2").
   - If asked for "pitch", expect ONLY the pitch class and octave (e.g., "C#4 A2").
   - If asked for "tempo", it actually refers to the rhythm/durations ONLY (e.g., "s e s q~").
"""
NOTATION_PROMPT_END = "Read the specific bar and clef mentioned in the question. Compare your visual analysis with the options. The uppercase answer letter MUST be the very first character of your response. Please output ONLY the single letter of the correct option (A, B, C, or D). Do not explain or add any other text."


GUITAR_PROMPT_TAIL = """
You are an expert Optical Music Recognition (OMR) AI specializing in Guitar Tablature.
Carefully analyze the provided guitar tab image and answer the multiple-choice question.
The uppercase answer letter MUST be the very first character of your response.
Please output ONLY the single letter of the correct option (A, B, C, or D). Do not explain or add any other text.

[CRITICAL: Guitar Tablature Format Rules]
The options use a specific 1-dimensional text format to represent the 2D visual tablature. You MUST decode them using the following rules:

1. Fingering Notation (String and Fret):
   - Format: S<String_Number>_F<Fret_Number>
   - "S" represents the String.
     * IMPORTANT VISUAL MAPPING: In the image, the TOP line is the 1st string (S1, high E), and the BOTTOM line is the 6th string (S6, low E).
   - "F" represents the Fret number written on the line. F0 means an open string (a '0' written on the line).
   - Example: "S6_F3" means a '3' is written on the bottom line (6th string).

2. Chords (Simultaneous Notes):
   - Notes aligned vertically in the same column are played together as a chord.
   - In the text options, they are joined by a plus sign '+'.
   - Example: "S6_F0+S5_F2+S4_F2" means the numbers 0, 2, and 2 are written vertically on the 6th, 5th, and 4th strings respectively.

3. Scientific Pitch Notation:
   - If the question asks for "scientific pitches", the options will be absolute pitches (e.g., E2, G3) instead of string/fret combinations.
   - Standard tuning applies: S6=E2, S5=A2, S4=D3, S3=G3, S2=B3, S1=E4.

Read the specific segment or bar mentioned in the question. Compare your visual analysis with the options.
"""


JIANPU_PROMPT_TAIL = """
[CRITICAL REQUIREMENT]
You are taking a Jianpu (Numbered Musical Notation) exam.
1. Format each note: [Pitch Modifier][Note Number]([Duration Fraction])
   - Note Number: 1-7 for pitch, 0 for rest.
   - Pitch Modifier: _ for one octave lower (_5), __ for two octaves lower (__2), ^ for one octave higher (^1). No prefix for middle register.
   - Duration Fraction: Fractions in parentheses, e.g., (1/4), (1/8), (1/16), (3/16), (1/2).
2. Measures: Use '|' to separate measures.
3. Separation: Use a single space to separate notes.
Please answer by outputting ONLY the single letter of the correct option (A, B, C, or D). Do not explain or add any other text.
"""


def construct_mcq_prompt(question, options, task):
    prompt = build_prompt_base(question, options)
    if task == "notation":
        prompt += NOTATION_PROMPT_TAIL + NOTATION_PROMPT_END
    elif task == "guitar":
        prompt += GUITAR_PROMPT_TAIL
    elif task == "jianpu":
        prompt += JIANPU_PROMPT_TAIL
    return prompt


TASK_CONFIG = {
    "notation": {
        "qa_json": os.path.join(DATA_ROOT, "data", "extracted_questions_per_id.json"),
        "image_dir": os.path.join(DATA_ROOT, "data", "images 0-100"),
        "output_excel": os.path.join(DATA_ROOT, "data", "pitch_eval_results.xlsx"),
        "rag_desc": "standard staff notation",
        "image_suffix": "",  # {doc_id}.png
    },
    "guitar": {
        "qa_json": os.path.join(DATA_ROOT, "data", "guitar_notation", "guitar_qa_82_mcq.json"),
        "image_dir": os.path.join(DATA_ROOT, "data", "guitar_notation", "images"),
        "output_excel": os.path.join(DATA_ROOT, "data", "guitar_notation", "vqa_mcq_evaluation_results.xlsx"),
        "rag_desc": "six-string guitar tablature",
        "image_suffix": "",
    },
    "jianpu": {
        "qa_json": os.path.join(DATA_ROOT, "data", "jianpu_qa_100.json"),
        "image_dir": os.path.join(DATA_ROOT, "data", "simple_notation_images 0-100"),
        "output_excel": os.path.join(DATA_ROOT, "data", "jianpu_eval_results.xlsx"),
        "rag_desc": "Jianpu numbered musical notation",
        "image_suffix": "_1",  # {doc_id}_1.png
    },
}


def run_task(task, output_excel=None):
    cfg = TASK_CONFIG[task]
    qa_json = cfg["qa_json"]
    image_dir = cfg["image_dir"]
    output_excel = output_excel or cfg["output_excel"]

    if not os.path.exists(qa_json):
        print(f"[Error] Question-bank JSON file not found: {qa_json}")
        return

    with open(qa_json, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)

    label = {"notation": "staff notation", "guitar": "guitar tablature", "jianpu": "Jianpu"}[task]
    print(f"Loaded {len(qa_data)} {label} multiple-choice questions.")

    results = []
    correct_count = 0
    valid_count = 0

    for item in tqdm(qa_data, desc=f"Evaluating {label} questions"):
        doc_id = item.get("doc_id", "")
        question = item.get("question", "")
        options = item.get("options", [])
        ground_truth = item.get("answer", "").upper()

        img_path_png = os.path.join(image_dir, f"{doc_id}{cfg['image_suffix']}.png")
        img_path_jpg = os.path.join(image_dir, f"{doc_id}{cfg['image_suffix']}.jpg")
        image_path = img_path_png if os.path.exists(img_path_png) else img_path_jpg

        if not os.path.exists(image_path):
            tqdm.write(f"[Warning] Image for {label} item {doc_id} not found; skipping.")
            continue

        prompt = construct_mcq_prompt(question, options, task)
        raw_response = evaluate_mcq_question(image_path, prompt, cfg["rag_desc"])

        if "[Error]" not in raw_response:
            predicted_letter = extract_answer_letter(raw_response)
            is_correct = (predicted_letter == ground_truth)
            if is_correct:
                correct_count += 1
                tqdm.write(f"{doc_id} | Prediction: {predicted_letter} | Reference: {ground_truth}")
            else:
                tqdm.write(f"{doc_id} | Prediction: {predicted_letter} | Reference: {ground_truth} (Raw: {raw_response[:20]})")
            valid_count += 1
            results.append({
                "Doc ID": doc_id, "Question": question, "Ground Truth": ground_truth,
                "AI Prediction": predicted_letter, "Is Correct": is_correct,
                "AI Raw Output": raw_response,
            })
        else:
            tqdm.write(f"[Warning] API call failed for {doc_id}: {raw_response}")

        pd.DataFrame(results).to_excel(output_excel, index=False)
        time.sleep(1)

    if valid_count > 0:
        accuracy = (correct_count / valid_count) * 100
        print("\n" + "=" * 50)
        print(f"{label} evaluation complete.")
        print(f"Valid responses: {valid_count}")
        print(f"Correct responses: {correct_count}")
        print(f"Exact accuracy: {accuracy:.2f}%")
        print(f"Detailed results saved to: {output_excel}")
        print("=" * 50)



def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visual score understanding evaluation")
    parser.add_argument("--task", choices=list(TASK_CONFIG), default="notation",
                        help="Task: notation / guitar / jianpu")
    parser.add_argument("--output", default=None, help="Override the output Excel path")
    args = parser.parse_args()
    run_task(args.task, output_excel=args.output)


if __name__ == "__main__":
    main()
