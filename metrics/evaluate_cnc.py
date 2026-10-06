import os
import sys
import json
import re
import base64
import time
from pathlib import Path
from openai import OpenAI, RateLimitError
from difflib import SequenceMatcher

_KC_ROOT_ENV = os.environ.get("KNOWLEDGECLAW_ROOT")
KNOWLEDGECLAW_ROOT = Path(_KC_ROOT_ENV) if _KC_ROOT_ENV else Path(__file__).resolve().parents[2]
if str(KNOWLEDGECLAW_ROOT) not in sys.path:
    sys.path.insert(0, str(KNOWLEDGECLAW_ROOT))

DATA_ROOT = os.environ.get("MUSICBENCH_ROOT", r"D:\MusicBench")

from integration.musicbench_rag_adapter import rag_system_message, retrieve_task_context
from integration.cnc.json_response_parser import parse_json_object

API_KEY = os.environ.get("APIYI_API_KEY", "")
if not API_KEY and os.path.exists("api_key_apiyi.txt"):
    API_KEY = open("api_key_apiyi.txt", encoding="utf-8").read().strip()
if not API_KEY:
    raise RuntimeError("APIYI_API_KEY is not set. Export it or create api_key_apiyi.txt.")

API_URL = "https://api.apiyi.com/v1"
MODEL_NAME = 'gemini-2.5-pro'
client = OpenAI(api_key=API_KEY, base_url=API_URL)

_RAG_CACHE = {}


def get_rag_context(task):
    if task not in _RAG_CACHE:
        _RAG_CACHE[task], _ = retrieve_task_context("cnc", task)
    return _RAG_CACHE[task]


def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')


def get_model_prediction(image_path, system_prompt, max_retries=3):
    """发送图片给模型，返回解析后的 JSON 字典；失败返回 None。"""
    base64_image = encode_image(image_path)
    rag_context = get_rag_context(_CURRENT_TASK)
    rag_messages = [rag_system_message(rag_context, "cnc")] if rag_context else []

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=rag_messages + [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract the notation from this image."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                    ]}
                ],
                response_format={"type": "json_object"},
                max_tokens=2048,
                temperature=0.01,
            )
            raw_content = (response.choices[0].message.content or "").strip()
            return parse_json_object(raw_content)
        except json.JSONDecodeError as e:
            print(f"  [JSON Parse Error] Attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(2)
        except RateLimitError:
            wait = (attempt + 1) * 3
            print(f"  [Rate Limit 429] Waiting {wait} seconds...")
            time.sleep(wait)
        except Exception as e:
            print(f"  [Runtime Error] Attempt {attempt + 1}/{max_retries}: {e}")
            time.sleep(2)
    return None


_CURRENT_TASK = "guitar_to_staff"
GUITAR_SYSTEM_PROMPT = """You are a highly advanced Music OCR AI specializing in exhaustive pitch detection.
Your goal is to transcribe every single note visible in the sheet music with maximum recall.
Return ONLY a JSON object with the key "pitches"

### CORE OBJECTIVES:
1. **Exhaustive Detection**: Identify every note head. Do not skip notes on ledger lines or within chords.
2. **Horizontal Precision**: Scan strictly from left to right. For vertical chords, list pitches from lowest to highest.
3. **Accidental Awareness**: Correct identify sharps (#), flats (b), and naturals (n).
4. **Pitch Range**: Only recognize pitches within the standard guitar range, typically from **E2 to E6**.
### ANTI-HALLUCINATION & NO-EXAMPLE RULES:
- **STRICT PROHIBITION**: NEVER use pitches or sequences from the "EXAMPLE OUTPUT" below as your actual answer.
- **REAL-TIME EXTRACTION**: Your output must be based EXCLUSIVELY on the provided image.
- **UNCERTAINTY HANDLING**: If a note is blurry or unclear, use your best professional judgment based on the context of the staff, but NEVER invent data or copy from examples.

### OUTPUT FORMAT:
- Return ONLY a JSON object with the key "pitches".
- Use Scientific Pitch Notation (e.g., ["C4", "Eb5"]).
- NO durations, NO rhythm, NO measure bars, NO conversational text."""


def run_guitar_to_staff():
    IMAGE_DIR = os.path.join(DATA_ROOT, "data", "guitar_notation", "images")
    GROUND_TRUTH_PATH = os.path.join(DATA_ROOT, "data", "guitar_notation", "all_pitches_summary.json")

    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"找不到标准答案文件: {GROUND_TRUTH_PATH}")
        return
    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    all_scores = []

    for filename in files:
        json_key = os.path.splitext(filename)[0] + ".json"
        if json_key not in ground_truth:
            print(f"跳过 {filename}: 在 JSON 中找不到对应的 Key '{json_key}'")
            continue

        print(f"--- 正在处理: {filename} ---")
        prediction = get_model_prediction(os.path.join(IMAGE_DIR, filename), GUITAR_SYSTEM_PROMPT)

        if prediction and "pitches" in prediction:
            pred_list = [p.upper() for p in prediction["pitches"]]
            target_list = [t.upper() for t in ground_truth[json_key]]
            score = SequenceMatcher(None, pred_list, target_list).ratio()
            all_scores.append(score)
            print(f"  预测: {pred_list}")
            print(f"  标准: {target_list}")
            print(f"  准确率: {score:.2%}")
        else:
            print(f"  {filename} 识别失败")

    if all_scores:
        print(f"\n评估完成！总样本数: {len(all_scores)}")
        print(f"平均音高正确率: {sum(all_scores) / len(all_scores):.2%}")


_CURRENT_TASK = "jianpu_to_staff"
JIANPU_TO_STAFF_SYSTEM_PROMPT = """You are a highly precise Music OCR AI. Your task is to transcribe sheet music images directly into Scientific Pitch Notation with Durations.

### CORE OBJECTIVES:
1. **Exhaustive Detection**: Identify every single note head. Do not skip notes on ledger lines or within chords.
2. **Horizontal Precision**: Scan strictly from left to right. For vertical chords, list pitches from the LOWEST (bottom) to the HIGHEST (top).
3. **Guitar Range**: Only recognize pitches within the standard guitar range (E2 to E6).

### OUTPUT FORMAT:
- Format: `Pitch(Duration)` (e.g., "G#3(1/16)", "C4(1/4)").
- Return ONLY a JSON object with the key "notes".
- Pitches: Scientific Pitch Notation (C4, Eb5).
- Durations: Fractions in parentheses (1/4, 1/8, 3/16).

### EXAMPLE OUTPUT (FOR FORMATTING ONLY):
{
  "notes": ["E2(1/4)", "G#3(1/8)", "B3(1/16)"]
}
"""


def calculate_detailed_accuracy(pred_list, target_list):
    if not target_list or not pred_list:
        return 0.0, 0.0, 0.0
    pred_pitches = [re.sub(r'\(.*\)', '', n) for n in pred_list]
    target_pitches = [re.sub(r'\(.*\)', '', n) for n in target_list]

    def get_durations(lst):
        return [re.search(r'\(.*\)', n).group() if re.search(r'\(.*\)', n) else "" for n in lst]

    pred_rhythms = get_durations(pred_list)
    target_rhythms = get_durations(target_list)
    p_score = SequenceMatcher(None, pred_pitches, target_pitches).ratio()
    r_score = SequenceMatcher(None, pred_rhythms, target_rhythms).ratio()
    return p_score, r_score, (p_score + r_score) / 2


def run_jianpu_to_staff():
    IMAGE_DIR = os.path.join(DATA_ROOT, "data", "simple_notation_images_rgb")
    GROUND_TRUTH_PATH = os.path.join(DATA_ROOT, "data", "pitch_duration_summarywuxian.json")

    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"错误: 找不到文件 {GROUND_TRUTH_PATH}")
        return
    with open(GROUND_TRUTH_PATH, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    history = {"pitch": [], "rhythm": [], "average": []}
    images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg'))]
    print(f"开始评估，共计 {len(images)} 个文件...")

    for filename in images:
        file_id = os.path.splitext(filename)[0]
        base_id = file_id.rsplit('_', 1)[0] if '_' in file_id else file_id
        candidates = [file_id, base_id, f"{file_id}.json", f"{base_id}.json"]
        json_key = next((c for c in candidates if c in ground_truth), None)
        if json_key is None:
            print(f"跳过 {filename}: 在标准答案中未找到对应 Key (尝试: {candidates})")
            continue

        print(f"\n" + "-" * 50)
        print(f"正在处理: {filename}")
        pred_notes = get_model_prediction(os.path.join(IMAGE_DIR, filename), JIANPU_TO_STAFF_SYSTEM_PROMPT)
        if pred_notes:
            target_notes = ground_truth[json_key]
            p_acc, r_acc, avg_acc = calculate_detailed_accuracy(pred_notes, target_notes)
            history["pitch"].append(p_acc)
            history["rhythm"].append(r_acc)
            history["average"].append(avg_acc)
            print(f"  [预测示例]: {pred_notes[:8]}")
            print(f"  [标准示例]: {target_notes[:8]}")
            print(f"  [准确率详情] 音高: {p_acc:.2%} | 节奏: {r_acc:.2%} | 平均: {avg_acc:.2%}")
        else:
            print("  [失败] 无法获取模型响应")

    if history["average"]:
        num = len(history["average"])
        print("\n" + "=" * 60)
        print(f"评估任务完成！处理样本数: {num}")
        print(f"最终平均音高正确率: {sum(history['pitch']) / num:.2%}")
        print(f"最终平均节奏正确率: {sum(history['rhythm']) / num:.2%}")
        print(f"最终平均总正确率:   {sum(history['average']) / num:.2%}")
        print("=" * 60)


_CURRENT_TASK = "staff_to_jianpu"
STAFF_TO_JIANPU_SYSTEM_PROMPT = """You are a highly precise Music OCR and Transcription AI.
Convert the provided sheet music image into Numbered Musical Notation (Jianpu) JSON format.

### SYMBOL DEFINITIONS:
- **PITCH**:
  - Central Octave: `1, 2, 3, 4, 5, 6, 7`
  - High Octave: `<n>`, Double High: `<<n>>`
  - Low Octave: `_n`, Double Low: `__n`
- **DURATION**: Written in parentheses after pitch, e.g., `(1/4)`, `(3/16)`, `(1/8)`.
- **MEASURE**: Use ` | ` to separate measures.
- **STRUCTURE**:
  - Use `"treble"` key for Treble Clef staff.
  - Use `"bass"` key for Bass Clef staff.
  - If the image has both, provide both keys in one JSON object.

### STRICT OUTPUT RULES:
1. DO NOT include any conversational text or explanations.
2. ONLY return a single valid JSON object.
3. Ensure every note follows the `Pitch(Duration)` format.
4. **DO NOT COPY THE NOTES FROM THE EXAMPLES.**
### EXAMPLES:
Example 1 (Single Bass Staff):
{
  "bass": "_5(1/16) _3(3/16) _3(1/4) | _5(1/8) __7(1/16) _5(1/16)"
}

Example 2 (Double Staff):
{
  "treble": "<5>(3/16) <<3>>(1/16) <3>(1/4) | <5>(1/4) <3>(3/16)",
  "bass": "_3(1/16) _5(3/16) _5(1/8) | _3(3/16) __5(1/16)"
}
"""


def calculate_accuracy(pred_str, target_str):
    if not pred_str:
        return 0.0, 0.0
    p_str = str(pred_str).replace('（', '(').replace('）', ')')
    t_str = str(target_str).replace('（', '(').replace('）', ')')
    pattern = r'([_<>]*\d)\s*\(\s*(\d+/?\d*)\s*\)'
    pred_notes = re.findall(pattern, p_str)
    target_notes = re.findall(pattern, t_str)
    if not target_notes:
        return 0.0, 0.0
    p_pred, r_pred = [n[0] for n in pred_notes], [n[1] for n in pred_notes]
    p_tar, r_tar = [n[0] for n in target_notes], [n[1] for n in target_notes]
    ratio = lambda a, b: SequenceMatcher(None, a, b).ratio()
    return ratio(p_pred, p_tar), ratio(r_pred, r_tar)


def run_staff_to_jianpu():
    IMAGE_DIR = os.path.join(DATA_ROOT, "data", "images 0-100")
    JSON_PATH = os.path.join(DATA_ROOT, "data", "converted_simple_notation_metadata 0-100.json")

    if not os.path.exists(JSON_PATH):
        print("Ground truth file missing.")
        return
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    all_scores = []
    for i in range(101):
        file_id = str(i).zfill(7)
        img_path = os.path.join(IMAGE_DIR, f"{file_id}.png")
        if not os.path.exists(img_path) or file_id not in ground_truth:
            continue

        print(f"--- Processing {file_id} ---")
        prediction = get_model_prediction(img_path, STAFF_TO_JIANPU_SYSTEM_PROMPT)
        if prediction:
            print(f"  [Model Output]: {json.dumps(prediction, ensure_ascii=False)}")
            target = ground_truth[file_id]
            item_scores = []
            for clef in ['treble', 'bass']:
                if clef in target:
                    pred_content = prediction.get(clef, "")
                    p_acc, r_acc = calculate_accuracy(pred_content, target[clef])
                    item_scores.append((p_acc * 0.5) + (r_acc * 0.5))
                    print(f"  [{clef}] Pitch: {p_acc:.2%}, Rhythm: {r_acc:.2%}")
            if item_scores:
                all_scores.append(sum(item_scores) / len(item_scores))
                print(f"  >> Average: {all_scores[-1]:.2%}")
        else:
            print(f"  Result: Failed.")

    if all_scores:
        print(f"\nFinal Benchmarking Result: {sum(all_scores) / len(all_scores):.2%}")


TASKS = {
    "guitar_to_staff": run_guitar_to_staff,
    "jianpu_to_staff": run_jianpu_to_staff,
    "staff_to_jianpu": run_staff_to_jianpu,
}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="CNC 记谱图片 OCR/互转评测")
    parser.add_argument("--task", choices=list(TASKS), default="guitar_to_staff",
                        help="子任务：guitar_to_staff / jianpu_to_staff / staff_to_jianpu")
    args = parser.parse_args()
    # 让 get_model_prediction 知道当前 task 以拉取对应 RAG 上下文
    global _CURRENT_TASK
    _CURRENT_TASK = args.task
    TASKS[args.task]()


if __name__ == "__main__":
    main()
