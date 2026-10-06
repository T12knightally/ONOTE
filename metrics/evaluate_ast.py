import os
import sys
import json
import time
import tempfile
import difflib
import re
import base64
import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment
from pathlib import Path
from openai import OpenAI

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

BASE_URL = os.environ.get("APIYI_BASE_URL", "https://api.apiyi.com/v1")
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

AUDIO_MODEL = "gemini-3.1-flash-lite-preview"
CHUNK_LENGTH_MS = 10000  # 切片时长（毫秒）

_RAG_CACHE = {}


def get_rag_context(domain: str, task: str):
    """拉取并缓存指定子任务的 RAG 上下文。"""
    key = (domain, task)
    if key not in _RAG_CACHE:
        _RAG_CACHE[key], _ = retrieve_task_context(domain, task)
    return _RAG_CACHE[key]

FLAT_TO_SHARP = {
    'Bb': 'A#', 'Eb': 'D#', 'Ab': 'G#', 'Db': 'C#',
    'Gb': 'F#', 'Cb': 'B', 'Fb': 'E',
}


def normalize_enharmonic(text):
    for flat, sharp in FLAT_TO_SHARP.items():
        text = text.replace(flat, sharp)
    return text


def calculate_metric(gt_list, trans_list):
    """基于 difflib.SequenceMatcher（Ratcliff/Obershelp）的序列比对打分。"""
    if not trans_list:
        return 0.0, 0
    matcher = difflib.SequenceMatcher(None, gt_list, trans_list)
    matched = sum(triple.size for triple in matcher.get_matching_blocks())
    return (matched / len(trans_list)) * 100, matched


def transcribe_audio(audio_path, prompt, rag_tag, sanitize_fn=None,
                     model=AUDIO_MODEL, max_tokens=None, temperature=0.01, top_p=None):
    """把前 10 秒音频切片编码后发给模型，返回（清洗后的）文本。失败返回含 [Error] 的字符串。"""
    if not os.path.exists(audio_path):
        return "[Error] File not found"
    try:
        audio = AudioSegment.from_file(audio_path)[:CHUNK_LENGTH_MS]
    except Exception as e:
        return f"[Error] Audio read failed: {e}"

    with tempfile.TemporaryDirectory() as tmp:
        chunk_path = os.path.join(tmp, "chunk.wav")
        audio.export(chunk_path, format="wav")
        with open(chunk_path, "rb") as f:
            base64_audio = base64.b64encode(f.read()).decode("utf-8")

        rag_context = get_rag_context("ast", rag_tag)
        rag_messages = [rag_system_message(rag_context, "ast")] if rag_context else []

        kwargs = dict(
            model=model,
            messages=rag_messages + [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"data": base64_audio, "format": "wav"}},
                    ],
                }
            ],
            temperature=temperature,
        )
        if top_p is not None:
            kwargs["top_p"] = top_p
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        try:
            response = client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content
            return sanitize_fn(raw) if sanitize_fn else (raw or "")
        except Exception as e:
            tqdm.write(f"      [Error] API Failure: {e}")
            return "[Error] API Failure"


ABC_PROMPT = """You are an extremely precise AI music transcription expert.
Please transcribe the first 10 seconds of this audio into a space-separated sequence of notes.

【Mandatory Requirements】
1. Format: <Pitch><Octave>(<Duration>) for example:C4(1/4)
2. Transcribe the sequence exactly as you hear it.
3. ANTI-LOOP RULE: Do not repeat the same note indefinitely. If you are unsure, stop transcribing.
4. QUANTITY LIMIT: Transcribe no more than 60 notes for these 10 seconds.
5. Only output the notes. No conversation."""


def abc_parse_token(token):
    match = re.match(r"([A-Ga-g][#b]?\d)\(([^)]+)\)", token)
    if match:
        return match.group(1), match.group(2)
    return token, ""


def abc_extract_staff_from_json(gt_data):
    flat_notes = []
    if not gt_data or "bars" not in gt_data:
        return []
    for bar in gt_data["bars"]:
        staves = bar.get("staves", {})
        for stave_name in ["treble", "bass"]:
            for n in staves.get(stave_name, []):
                p, d = n.get("pitch"), n.get("duration")
                if p and d:
                    flat_notes.append(normalize_enharmonic(f"{p}({d})"))
    return flat_notes


def abc_sanitize(raw_text):
    if not raw_text:
        return ""
    tokens = raw_text.strip().split()
    if len(tokens) > 80:
        tqdm.write("      [警告] 监测到疑似幻觉/死循环，执行物理截断")
        tokens = tokens[:60]
    for i in range(len(tokens) - 5):
        if len(set(tokens[i:i + 6])) == 1:
            tqdm.write(f"      [警告] 监测到音符复读，从第 {i} 个音符处切断")
            tokens = tokens[:i + 1]
            break
    return " ".join(normalize_enharmonic(t) for t in tokens)


def run_abc(output_excel=None):
    AUDIO_DIR = os.path.join(DATA_ROOT, "data", "Audio 0-100")
    METADATA_PATH = os.path.join(DATA_ROOT, "data", "metadata 0-100.json")
    OUTPUT_EXCEL = output_excel or os.path.join(DATA_ROOT, "data", "staff_trans_analyzes.xlsx")

    if not os.path.exists(METADATA_PATH):
        print("❌ 找不到 metadata JSON 文件")
        return

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(('.mp3', '.wav'))]
    results = []

    for filename in tqdm(audio_files, desc="📊 音高/长度多维评测"):
        file_key = os.path.splitext(filename)[0]
        gt_entry = metadata.get(file_key)
        if not gt_entry:
            continue

        gt_full = abc_extract_staff_from_json(gt_entry)[:100]
        gt_pitches = [abc_parse_token(t)[0] for t in gt_full]
        gt_durations = [abc_parse_token(t)[1] for t in gt_full]

        raw_output = transcribe_audio(
            os.path.join(AUDIO_DIR, filename), ABC_PROMPT, "staff",
            sanitize_fn=abc_sanitize, top_p=0.1, max_tokens=4096)

        if "[Error]" not in raw_output:
            trans_full = raw_output.strip().split()
            trans_pitches = [abc_parse_token(t)[0] for t in trans_full]
            trans_durations = [abc_parse_token(t)[1] for t in trans_full]

            acc_full, _ = calculate_metric(gt_full, trans_full)
            acc_pitch, _ = calculate_metric(gt_pitches, trans_pitches)
            acc_dur, _ = calculate_metric(gt_durations, trans_durations)

            tqdm.write(f"\n🎵 文件: {filename} | 最终处理音符数: {len(trans_full)}")
            tqdm.write(f"   🎯 全要素: {acc_full:.1f}% | 🎹 纯音高: {acc_pitch:.1f}% | ⏳ 纯长度: {acc_dur:.1f}%")

            results.append({
                "File": filename, "Trans_Count": len(trans_full),
                "Full_Acc(%)": round(acc_full, 2), "Pitch_Acc(%)": round(acc_pitch, 2),
                "Duration_Acc(%)": round(acc_dur, 2), "Raw_Output": raw_output
            })
        else:
            results.append({"File": filename, "Full_Acc(%)": 0, "Raw_Output": "API Error"})

        pd.DataFrame(results).to_excel(OUTPUT_EXCEL, index=False)
        time.sleep(1)


JIAN_PROMPT = """You are a precise music transcription expert.
Transcribe the first 10 seconds of this audio into [Structured Jianpu Syntax].

【Mandatory Syntax Rules】
1. Format each note: [Pitch Modifier][Note Number]([Duration Fraction])
   - Note Number: 1-7 for pitch, 0 for rest.
   - Pitch Modifier: _ for one octave lower (_5), __ for two octaves lower (__2), ^ for one octave higher (^1). No prefix for middle register.
   - Duration Fraction: Fractions in parentheses, e.g., (1/4), (1/8), (1/16), (3/16), (1/2).
2. Measures: Use '|' to separate measures.
3. Separation: Use a single space to separate notes.

Example: 3(1/4) _5(1/4) 0(1/8) 3(1/16) 4(5/16) | _6(3/16) __2(1/16) 1(1/4) _6(1/2) |

Only output the structured notation. No conversational text."""


def jian_parse_components(text, is_gt=False):
    if is_gt:
        text = str(text).replace('<<', '^^').replace('>>', '').replace('<', '^').replace('>', '')
    raw_tokens = re.findall(r'[\^_]{0,2}[0-7]\([^)]*\)|\|', str(text))
    full, pitch, dur = [], [], []
    for token in raw_tokens:
        if token == '|':
            full.append('|'); pitch.append('|'); dur.append('|')
        else:
            m = re.match(r"([\^_]{0,2})([0-7])\(([^)]+)\)", token)
            if m:
                full.append(token)
                pitch.append(m.group(2))
                dur.append(m.group(3))
    return full, pitch, dur


def jian_sanitize(raw_text):
    tokens = raw_text.strip().split()
    if len(tokens) > 100:
        tokens = tokens[:80]
    for i in range(len(tokens) - 7):
        if len(set(tokens[i:i + 8])) == 1:
            return " ".join(tokens[:i + 1])
    return raw_text


def run_jian(output_excel=None):
    AUDIO_DIR = os.path.join(DATA_ROOT, "data", "Audio 0-100")
    METADATA_PATH = os.path.join(DATA_ROOT, "data", "converted_simple_notation_metadata 0-100.json")
    OUTPUT_EXCEL = output_excel or os.path.join(DATA_ROOT, "data", "jianpu_trans_analyses.xlsx")

    if not os.path.exists(METADATA_PATH):
        print("❌ 找不到 JSON！")
        return

    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(('.mp3', '.wav'))]
    results = []

    for filename in tqdm(audio_files, desc="📊 多维数据评测"):
        file_key = os.path.splitext(filename)[0]
        if file_key not in metadata:
            continue

        raw_output = transcribe_audio(
            os.path.join(AUDIO_DIR, filename), JIAN_PROMPT, "jianpu",
            sanitize_fn=jian_sanitize, max_tokens=4096)

        if "[Error]" not in raw_output:
            gt_text = " ".join(str(v) for v in metadata[file_key].values())
            gt_full, gt_pitch, gt_dur = jian_parse_components(gt_text, is_gt=True)
            gt_full, gt_pitch, gt_dur = gt_full[:100], gt_pitch[:100], gt_dur[:100]
            trans_full, trans_pitch, trans_dur = jian_parse_components(raw_output, is_gt=False)

            acc_full, _ = calculate_metric(gt_full, trans_full)
            acc_pitch, _ = calculate_metric(gt_pitch, trans_pitch)
            acc_dur, _ = calculate_metric(gt_dur, trans_dur)

            tqdm.write(f"🎯 成绩 -> 全要素: {acc_full:.1f}% | 🎹 音名: {acc_pitch:.1f}% | ⏳ 时值: {acc_dur:.1f}%")

            results.append({
                "File": filename, "Note_Count": len(trans_full),
                "Full_Accuracy(%)": round(acc_full, 2),
                "Pitch_Accuracy(%)": round(acc_pitch, 2),
                "Duration_Accuracy(%)": round(acc_dur, 2),
                "AI_Raw": raw_output
            })
        else:
            results.append({"File": filename, "Full_Accuracy(%)": 0, "AI_Raw": raw_output})

        pd.DataFrame(results).to_excel(OUTPUT_EXCEL, index=False)

    print(f"\n🎉 评测完成！结果已保存至：{OUTPUT_EXCEL}")


GUITAR_TRANSCRIBER_PROMPT = """You are an expert AI guitar transcription model. Listen to the first 10 seconds of this guitar audio.
You must transcribe the notes into a 1-dimensional "Linear Tablature" sequence.

【Linear Tab Format Rules】
1. Format: S<String>_F<Fret>. Strings: 1 (high e) to 6 (low E). Frets: 0-24. Example: S6_F3
2. Chords: Join with '+', e.g., S6_F0+S5_F2
3. Output a space-separated sequence.

【ANTI-HALLUCINATION RULES】
- DO NOT generate sequential mathematical patterns (e.g., S1_F5 S2_F5 S3_F5 S4_F5).
- DO NOT iterate through frets logically. ONLY transcribe the actual acoustic sounds you hear.
- Stop immediately when the 10-second audio ends. Maximum 50 notes.

Only output the sequence. No text."""

GUITAR_EVALUATOR_PROMPT = """You are a precise data extractor.
Read the provided Ground Truth JAMS/JSON data and extract the absolute pitches of the FIRST 60 notes.
Convert all pitches to Scientific Pitch format (e.g., E2, B3, G4).
DO NOT format as a JSON object. Output ONLY a flat, space-separated string of pitches.

Example output:
E2 B2 E3 G3 D4"""

BASE_MIDI = {'S1': 64, 'S2': 59, 'S3': 55, 'S4': 50, 'S5': 45, 'S6': 40}
PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def guitar_translate_tab_to_pitch(tab_sequence):
    translated = []
    for token in tab_sequence.strip().split():
        for part in token.split('+'):
            m = re.match(r'(S[1-6])_F(\d+)', part)
            if m:
                string, fret = m.groups()
                midi = BASE_MIDI[string] + int(fret)
                octave = (midi // 12) - 1
                note_name = PITCH_NAMES[midi % 12]
                translated.append(f"{note_name}{octave}")
    return " ".join(translated)


def guitar_sanitize(raw_text):
    tokens = raw_text.strip().split()
    if len(tokens) > 60:
        tqdm.write("      [系统拦截] 监测到极长幻觉序列，已执行物理切断！")
        tokens = tokens[:60]
    return " ".join(tokens)


def guitar_extract_jams_ground_truth(jams_data):
    json_str = json.dumps(jams_data, ensure_ascii=False)[:10000]
    try:
        response = client.chat.completions.create(
            model=AUDIO_MODEL,
            messages=[
                {"role": "system", "content": GUITAR_EVALUATOR_PROMPT},
                {"role": "user", "content": f"JAMS JSON Data:\n{json_str}"}
            ],
            temperature=0.01, max_tokens=1024)
        return response.choices[0].message.content.strip()
    except Exception as e:
        tqdm.write(f"      [Error] GT extraction failed: {e}")
        return ""


def run_guitar(output_excel=None):
    AUDIO_DIR = os.path.join(DATA_ROOT, "data", "guitar_notation", "audio")
    ANNOTATION_DIR = os.path.join(DATA_ROOT, "data", "guitar_notation", "annotation")
    OUTPUT_EXCEL = output_excel or os.path.join(DATA_ROOT, "data", "guitar_trans_analyzes.xlsx")

    if not os.path.exists(AUDIO_DIR) or not os.path.exists(ANNOTATION_DIR):
        print("❌ 路径错误")
        return

    audio_files = [f for f in os.listdir(AUDIO_DIR) if f.lower().endswith(('.mp3', '.wav'))]
    results = []

    for audio_filename in tqdm(audio_files, desc="🎸 吉他多维评测进度"):
        base_name = os.path.splitext(audio_filename)[0].replace("_mic", "")
        jams_path = os.path.join(ANNOTATION_DIR, f"{base_name}.jams")
        if not os.path.exists(jams_path):
            continue

        with open(jams_path, 'r', encoding='utf-8') as f:
            jams_data = json.load(f)

        tqdm.write(f"\n🎧 正在处理: {audio_filename}")
        raw_transcription = transcribe_audio(
            os.path.join(AUDIO_DIR, audio_filename), GUITAR_TRANSCRIBER_PROMPT, "guitar",
            sanitize_fn=guitar_sanitize, max_tokens=4096)

        if "[Error]" not in raw_transcription:
            translated_pitches = guitar_translate_tab_to_pitch(raw_transcription)
            gt_pitches = guitar_extract_jams_ground_truth(jams_data)

            trans_seq = translated_pitches.split()
            gt_seq = gt_pitches.split()

            tqdm.write(f"      [Step 2 终极比对] 真值: {gt_seq[:8]}...")
            tqdm.write(f"      [Step 2 终极比对] 听写: {trans_seq[:8]}...")

            if trans_seq and gt_seq:
                matcher = difflib.SequenceMatcher(None, gt_seq, trans_seq)
                matched = sum(t.size for t in matcher.get_matching_blocks())
                match_ratio = min((matched / len(trans_seq)) * 100, 100.0)
            else:
                match_ratio, matched = 0.0, 0

            results.append({
                "File": audio_filename,
                "Linear Tab (Raw)": raw_transcription,
                "Translated Pitches": translated_pitches,
                "Accuracy(%)": round(match_ratio, 2),
                "Matched": f"{matched}/{len(trans_seq)}"
            })
        else:
            results.append({"File": audio_filename, "Accuracy(%)": 0})

        pd.DataFrame(results).to_excel(OUTPUT_EXCEL, index=False)
        time.sleep(1)


TASKS = {"abc": run_abc, "jian": run_jian, "guitar": run_guitar}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="AST 音频转记谱评测")
    parser.add_argument("--task", choices=list(TASKS), default="abc",
                        help="子任务：abc / jian / guitar")
    parser.add_argument("--output", default=None, help="覆盖输出 Excel 路径")
    args = parser.parse_args()
    TASKS[args.task](output_excel=args.output)


if __name__ == "__main__":
    main()
