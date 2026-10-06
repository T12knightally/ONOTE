import os
import sys
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
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

API_URL = "https://api.apiyi.com/v1"
COMPOSER_MODEL = "gemini-2.5-pro"
CRITIC_MODEL = "gpt-5-mini"
TOTAL_SONGS = 80
client = OpenAI(api_key=API_KEY, base_url=API_URL)

_RAG_CACHE = {}


def get_rag_context(task):
    if task not in _RAG_CACHE:
        _RAG_CACHE[task], _ = retrieve_task_context("smg", task)
    return _RAG_CACHE[task]


def generate_score(composer_prompt, max_retries=3):
    rag_context = get_rag_context(_CURRENT_TASK)
    messages = [rag_system_message(rag_context, "smg")] if rag_context else []
    messages.append({"role": "user", "content": composer_prompt})
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=COMPOSER_MODEL, messages=messages,
                temperature=0.8, max_tokens=2048)
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "limit" in str(e).lower() or "429" in str(e):
                time.sleep((attempt + 1) * 3)
            else:
                return f"生成报错: {e}"
    return "生成超时失败"


def evaluate_score(abc_score, critic_prompt, max_tokens=4000, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=CRITIC_MODEL,
                messages=[
                    {"role": "system", "content": critic_prompt},
                    {"role": "user", "content": f"【待评审乐谱】\n{abc_score}"}
                ],
                temperature=0.1, max_tokens=max_tokens)
            return response.choices[0].message.content.strip()
        except Exception as e:
            if "limit" in str(e).lower() or "429" in str(e):
                time.sleep((attempt + 1) * 3)
            else:
                return f"评审报错: {e}"
    return "评审超时失败"


STAFF_COMPOSER_PROMPT = """
As a top-tier composer, please compose an original classical-style melody for me.

Key Signature: C Major
Time Signature: 4/4
Length: Strictly 8 measures.
Format: Please output ONLY standard ABC Notation code.

[Scoring Criteria]
Rhythmic Value Calculation: Since the time signature is 4/4, the sum of the rhythmic values for all notes and rests within a single measure [must absolutely equal 4.0 beats]. A quarter note equals 1 beat, and an eighth note equals 0.5 beats.
Musicality: The rhythm should be diverse, incorporating various rhythmic patterns such as dotted notes. Additionally, the melody should be beautiful, and the musical motif must be complete and well-developed.
"""

STAFF_CRITIC_PROMPT = """
As an extremely strict music theory professor at a music conservatory, your task is to review and score AI-generated text-based ABC notation provided by the user. The core objective is to determine the model's proficiency in generating numbered musical notation by analyzing the rhythmic values and musicality of the score. The evaluation requires assessing two separate modules, starting from fundamental principles, to ultimately form a comprehensive assessment with a continuous score ranging from 1 to 5. Please refer to the detailed scoring criteria below.


Scoring Criteria:
1. Rhythmic Value (Timing) Verification
A correct numbered musical notation score should contain exactly 4.0 beats per measure; that is, the sum of the rhythmic values of all notes between two | symbols must equal 4.0.

Scoring Examples:
Score 5: The sum of the rhythmic values is correct for all measures.
Score 3: The sum of the rhythmic values is incorrect for approximately half of the measures.
Score 1: The sum of the rhythmic values is incorrect for all measures.

2. Aesthetic Analysis
The motif development, melodic contour (judged by the rise and fall of the notation numbers), and rhythmic groove of the score should be rich and logical.

Scoring Examples:
Score 5: The piece features a complete motif, a beautiful melody, and a rich, logical rhythm.
Score 3: The piece features a discernible motif, a listenable melody, and a reasonable rhythm.
Score 1: The piece lacks any motif, the melody is unappealing, and it does not constitute actual music.

Final Scoring
Please only state the following at the end of your evaluation:
Technical Score: [Score]/5
Aesthetic Score: [Score]/5
Average Score: [Score]/5
"""

GUITAR_COMPOSER_PROMPT = """Act as a top-tier fingerstyle guitar master. Please compose a classical-style acoustic guitar melody for me.
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
(Notice how each measure is precisely 16 characters wide)
Beat|1 . . . 2 . . . 3 . . . 4 . . . |
e   |--------------------------------|
B   |--------1---------------1-------|
G   |----0-------0-------0-------0---|
D   |--2---------------2-------------|
A   |3---------------3---------------|
E   |--------------------------------|

[Scoring Criteria]
Strict Layout: The numbers and hyphens across the 6 strings must be absolutely vertically aligned. You must never add or omit any hyphens.
Fingering Feasibility: Please ensure that notes meant to be played simultaneously are physically possible for a human left hand to fret (do not include extreme, inhuman fret combinations with massive stretches, such as pressing the 2nd fret of the 6th string and the 15th fret of the 1st string at the same time).
Musicality:  Fingerstyle playing should have distinct, clear layers (Low bass root notes, middle-voice chords, and high treble melody).
"""

GUITAR_CRITIC_PROMPT = """As an extremely strict fingerstyle guitar master, your task is to review and score AI-generated ASCII guitar tablature (tabs) provided by the user. Your core objective is to evaluate the model's proficiency in generating guitar tabs by analyzing the layout and timing, fingering, and musicality of the sheet music. The evaluation requires assessing three separate modules, starting from fundamental principles, to ultimately form a comprehensive assessment with a continuous score ranging from 1 to 5. Please refer to the detailed scoring criteria below.

1. Layout Alignment and Timing Verification
Check Vertical Alignment: In a correctly formatted guitar tab, the 6 strings (from top to bottom) must be perfectly aligned vertically.

Calculate Character Count per Measure: Assuming a 4/4 time signature, the number of characters (dashes plus numbers) on each string between two bar lines (|) must be completely identical, and each measure should contain a length of exactly 16 characters.

Scoring Examples:
Score 5: The layout is completely correct, and the character count for every measure is exactly 16.
Score 3: There are partial layout errors (e.g., misaligned strings), or approximately half of the measures do not have exactly 16 characters.
Score 1: The layout is mostly incorrect, or almost all measures have an incorrect character count.

2. Musicality Analysis
Voice Arrangement (Voicing): Fingerstyle playing should have distinct layers (bass root notes, middle-voice harmony, and high-voice melody).

Scoring Examples:
Score 5: The tab features a rich melodic range, diverse rhythm patterns, and is aesthetically pleasing to hear.
Score 3: The range and rhythm in the tab are relatively monotonous.
Score 1: The range and rhythm arrangement are illogical and do not constitute actual music.

3. Guitar Fingering Analysis
Fingering Feasibility: Analyze the chords or note combinations that appear at the exact same time point (in the same column). A reasonable stretch is defined as a maximum span of 7 frets or less.

Scoring Examples:
Score 5: The fingering arrangement is logical, with very few instances of simultaneous notes spanning more than 7 frets.
Score 3: There are frequent instances of unreasonable fingering arrangements.
Score 1: The fingering arrangement is completely unreasonable and physically impossible to play.

4. Final Scoring
Please only write out the following at the end of your review:
Technical Layout Score: [Score]/5
Fingering Score: [Score]/5
Musicality Score: [Score]/5
Average Score: [Score]/5
"""

JIANPU_COMPOSER_PROMPT = """
As a top-tier composer and algorithmic music expert, please compose an original melody in a classical style for me.

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

[Scoring Criteria]

Strict Beat Calculation: Since it is in 4/4 time, the total duration of a measure equals one whole note. Therefore, the sum of the fractions for all notes and rests within each measure [must absolutely equal 1 (i.e., 16/16)].
Musicality: Frequently use stepwise motion, logically distribute long and short notes (e.g., combinations of (1/8) and (1/16)), and avoid a monotonous use of (1/4) throughout the piece.
"""

JIANPU_CRITIC_PROMPT = """
As an extremely strict music theory professor at a music conservatory, your task is to review and score AI-generated text-based numbered musical notation (Jianpu) provided by the user. The core objective is to determine the model's proficiency in generating numbered musical notation by analyzing the rhythmic values and musicality of the score. The evaluation requires assessing two separate modules, starting from fundamental principles, to ultimately form a comprehensive assessment with a continuous score ranging from 1 to 5. Please refer to the detailed scoring criteria below.

[Syntax Guide]
| separates measures, and spaces separate complete beats.
A standalone number (e.g., 1, 0) represents a quarter note, occupying 1 beat.
A short dash - is an extension dash, occupying 1 beat.
Two numbers enclosed in parentheses (e.g., (3 4)) represent two eighth notes, which together occupy 1 beat.

Scoring Criteria:
1. Rhythmic Value (Timing) Verification
A correct numbered musical notation score should contain exactly 4.0 beats per measure; that is, the sum of the rhythmic values of all notes between two | symbols must equal 4.0.

Scoring Examples:
Score 5: The sum of the rhythmic values is correct for all measures.
Score 3: The sum of the rhythmic values is incorrect for approximately half of the measures.
Score 1: The sum of the rhythmic values is incorrect for all measures.

2. Aesthetic Analysis
The motif development, melodic contour (judged by the rise and fall of the notation numbers), and rhythmic groove of the score should be rich and logical.

Scoring Examples:
Score 5: The piece features a complete motif, a beautiful melody, and a rich, logical rhythm.
Score 3: The piece features a discernible motif, a listenable melody, and a reasonable rhythm.
Score 1: The piece lacks any motif, the melody is unappealing, and it does not constitute actual music.

Final Scoring
Please only state the following at the end of your evaluation:
Technical Score: [Score]/5
Aesthetic Score: [Score]/5
Average Score: [Score]/5
"""

TASK_CONFIG = {
    "staff": {
        "composer_prompt": STAFF_COMPOSER_PROMPT,
        "critic_prompt": STAFF_CRITIC_PROMPT,
        "output_csv": os.path.join(DATA_ROOT, "data", "ai_music_evaluation_results.csv"),
        "critic_max_tokens": 4000,
    },
    "guitar": {
        "composer_prompt": GUITAR_COMPOSER_PROMPT,
        "critic_prompt": GUITAR_CRITIC_PROMPT,
        "output_csv": os.path.join(DATA_ROOT, "data", "ai_music_evaluation_results_guitar.csv"),
        "critic_max_tokens": 8000,
    },
    "jianpu": {
        "composer_prompt": JIANPU_COMPOSER_PROMPT,
        "critic_prompt": JIANPU_CRITIC_PROMPT,
        "output_csv": os.path.join(DATA_ROOT, "data", "ai_music_evaluation_results_JIAN.csv"),
        "critic_max_tokens": 4000,
    },
}


def run_composer(task, output_csv=None):
    cfg = TASK_CONFIG[task]
    output_csv = output_csv or cfg["output_csv"]

    print(f"🚀 开始表格流水线：准备生成并评审 {TOTAL_SONGS} 首乐谱")

    results = []
    processed_count = 0
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            df_old = pd.read_csv(output_csv)
            results = df_old.to_dict('records')
            processed_count = len(results)
            print(f"📈 检测到进度，已完成 {processed_count} 首，将继续运行...")
        except Exception as e:
            print(f"⚠️ 读取旧 CSV 失败 ({e})，将重新开始。")

    for i in tqdm(range(processed_count, TOTAL_SONGS), initial=processed_count, total=TOTAL_SONGS):
        tqdm.write(f"\n🎵 创作第 {i + 1} 首...")
        abc_score = generate_score(cfg["composer_prompt"])

        if "报错" in abc_score or "失败" in abc_score:
            tqdm.write(f"❌ 生成第 {i + 1} 首失败: {abc_score}")
            continue

        tqdm.write(f"🧐 评审第 {i + 1} 首...")
        critic_reply = evaluate_score(abc_score, cfg["critic_prompt"], max_tokens=cfg["critic_max_tokens"])

        if "报错" in critic_reply or "失败" in critic_reply:
            tqdm.write(f"⚠️ 评审第 {i + 1} 首失败: {critic_reply}")
        else:
            tqdm.write(f"✅ 第 {i + 1} 首完成！(评审报告已收录)")

        record = {
            "曲目编号": i + 1,
            "AI生成的谱子(ABC)": abc_score,
            "AI教授的评审报告": critic_reply,
        }
        results.append(record)

        df_current = pd.DataFrame(results)
        df_current.to_csv(output_csv, index=False, encoding='utf-8-sig')
        time.sleep(3)

    print(f"\n🎉 全部完成！表格已保存至：{output_csv}")


_CURRENT_TASK = "staff"


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SMG 作曲 critic 评测（OpenAI 接口）")
    parser.add_argument("--task", choices=list(TASK_CONFIG), default="staff",
                        help="子任务：staff / guitar / jianpu")
    parser.add_argument("--output", default=None, help="覆盖输出 CSV 路径")
    args = parser.parse_args()
    global _CURRENT_TASK
    _CURRENT_TASK = args.task
    run_composer(args.task, output_csv=args.output)


if __name__ == "__main__":
    main()
