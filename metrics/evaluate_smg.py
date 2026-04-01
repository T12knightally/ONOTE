"""
ONOTE Benchmark: Symbolic Music Generation (SMG) Dual-Agent Pipeline
This script utilizes a "Composer-Critic" paradigm to evaluate the generative 
capabilities of Omnimodal LLMs across different musical notation formats.

Usage:
    export COMPOSER_API_KEY="sk-..."
    export CRITIC_API_KEY="sk-..."
    python generate_smg.py --format tab --total 80 --output results_tab.csv
"""

import os
import time
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

# ================= Prompt Dictionary =================
# Stores the specific Composer and Critic prompts for the 3 heterogeneous formats
PROMPTS = {
    "abc": {
        "composer": """As a top-tier composer, please compose an original classical-style melody for me.
Key Signature: C Major
Time Signature: 4/4
Length: Strictly 8 measures.
Format: Please output ONLY standard ABC Notation code.

[Scoring Criteria]
Rhythmic Value Calculation: Since the time signature is 4/4, the sum of the rhythmic values for all notes and rests within a single measure [must absolutely equal 4.0 beats]. A quarter note equals 1 beat, and an eighth note equals 0.5 beats.
Musicality: The rhythm should be diverse, incorporating various rhythmic patterns such as dotted notes. Additionally, the melody should be beautiful, and the musical motif must be complete and well-developed.
""",
        "critic": """As an extremely strict music theory professor at a music conservatory, your task is to review and score AI-generated text-based ABC notation provided by the user. The core objective is to determine the model's proficiency in generating numbered musical notation by analyzing the rhythmic values and musicality of the score. The evaluation requires assessing two separate modules, starting from fundamental principles, to ultimately form a comprehensive assessment with a continuous score ranging from 1 to 5. Please refer to the detailed scoring criteria below.

Scoring Criteria:
1. Rhythmic Value (Timing) Verification
A correct numbered musical notation score should contain exactly 4.0 beats per measure; that is, the sum of the rhythmic values of all notes between two | symbols must equal 4.0.

Scoring Examples:
Score 5: The sum of the rhythmic values is correct for all measures.
Score 3: The sum of the rhythmic values is incorrect for approximately half of the measures.
Score 1: The sum of the rhythmic values is incorrect for all measures.

2. Aesthetic Analysis
The motif development, melodic contour, and rhythmic groove of the score should be rich and logical.

Final Scoring
Please [only!!!] state the following at the end of your evaluation, no need to analyze:
Technical Score: [Score]/5
Aesthetic Score: [Score]/5
Average Score: [Score]/5
"""
    },
    
    "tab": {
        "composer": """Act as a top-tier fingerstyle guitar master. Please compose a classical-style acoustic guitar melody for me.
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
Beat|1 . . . 2 . . . 3 . . . 4 . . . |
e   |--------------------------------|
B   |--------1---------------1-------|
G   |----0-------0-------0-------0---|
D   |--2---------------2-------------|
A   |3---------------3---------------|
E   |--------------------------------|

[Scoring Criteria]
Strict Layout: The numbers and hyphens across the 6 strings must be absolutely vertically aligned. You must never add or omit any hyphens.
Fingering Feasibility: Please ensure that notes meant to be played simultaneously are physically possible for a human left hand to fret (max stretch of 7 frets).
Musicality: Fingerstyle playing should have distinct layers (Low bass root notes, middle-voice chords, and high treble melody).
""",
        "critic": """As an extremely strict fingerstyle guitar master, your task is to review and score AI-generated ASCII guitar tablature (tabs) provided by the user. Your core objective is to evaluate the model's proficiency in generating guitar tabs by analyzing the layout and timing, fingering, and musicality of the sheet music. The evaluation requires assessing three separate modules with a continuous score ranging from 1 to 5.

1. Layout Alignment and Timing Verification
Check Vertical Alignment: In a correctly formatted guitar tab, the 6 strings must be perfectly aligned vertically.
Calculate Character Count per Measure: Assuming a 4/4 time signature, the number of characters (dashes plus numbers) on each string between two bar lines (|) must be completely identical, and each measure should contain a length of exactly 16 characters.

2. Musicality Analysis
Voice Arrangement (Voicing): Fingerstyle playing should have distinct layers (bass root notes, middle-voice harmony, and high-voice melody).

3. Guitar Fingering Analysis
Fingering Feasibility: Analyze the chords or note combinations that appear at the exact same time point (in the same column). A reasonable stretch is defined as a maximum span of 7 frets or less.

Final Scoring
Please [only!!!] state the following at the end of your evaluation, no need to analyze:
Technical Layout Score: [Score]/5
Fingering Score: [Score]/5
Musicality Score: [Score]/5
Average Score: [Score]/5
"""
    },
    
    "jianpu": {
        "composer": """As a top-tier composer and algorithmic music expert, please compose an original melody in a classical style for me.
Key Signature: C Major (1=C)
Time Signature: 4/4
Length: 8 measures.
Format: Please only output the structured text-based numbered musical notation (Jianpu) with precise fractional rhythmic values as provided below.

[Structured Jianpu Syntax]
You must strictly use the following format to generate each note or rest: [Pitch Modifier][Note Number]([Duration Fraction])
Note Number: Use 1-7 to represent pitch, and 0 for a rest.
Pitch Modifier: Add _ before the number to indicate one octave lower (e.g., _5), and __ for two octaves lower. No prefix means the middle register (e.g., 3), and ^ indicates one octave higher.
Duration Fraction: Strictly represented by fractions enclosed in parentheses. A quarter note is written as (1/4), an eighth note as (1/8), a sixteenth note as (1/16), a dotted eighth note as (3/16), and a half note as (1/2).
Measures and Separation: Use | to separate measures. Use a single space to separate notes.

[Scoring Criteria]
Strict Beat Calculation: Since it is in 4/4 time, the total duration of a measure equals one whole note. Therefore, the sum of the fractions for all notes and rests within each measure [must absolutely equal 1 (i.e., 16/16)].
Musicality: Frequently use stepwise motion, logically distribute long and short notes.
""",
        "critic": """As an extremely strict music theory professor at a music conservatory, your task is to review and score AI-generated text-based numbered musical notation (Jianpu) provided by the user. The core objective is to determine the model's proficiency in generating numbered musical notation by analyzing the rhythmic values and musicality of the score.

[Syntax Guide]
| separates measures, and spaces separate complete beats.
A standalone number (e.g., 1, 0) represents a quarter note, occupying 1 beat.
A short dash - is an extension dash, occupying 1 beat.
Two numbers enclosed in parentheses (e.g., (3 4)) represent two eighth notes, which together occupy 1 beat.

Scoring Criteria:
1. Rhythmic Value (Timing) Verification
A correct numbered musical notation score should contain exactly 4.0 beats per measure; that is, the sum of the rhythmic values of all notes between two | symbols must equal 4.0.

2. Aesthetic Analysis
The motif development, melodic contour (judged by the rise and fall of the notation numbers), and rhythmic groove of the score should be rich and logical.

Final Scoring
Please only state the following at the end of your evaluation:
Technical Score: [Score]/5
Aesthetic Score: [Score]/5
Average Score: [Score]/5
"""
    }
}

# ================= Core Functions =================

def generate_score(client: OpenAI, model: str, prompt: str, max_retries: int = 3) -> str:
    """Invokes the Composer LLM to generate the musical score."""
    for attempt in range(max_retries):
        try:
            combined_prompt = f"{prompt}\n\nTask: Please generate the score now based on the instructions above."
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": combined_prompt}],
                temperature=0.8,
                max_tokens=2048
            )
            
            content = response.choices[0].message.content
            
            if not content or content.strip() == "":
                tqdm.write(f"  [Warning] Composer returned empty content. Reason: {response.choices[0].finish_reason}")
                time.sleep(3)
                continue
                
            return content.strip()
            
        except Exception as e:
            if "limit" in str(e).lower() or "429" in str(e):
                time.sleep((attempt + 1) * 3)
            else:
                return f"[Error] Generation failed: {e}"
                
    return "[Error] Timeout or consecutive empty returns."

def evaluate_score(client: OpenAI, model: str, system_prompt: str, generated_score: str, max_retries: int = 3) -> str:
    """Invokes the Critic LLM to evaluate the generated score."""
    for attempt in range(max_retries):
        try:
            combined_prompt = f"{system_prompt}\n\n[Generated Score for Review]\n{generated_score}"
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": combined_prompt}],
                temperature=0.1,
                max_tokens=4000
            )
            
            content = response.choices[0].message.content
            
            if not content or content.strip() == "":
                tqdm.write("  [Warning] Critic returned empty content. Retrying...")
                time.sleep(2)
                continue
                
            return content.strip()
            
        except Exception as e:
            if "limit" in str(e).lower() or "429" in str(e):
                time.sleep((attempt + 1) * 3)
            else:
                return f"[Error] Evaluation failed: {e}"
                
    return "[Error] Timeout or consecutive empty returns."

# ================= Main Pipeline =================

def main():
    parser = argparse.ArgumentParser(description="ONOTE SMG Dual-Agent Pipeline")
    parser.add_argument("--format", type=str, choices=["abc", "tab", "jianpu"], required=True, help="Target notation format")
    parser.add_argument("--total", type=int, default=80, help="Total number of songs to generate")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (defaults to results_{format}.csv)")
    parser.add_argument("--composer", type=str, default="qwen3-omni-30b-a3b-captioner", help="Composer model name")
    parser.add_argument("--critic", type=str, default="gpt-4o-mini", help="Critic model name")
    args = parser.parse_args()

    # 1. Environment Variable Checks
    composer_key = os.getenv("COMPOSER_API_KEY")
    critic_key = os.getenv("CRITIC_API_KEY")
    
    if not composer_key or not critic_key:
        print("❌ Error: API Keys are not set in environment variables.")
        print("Please set COMPOSER_API_KEY and CRITIC_API_KEY before running.")
        return

    # Base URLs can be optionally overridden in env vars
    composer_url = os.getenv("COMPOSER_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    critic_url = os.getenv("CRITIC_BASE_URL", "https://api.apiyi.com/v1")

    # 2. Initialize Clients
    composer_client = OpenAI(api_key=composer_key, base_url=composer_url)
    critic_client = OpenAI(api_key=critic_key, base_url=critic_url)

    # 3. Setup File Paths & Prompts
    output_csv = args.output if args.output else f"results_{args.format}.csv"
    prompt_set = PROMPTS[args.format]
    
    print(f"🚀 Starting Dual-Agent Pipeline for format: [{args.format.upper()}]")
    print(f"Composer: {args.composer} | Critic: {args.critic}")
    print(f"Target: {args.total} songs | Output: {output_csv}")
    
    results = []
    processed_count = 0
    
    # Checkpoint Recovery
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        try:
            df_old = pd.read_csv(output_csv)
            results = df_old.to_dict('records')
            processed_count = len(results)
            print(f"📈 Found existing progress: {processed_count} songs completed. Resuming...")
        except Exception as e:
            print(f"⚠️ Failed to read old CSV ({e}). Starting fresh.")

    # 4. Generation & Evaluation Loop
    for i in tqdm(range(processed_count, args.total), initial=processed_count, total=args.total):
        
        tqdm.write(f"\n🎵 Composing Song #{i+1}...")
        generated_score = generate_score(composer_client, args.composer, prompt_set["composer"])
        
        if "[Error]" in generated_score:
            tqdm.write(f"❌ Generation Failed for #{i+1}: {generated_score}")
            continue
            
        tqdm.write(f"🧐 Critic Evaluating Song #{i+1}...")
        critic_reply = evaluate_score(critic_client, args.critic, prompt_set["critic"], generated_score)
        
        if "[Error]" in critic_reply:
            tqdm.write(f"⚠️ Evaluation Failed for #{i+1}: {critic_reply}")
        else:
            tqdm.write(f"✅ Song #{i+1} completed successfully!")
        
        # Save Record
        record = {
            "Song_ID": i + 1,
            "Format": args.format.upper(),
            "Generated_Score": generated_score,
            "Critic_Review": critic_reply
        }
        results.append(record)
        
        # Save to CSV dynamically
        df_current = pd.DataFrame(results)
        df_current.to_csv(output_csv, index=False, encoding='utf-8-sig')
            
        time.sleep(2)

    print(f"\n🎉 Pipeline Complete! Results saved to: {output_csv}")

if __name__ == "__main__":
    main()
