"""
ONOTE Benchmark: Scientific Pitch Notation (SPN) OCR Evaluator
Evaluates Vision-Language Models (VLMs) on transcribing sheet music directly into SPN.

This script supports two ablation modes:
1. 'full': Transcribes both Pitch and Duration (e.g., "C4(1/4)").
2. 'pitch_only': Focuses on exhaustive pitch detection for maximum recall.

Usage:
    export VISION_API_KEY="sk-..."
    python evaluate_spn_ocr.py --mode full --metadata data/gt.json --image_dir data/images
"""

import os
import json
import re
import base64
import time
import argparse
import httpx
import pandas as pd
from tqdm import tqdm
from openai import OpenAI, RateLimitError
from difflib import SequenceMatcher

# ================= Prompt Engineering Dictionary =================

PROMPTS = {
    "full": """You are a highly precise Music OCR AI. Your task is to transcribe sheet music images directly into Scientific Pitch Notation with Durations.

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
}""",

    "pitch_only": """You are a highly advanced Music OCR AI specializing in exhaustive pitch detection. 
Your goal is to transcribe every single note visible in the sheet music with maximum recall.

### CORE OBJECTIVES:
1. **Exhaustive Detection**: Identify every note head. Do not skip notes on ledger lines or within chords.
2. **Horizontal Precision**: Scan strictly from left to right. For vertical chords, list pitches from lowest to highest.
3. **Accidental Awareness**: Correctly identify sharps (#), flats (b), and naturals (n).
4. **Pitch Range**: Only recognize pitches within the standard guitar range, typically from **E2 to E6**.

### ANTI-HALLUCINATION & NO-EXAMPLE RULES:
- **STRICT PROHIBITION**: NEVER use pitches or sequences from the "EXAMPLE OUTPUT" below as your actual answer. 
- **REAL-TIME EXTRACTION**: Your output must be based EXCLUSIVELY on the provided image. 
- **UNCERTAINTY HANDLING**: If a note is blurry or unclear, use your best professional judgment based on the context of the staff, but NEVER invent data or copy from examples.

### OUTPUT FORMAT:
- Return ONLY a JSON object with the key "pitches".
- Use Scientific Pitch Notation (e.g., ["C4", "Eb5"]).
- NO durations, NO rhythm, NO measure bars, NO conversational text.
"""
}

# ================= Core Functions =================

def encode_image(image_path: str) -> str:
    """Encodes an image into a Base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_model_prediction(client: OpenAI, model_name: str, image_path: str, prompt_text: str, max_retries: int = 3):
    """Sends the image and prompt to the VLM and robustly extracts JSON."""
    base64_image = encode_image(image_path)

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": prompt_text},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Transcribe the musical notes in this image into the specified JSON format. Do not use example data."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                        ]
                    }
                ],
                temperature=0.01 
            )

            raw_content = response.choices[0].message.content.strip()

            # Regex JSON Extractor to bypass markdown wrapping hallucinations
            match = re.search(r'\{[\s\S]*\}', raw_content)
            clean_json_str = match.group(0) if match else raw_content

            return json.loads(clean_json_str)

        except json.JSONDecodeError:
            tqdm.write(f"      [JSON Parse Error] Attempt {attempt+1} failed. Retrying...")
            time.sleep(2)
        except RateLimitError:
            wait = (attempt + 1) * 3
            tqdm.write(f"      [Rate Limit 429] Waiting {wait} seconds...")
            time.sleep(wait)
        except Exception as e:
            tqdm.write(f"      [Runtime Error] {str(e)}")
            time.sleep(2)

    return None

def calculate_accuracy(pred_list: list, target_list: list, mode: str) -> dict:
    """Calculates Sequence Alignment Accuracy based on the selected mode."""
    if not pred_list or not target_list:
        return {"Pitch Acc": 0.0, "Rhythm Acc": 0.0, "Total Score": 0.0} if mode == "full" else {"Pitch Acc": 0.0, "Total Score": 0.0}

    def get_ratio(seq_a, seq_b):
        return SequenceMatcher(None, seq_a, seq_b).ratio()

    if mode == "pitch_only":
        # Pure string matching for pitch-only arrays (e.g., ["C4", "Eb5"])
        p_acc = get_ratio(pred_list, target_list)
        return {"Pitch Acc": p_acc, "Total Score": p_acc}

    elif mode == "full":
        # Regex splitting for "Pitch(Duration)" arrays (e.g., ["C4(1/4)", "Eb5(1/8)"])
        pattern = r'([A-Ga-g][#b]?\d)\s*\(\s*(\d+/?\d*)\s*\)'
        
        p_pred, r_pred = [], []
        for note in pred_list:
            match = re.search(pattern, str(note).replace('（', '(').replace('）', ')'))
            if match:
                p_pred.append(match.group(1))
                r_pred.append(match.group(2))
                
        p_tar, r_tar = [], []
        for note in target_list:
            match = re.search(pattern, str(note).replace('（', '(').replace('）', ')'))
            if match:
                p_tar.append(match.group(1))
                r_tar.append(match.group(2))

        p_acc = get_ratio(p_pred, p_tar)
        r_acc = get_ratio(r_pred, r_tar)
        return {"Pitch Acc": p_acc, "Rhythm Acc": r_acc, "Total Score": (p_acc * 0.5) + (r_acc * 0.5)}

# ================= Main Pipeline =================

def main():
    parser = argparse.ArgumentParser(description="ONOTE SPN OCR Task Evaluator")
    parser.add_argument("--mode", type=str, choices=["full", "pitch_only"], required=True, help="Evaluation mode: 'full' (Pitch+Duration) or 'pitch_only' (Exhaustive Pitch)")
    parser.add_argument("--metadata", type=str, required=True, help="Path to ground truth JSON")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing score images")
    parser.add_argument("--output", type=str, default=None, help="Output Excel path")
    parser.add_argument("--model", type=str, default="qwen2.5-omni-7b", help="Vision model API name")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API Base URL")
    args = parser.parse_args()

    # 1. API Initialization
    api_key = os.getenv("VISION_API_KEY")
    if not api_key:
        print("❌ Error: VISION_API_KEY environment variable is missing.")
        print("Please run: set VISION_API_KEY=your_key")
        return

    timeout_config = httpx.Timeout(connect=30.0, read=180.0, write=60.0, pool=120.0)
    custom_http_client = httpx.Client(timeout=timeout_config)
    client = OpenAI(api_key=api_key, base_url=args.base_url, http_client=custom_http_client)

    # 2. Data Loading
    if not os.path.exists(args.metadata):
        print(f"❌ Ground truth file missing: {args.metadata}")
        return

    with open(args.metadata, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    # 3. Dynamic Configuration
    output_excel = args.output if args.output else f"spn_ocr_results_{args.mode}.xlsx"
    target_json_key = "notes" if args.mode == "full" else "pitches"
    system_prompt = PROMPTS[args.mode]

    # 4. Checkpoint Recovery
    results_data = []       
    processed_files = set() 
    all_scores = []         

    if os.path.exists(output_excel):
        print(f"📂 Found existing records. Loading checkpoint: {output_excel}")
        try:
            df_existing = pd.read_excel(output_excel, dtype={'File ID': str})
            results_data = df_existing.to_dict('records')
            for row in results_data:
                file_id = str(row.get('File ID', '')).replace(".0", "").zfill(7)
                processed_files.add(file_id)
                if row.get('Status') == 'Success':
                    all_scores.append(row.get('Total Score', 0.0))
            print(f"✅ Resuming {len(processed_files)} records...")
        except Exception as e:
            print(f"⚠️ Failed to read old Excel ({e}). Starting fresh.")

    print(f"🎵 Starting SPN OCR Pipeline in [{args.mode.upper()}] mode for {len(ground_truth)} files...")

    # 5. Evaluation Loop
    for file_id, target in tqdm(ground_truth.items(), desc=f"🤖 Evaluating {args.mode}", initial=len(processed_files), total=len(ground_truth)):
        
        file_id = str(file_id).zfill(7)
        img_path = os.path.join(args.image_dir, f"{file_id}.png")

        if not os.path.exists(img_path):
            tqdm.write(f"      [Warning] Image missing for {file_id}. Skipping.")
            continue
            
        if file_id in processed_files:
            continue

        tqdm.write(f"\n🎧 Processing: {file_id}")
        prediction = get_model_prediction(client, args.model, img_path, system_prompt)
        
        new_row = {'File ID': file_id, 'Mode': args.mode}

        if prediction and isinstance(prediction, dict):
            # Extract lists based on prompt rules
            pred_list = prediction.get(target_json_key, [])
            target_list = target.get(target_json_key, [])
            
            scores = calculate_accuracy(pred_list, target_list, args.mode)
            
            tqdm.write(f"      Pitch Acc:  {scores.get('Pitch Acc', 0):.2%}")
            if args.mode == "full":
                tqdm.write(f"      Rhythm Acc: {scores.get('Rhythm Acc', 0):.2%}")
            tqdm.write(f"      >> Overall: {scores['Total Score']:.2%}")

            all_scores.append(scores['Total Score'])
            
            new_row.update({
                'Status': 'Success',
                **scores,
                'Raw Output': json.dumps(prediction, ensure_ascii=False)
            })
        else:
            tqdm.write("      [Error] Result: Failed to generate valid JSON or key.")
            new_row.update({'Status': 'Failed', 'Total Score': 0.0, 'Raw Output': str(prediction)})

        results_data.append(new_row)
        pd.DataFrame(results_data).to_excel(output_excel, index=False)
        processed_files.add(file_id)
        
        time.sleep(2)

    # 6. Final Report
    if all_scores:
        avg = sum(all_scores) / len(all_scores)
        print("\n" + "=" * 60)
        print(f"🎉 SPN OCR EVALUATION COMPLETE [{args.mode.upper()}]!")
        print(f"📁 Detailed report saved to: {output_excel}")
        print(f"🏆 Final Benchmarking Result (Average): {avg:.2%}")
        print("=" * 60)
    else:
        print("\n⚠️ No successful records processed.")

if __name__ == "__main__":
    main()
