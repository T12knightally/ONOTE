"""
ONOTE Benchmark: Visual Score Understanding (VSU) Evaluation Script
This script evaluates Vision-Language Models (VLMs) on their ability to visually 
read and interpret musical scores (Jianpu/Staff/Tab) via Multiple Choice Questions (MCQs).

Key Features:
- Direct Image Parsing: Feeds Base64 encoded PNG/JPG images directly to the VLM.
- Robust Network Client: Custom HTTPX timeout configurations to prevent API disconnections.
- Seamless Checkpointing: Reads existing Excel results to resume evaluations without data loss.
- Regex Answer Extraction: Safely extracts the A/B/C/D choice from verbose AI responses.

Usage:
    export VISION_API_KEY="sk-..."
    python evaluate_vsu.py --qa_json data/jianpu_qa.json --image_dir data/images --output results_vsu.xlsx
"""

import os
import json
import time
import base64
import argparse
import mimetypes
import re
import pandas as pd
import httpx
from tqdm import tqdm
from openai import OpenAI

# ================= Prompt Engineering =================

def construct_mcq_prompt(question: str, options: list) -> str:
    """Constructs a strict Multiple Choice Question prompt."""
    prompt = f"{question}\n\nOptions:\n"
    for i, opt in enumerate(options):
        letter = chr(65 + i) 
        prompt += f"{letter}. {opt}\n"
        
    prompt += "\n【CRITICAL REQUIREMENT】\n"
    prompt += """You are taking a Jianpu (Numbered Musical Notation) exam.
1. Format each note: [Pitch Modifier][Note Number]([Duration Fraction])
   - Note Number: 1-7 for pitch, 0 for rest.
   - Pitch Modifier: _ for one octave lower (_5), __ for two octaves lower (__2), ^ for one octave higher (^1). No prefix for middle register.
   - Duration Fraction: Fractions in parentheses, e.g., (1/4), (1/8), (1/16), (3/16), (1/2).
2. Measures: Use '|' to separate measures.
3. Separation: Use a single space to separate notes."""
    prompt += "\nPlease answer by outputting ONLY the single letter of the correct option (A, B, C, or D). Do not explain or add any other text."
    return prompt

def extract_answer_letter(raw_response: str) -> str:
    """Extracts the exact option letter from the AI's response."""
    text = raw_response.strip().upper()
    match = re.search(r'^([A-D])$', text)
    if match: return match.group(1)
    match = re.search(r'\b([A-D])\b', text)
    if match: return match.group(1)
    match = re.search(r'([A-D])', text)
    if match: return match.group(1)
    return "UNKNOWN"

# ================= Vision API Invocation =================

def evaluate_vsu_question(client: OpenAI, model_name: str, image_path: str, prompt: str, max_retries: int = 3) -> str:
    """Encodes the image and queries the Vision-Language Model."""
    if not os.path.exists(image_path):
        return "[Error] Image Not Found"
        
    for attempt in range(max_retries): 
        try:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
                
            mime_type, _ = mimetypes.guess_type(image_path)
            if not mime_type:
                mime_type = "image/jpeg"
                
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.01 
            )
            
            raw_text = response.choices[0].message.content
            if not raw_text or raw_text.strip() == "":
                tqdm.write("      [Warning] AI returned empty content. Retrying...")
                time.sleep(2)
                continue
                
            return raw_text.strip()
            
        except Exception as e:
            tqdm.write(f"      [Retry {attempt+1}] API Call Error: {e}")
            time.sleep(2)
            
    return "[Error] API Call Failed"

# ================= Main Pipeline =================

def main():
    parser = argparse.ArgumentParser(description="ONOTE VSU Task Evaluator")
    parser.add_argument("--qa_json", type=str, required=True, help="Path to the JSON file containing questions")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing score images")
    parser.add_argument("--output", type=str, default="vsu_results.xlsx", help="Output Excel path")
    parser.add_argument("--model", type=str, default="qwen2.5-omni-7b", help="Vision model API name")
    parser.add_argument("--base_url", type=str, default="https://dashscope.aliyuncs.com/compatible-mode/v1", help="API Base URL")
    args = parser.parse_args()

    # 1. API Initialization
    api_key = os.getenv("VISION_API_KEY")
    if not api_key:
        print("❌ Error: VISION_API_KEY environment variable is missing.")
        return

    # Custom HTTP client to prevent timeout drops during heavy Vision tasks
    timeout_config = httpx.Timeout(connect=30.0, read=180.0, write=60.0, pool=120.0)
    custom_http_client = httpx.Client(timeout=timeout_config)
    
    client = OpenAI(api_key=api_key, base_url=args.base_url, http_client=custom_http_client)

    # 2. Data Loading
    if not os.path.exists(args.qa_json):
        print(f"❌ Question Bank JSON not found: {args.qa_json}")
        return
        
    with open(args.qa_json, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
        
    print(f"🎵 Successfully loaded {len(qa_data)} VSU questions. Preparing exam...")
    
    # 3. Checkpoint Recovery System
    results = []
    correct_count = 0
    valid_count = 0
    processed_doc_ids = set()

    if os.path.exists(args.output):
        try:
            # Enforce string dtype for Doc ID to prevent pandas from turning "123" into "123.0"
            df_old = pd.read_excel(args.output, dtype={'Doc ID': str})
            results = df_old.to_dict('records')
            
            for row in results:
                doc_id = str(row.get("Doc ID", "")).replace(".0", "").strip() 
                processed_doc_ids.add(doc_id)
                valid_count += 1
                if row.get("Is Correct") == True:
                    correct_count += 1
                    
            print(f"📈 Found existing scorecard! {len(processed_doc_ids)} completed. Resuming...")
        except Exception as e:
            print(f"⚠️ Failed to read old Excel ({e}). Starting fresh.")
            results, correct_count, valid_count, processed_doc_ids = [], 0, 0, set()
    
    # Filter out completed tasks
    pending_tasks = []
    for item in qa_data:
        doc_id = str(item.get("doc_id", "")).strip()
        if doc_id not in processed_doc_ids:
            pending_tasks.append(item)
            
    if not pending_tasks:
        print("🎉 All questions have been evaluated! Final Report:")
        accuracy = (correct_count / valid_count) * 100 if valid_count > 0 else 0
        print(f"🎯 Valid Answers: {valid_count} | ✅ Correct: {correct_count} | 🏆 Accuracy: {accuracy:.2f}%")
        return

    # 4. Evaluation Loop
    for item in tqdm(pending_tasks, desc="🤖 AI VSU Exam in Progress", initial=len(processed_doc_ids), total=len(qa_data)):
        doc_id = str(item.get("doc_id", "")).strip()
            
        question = item.get("question", "")
        options = item.get("options", [])
        ground_truth = str(item.get("answer", "")).upper()
        
        # Determine image format (Checks both PNG and JPG)
        img_path_png = os.path.join(args.image_dir, f"{doc_id}_1.png")
        img_path_jpg = os.path.join(args.image_dir, f"{doc_id}_1.jpg")
        image_path = img_path_png if os.path.exists(img_path_png) else img_path_jpg
        
        if not os.path.exists(image_path):
            tqdm.write(f"⚠️ Image not found for Doc ID: {doc_id}. Skipping.")
            continue
            
        prompt = construct_mcq_prompt(question, options)
        raw_response = evaluate_vsu_question(client, args.model, image_path, prompt)
        
        if "[Error]" not in raw_response:
            predicted_letter = extract_answer_letter(raw_response)
            is_correct = (predicted_letter == ground_truth)
            
            if is_correct:
                correct_count += 1
                tqdm.write(f"✅ {doc_id} | Pred: {predicted_letter} | GT: {ground_truth}")
            else:
                tqdm.write(f"❌ {doc_id} | Pred: {predicted_letter} | GT: {ground_truth} (Raw: {raw_response[:20]})")
                
            valid_count += 1
            results.append({
                "Doc ID": doc_id,
                "Question": question,
                "Ground Truth": ground_truth,
                "AI Prediction": predicted_letter,
                "Is Correct": is_correct,
                "AI Raw Output": raw_response
            })
            
            processed_doc_ids.add(doc_id)
            pd.DataFrame(results).to_excel(args.output, index=False)
            
        else:
            tqdm.write(f"⚠️ {doc_id} API Request Failed: {raw_response}")
            
        time.sleep(2) 
        
    # 5. Final Report
    if valid_count > 0:
        accuracy = (correct_count / valid_count) * 100
        print("\n" + "="*50)
        print("🎓 VSU EXAM COMPLETE! FINAL REPORT:")
        print(f"🎯 Total Valid Answers: {valid_count}")
        print(f"✅ Correct Answers:     {correct_count}")
        print(f"🏆 Overall Accuracy:    {accuracy:.2f}%")
        print(f"📁 Detailed results saved to: {args.output}")
        print("="*50)

if __name__ == "__main__":
    main()
