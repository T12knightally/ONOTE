"""
ONOTE Benchmark: Audio-to-Symbolic Transcription (AST) Pure Evaluation Script
This script calculates the alignment accuracy (Full, Pitch, Duration) using Levenshtein Distance.
It is model-agnostic and reads pre-generated model predictions from a JSON file.
"""

import os
import re
import json
import argparse
import Levenshtein
import pandas as pd
from tqdm import tqdm

# ================= Utility Functions =================

def parse_token(token: str):
    """Parses a note token into pitch and duration."""
    match = re.match(r"([A-Ga-g][#b]?\d)\(([^)]+)\)", token)
    if match:
        return match.group(1), match.group(2)
    return token, ""

def extract_staff_from_json(gt_data: dict) -> list:
    """Extracts the flattened sequence of notes from the ONOTE ground-truth JSON."""
    flat_notes = []
    if not gt_data or "bars" not in gt_data: 
        return []
        
    for bar in gt_data["bars"]:
        staves = bar.get("staves", {})
        for stave_name in ["treble", "bass"]:
            for n in staves.get(stave_name, []):
                p, d = n.get("pitch"), n.get("duration")
                if p and d: 
                    flat_notes.append(f"{p}({d})")
    return flat_notes

def calculate_metric(gt_list: list, trans_list: list) -> float:
    """
    Calculates the alignment accuracy using Levenshtein (Edit) Distance.
    Formula: Acc = max(0, 1 - (ED / max(|S_gt|, |S_pred|)))
    """
    if not gt_list and not trans_list: return 100.0
    if not gt_list or not trans_list: return 0.0
    
    ed = Levenshtein.distance(gt_list, trans_list)
    max_len = max(len(gt_list), len(trans_list))
    
    return max(0.0, 1.0 - (ed / max_len)) * 100.0

def sanitize_ai_output(raw_text: str) -> str:
    """Truncates hallucinations and prevents infinite generation loops."""
    if not raw_text: return ""
    tokens = raw_text.strip().split()
    
    # Limit maximum length to prevent memory overflow or malicious evaluation outputs
    if len(tokens) > 80:
        tokens = tokens[:60]
    
    # Truncate infinite repetition loops (e.g., model continuously outputs C4(1/4) C4(1/4)...)
    for i in range(len(tokens) - 5):
        if len(set(tokens[i:i+6])) == 1:
            return " ".join(tokens[:i+1])
            
    return " ".join(tokens)

# ================= Main Evaluation Logic =================

def main():
    parser = argparse.ArgumentParser(description="ONOTE Pure AST Evaluator")
    parser.add_argument("--metadata", type=str, required=True, help="Path to ground truth metadata.json")
    parser.add_argument("--preds", type=str, required=True, help="Path to model predictions JSON")
    parser.add_argument("--output", type=str, default="ast_eval_results.xlsx", help="Output Excel path")
    args = parser.parse_args()

    # 1. Load Data Files
    if not os.path.exists(args.metadata):
        print(f"[Error] Cannot find ground truth metadata: {args.metadata}")
        return
    if not os.path.exists(args.preds):
        print(f"[Error] Cannot find predictions file: {args.preds}")
        return
        
    with open(args.metadata, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    with open(args.preds, 'r', encoding='utf-8') as f:
        predictions = json.load(f)

    results = []
    
    # 2. Core Evaluation Loop
    for file_key, raw_output in tqdm(predictions.items(), desc="Evaluating Predictions"):
        gt_entry = metadata.get(file_key)
        
        # If the predicted filename is not found in Ground Truth, log a warning and skip
        if not gt_entry: 
            tqdm.write(f"  [Warning] Ground truth missing for key: {file_key}")
            continue
        
        # Extract Ground Truth (GT)
        gt_full = extract_staff_from_json(gt_entry)[:100]
        gt_pitches = [parse_token(t)[0] for t in gt_full]
        gt_durations = [parse_token(t)[1] for t in gt_full]
        
        # Extract and sanitize prediction results
        clean_output = sanitize_ai_output(raw_output)
        trans_full = clean_output.strip().split()
        trans_pitches = [parse_token(t)[0] for t in trans_full]
        trans_durations = [parse_token(t)[1] for t in trans_full]
        
        # Calculate Levenshtein Distance scores
        acc_full = calculate_metric(gt_full, trans_full)
        acc_pitch = calculate_metric(gt_pitches, trans_pitches)
        acc_dur = calculate_metric(gt_durations, trans_durations)
        
        # Save individual result record
        results.append({
            "File_Key": file_key,
            "GT_Count": len(gt_full),
            "Trans_Count": len(trans_full),
            "Full_Acc(%)": round(acc_full, 2),
            "Pitch_Acc(%)": round(acc_pitch, 2),
            "Duration_Acc(%)": round(acc_dur, 2),
            "Raw_Output": raw_output
        })
            
    # 3. Aggregate and Save
    df = pd.DataFrame(results)
    
    # Print global average scores
    print("\n" + "="*40)
    print("🏆 OVERALL BENCHMARK RESULTS")
    print("="*40)
    print(f"Total Evaluated:      {len(df)}")
    print(f"Average Full Acc:     {df['Full_Acc(%)'].mean():.2f}%")
    print(f"Average Pitch Acc:    {df['Pitch_Acc(%)'].mean():.2f}%")
    print(f"Average Duration Acc: {df['Duration_Acc(%)'].mean():.2f}%")
    print("="*40)
    
    df.to_excel(args.output, index=False)
    print(f"\n[Success] Evaluation Complete! Detailed results saved to: {args.output}")

if __name__ == "__main__":
    main()
