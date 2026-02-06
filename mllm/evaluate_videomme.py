import json
import os
import argparse
from collections import defaultdict

def normalize_prediction(pred):
    
    if not pred or not isinstance(pred, str):
        return ""
    
    pred = pred.strip()
    if len(pred) >= 1:
        return pred[0].lower()
    return ""

def calculate_metrics(input_file):

    stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    overall = {'correct': 0, 'total': 0}

    if not os.path.exists(input_file):
        print(f"Error: file {input_file} not found")
        return

    with open(input_file, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                
                duration = data.get('duration', 'unknown').lower()
                gt = str(data.get('ground_truth', '')).strip().lower()
                raw_pred = data.get('prediction', '')

                pred = normalize_prediction(raw_pred)

                is_correct = (pred == gt and gt != "")

                stats[duration]['total'] += 1
                overall['total'] += 1
                if is_correct:
                    stats[duration]['correct'] += 1
                    overall['correct'] += 1

            except json.JSONDecodeError:
                print(f"Warning: line {line_idx} is invalid，skip.")
            except Exception as e:
                print(f"Warning: line {line_idx} error: {e}")


    print("\n" + "="*40)
    print(f"{'Duration':<15} | {'Accuracy':<10} | {'Details'}")
    print("-" * 40)

    for d_type in ['short', 'medium', 'long']:
        if d_type in stats:
            res = stats[d_type]
            acc = (res['correct'] / res['total']) * 100
            print(f"{d_type:<15} | {acc:>8.2f}% | ({res['correct']}/{res['total']})")

    print("-" * 40)
    if overall['total'] > 0:
        overall_acc = (overall['correct'] / overall['total']) * 100
        print(f"{'Overall':<15} | {overall_acc:>8.2f}% | ({overall['correct']}/{overall['total']})")
    print("="*40 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video-MME static")
    parser.add_argument("--file", type=str, required=True, help="merged result.jsonl")
    args = parser.parse_args()

    calculate_metrics(args.file)
