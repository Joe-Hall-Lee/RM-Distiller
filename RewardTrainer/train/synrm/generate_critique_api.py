#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from api import query_model

def build_critique_prompt(prompt, response):
    return f"""User: {prompt}
Chatbot: {response}
Please provide a critique of the last response in one short paragraph. Your critique should be concise, specific, insightful and to the point. Aspects you should consider are: (1) Helpfulness. A good response should precisely/closely answer the user's request. (2) Correctness. A good response should be honest and factually correct."""


def load_completed_ids(checkpoint_file):
    """从 checkpoint .jsonl 加载已完成的 id"""
    completed = set()
    if not os.path.exists(checkpoint_file):
        return completed
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        if 'id' in item:
                            completed.add(item['id'])
                    except:
                        continue
    except Exception as e:
        print(f"⚠️ Warning: failed to load checkpoint: {e}")
    return completed


def process_single_item(item_with_id, model_name):
    """处理单条：生成 chosen 和 rejected 的 critique"""
    idx, item = item_with_id
    try:
        # Generate critique for chosen
        prompt_chosen = build_critique_prompt(item['prompt'], item['chosen'])
        critique_chosen = query_model(prompt_chosen, model_name=model_name).strip()

        # Generate critique for rejected
        prompt_rejected = build_critique_prompt(item['prompt'], item['rejected'])
        critique_rejected = query_model(prompt_rejected, model_name=model_name).strip()

        # Append critiques
        new_chosen = item['chosen'] + "\n\n" + critique_chosen
        new_rejected = item['rejected'] + "\n\n" + critique_rejected

        result = {
            "id": str(idx),
            "prompt": item["prompt"],
            "chosen": new_chosen,
            "rejected": new_rejected
        }
        return str(idx), result
    except Exception as e:
        error_msg = f"ERROR: {str(e)[:200]}"
        result = {
            "id": str(idx),
            "prompt": item.get("prompt", ""),
            "chosen": item.get("chosen", "") + "\n\n" + error_msg,
            "rejected": item.get("rejected", "") + "\n\n" + error_msg
        }
        return str(idx), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--num_threads", type=int, default=32)
    args = parser.parse_args()

    # Load input data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {args.input_file}")

    # Prepare work items with id
    work_items = [(i, item) for i, item in enumerate(data)]

    # Checkpoint file (use .jsonl for resume)
    checkpoint_file = args.output_file.replace('.json', '.jsonl')
    completed_ids = load_completed_ids(checkpoint_file)
    remaining_items = [item for item in work_items if str(item[0]) not in completed_ids]
    print(f"Total: {len(work_items)}, Remaining: {len(remaining_items)}")

    if not remaining_items:
        print("All items already processed!")
        rebuild_final_output(checkpoint_file, args.output_file)
        return

    # Multi-threaded processing
    def worker(item):
        return process_single_item(item, args.model_name)

    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {executor.submit(worker, item): str(item[0]) for item in remaining_items}

        pbar = tqdm(total=len(remaining_items), desc="Generating critiques")
        with open(checkpoint_file, 'a', encoding='utf-8') as f_out:
            for future in as_completed(futures):
                try:
                    idx_str, result = future.result()
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()
                except Exception as e:
                    idx_str = futures[future]
                    print(f"\nException for id {idx_str}: {e}")
                pbar.update(1)
        pbar.close()

    # Rebuild final JSON array
    rebuild_final_output(checkpoint_file, args.output_file)


def rebuild_final_output(checkpoint_file, final_output_file):
    """从 checkpoint .jsonl 重建有序 JSON 数组"""
    if not os.path.exists(checkpoint_file):
        print("Checkpoint file not found.")
        return

    items_by_id = {}
    with open(checkpoint_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    items_by_id[item['id']] = item
                except:
                    continue

    # Reconstruct in original order
    sorted_items = []
    idx = 0
    while str(idx) in items_by_id:
        sorted_items.append(items_by_id[str(idx)])
        idx += 1

    # Save final JSON
    with open(final_output_file, 'w', encoding='utf-8') as f:
        json.dump(sorted_items, f, indent=2, ensure_ascii=False)

    print(f"✅ Final output saved to {final_output_file} ({len(sorted_items)} items)")


if __name__ == "__main__":
    main()