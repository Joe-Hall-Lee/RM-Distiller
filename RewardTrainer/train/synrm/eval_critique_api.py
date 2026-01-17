#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from api import query_model

def make_critique_prompt(prompt, response):
    return f"""User: {prompt}
Chatbot: {response}
Please provide a critique of the last response in one short paragraph. Your critique should be concise, specific, insightful and to the point. Aspects you should consider are: (1) Helpfulness. A good response should precisely/closely answer the user's request. (2) Correctness. A good response should be honest and factually correct."""


def load_completed_ids(checkpoint_file):
    """从 checkpoint .jsonl 加载已完成的 id（字符串）"""
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
        # Critique for chosen
        prompt_chosen = make_critique_prompt(item['prompt'], item['chosen'])
        critique_chosen = query_model(prompt_chosen, model_name=model_name).strip()

        # Critique for rejected
        prompt_rejected = make_critique_prompt(item['prompt'], item['rejected'])
        critique_rejected = query_model(prompt_rejected, model_name=model_name).strip()

        # Append critiques
        item['chosen'] = item['chosen'] + "\n\n" + critique_chosen
        item['rejected'] = item['rejected'] + "\n\n" + critique_rejected
        item['id'] = str(idx)  # 添加原始行号 id

        return str(idx), item
    except Exception as e:
        error_msg = f"ERROR: {str(e)[:200]}"
        item['chosen'] = item.get('chosen', '') + "\n\n" + error_msg
        item['rejected'] = item.get('rejected', '') + "\n\n" + error_msg
        item['id'] = str(idx)
        return str(idx), item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output_file", type=str, required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--num_threads", type=int, default=32)
    args = parser.parse_args()

    # Load input JSONL
    data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"✅ Loaded {len(data)} items from {args.input_file}")

    # Prepare work items with id
    work_items = [(i, item) for i, item in enumerate(data)]

    # Use output_file as checkpoint (JSONL)
    checkpoint_file = args.output_file
    completed_ids = load_completed_ids(checkpoint_file)
    remaining_items = [item for item in work_items if str(item[0]) not in completed_ids]
    print(f"🔄 Total: {len(work_items)}, Remaining: {len(remaining_items)}")

    if not remaining_items:
        print("🎉 All done!")
        return

    # Multi-threaded processing
    def worker(item):
        return process_single_item(item, args.model_name)

    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)

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
                    print(f"\n⚠️ Exception for id {idx_str}: {e}")
                pbar.update(1)
        pbar.close()

    print(f"✅ Finished! Results saved to {args.output_file}")


if __name__ == "__main__":
    main()