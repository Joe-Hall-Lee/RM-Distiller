#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from api import query_model

# --- 配置 ---
INPUT_FILE = 'RewardTrainer/data/train/skywork_10k_orig_gpt_filtered.json'
OUTPUT_FILE = 'responses/skywork_10k_orig_gpt_modified.jsonl'
CHECKPOINT_FILE = OUTPUT_FILE  # 用输出文件作 checkpoint
API_MODEL = "gpt-4o"
MAX_WORKERS = 30

# --- 固定 Instruction 模板 ---
INSTRUCTION_TEMPLATE = """<Start of Instruction>
Given the question from the user, you need to modify the given non-preferred to get a preferred reply from the assistant. Begin your answer with why the given non-preferred response is not preferred compared to the preferred response and how this can be improved. Output your editing by strictly following this format:

<Start of Reasoning>
<End of Reasoning>

<Start of Modified Response>
<End of Modified Response>

You must maintain the original structure of the non-preferred response, ensure the edit distance between the modified response and the non-preferred response is as low as possible.
<End of Instruction>
"""


def build_full_prompt(item):
    context = f"""
<Start of Question>
{item['prompt']}
<End of Question>
"""
    preferred = f"""
<Start of Preferred Response>
{item['teacher_response']}
<End of Preferred Response>
"""
    non_preferred = f"""
<Start of Non-Preferred Response>
{item['rejected']}
<End of Non-Preferred Response>"""
    return INSTRUCTION_TEMPLATE + context + preferred + non_preferred


def extract_modified_response(generated_text):
    match = re.search(
        r'<Start of Modified Response>\s*(.*?)\s*<End of Modified Response>',
        generated_text,
        re.DOTALL
    )
    if match:
        return match.group(1).strip()

    match = re.search(
        r'<Start of Modified Response>\s*(.*?)\s*</End of Modified Response>',
        generated_text,
        re.DOTALL
    )
    if match:
        return match.group(1).strip()

    if '<End of Reasoning>' in generated_text:
        return generated_text.split('<End of Reasoning>')[-1].strip()

    return generated_text.strip()


def load_completed_ids(checkpoint_file):
    """从输出文件加载已完成的 id（字符串索引）"""
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
        print(f"Warning: failed to load checkpoint: {e}")
    print(f"Resuming: {len(completed)} items already processed.")
    return completed


def process_single_item(item_with_id):
    """处理带 id 的单条数据"""
    item, item_id = item_with_id
    try:
        prompt = build_full_prompt(item)
        completion = query_model(prompt, model_name=API_MODEL)
        response = extract_modified_response(completion)

        return {
            "id": item_id,
            "question": item["prompt"],
            "model": API_MODEL,
            "completion": completion,
            "response": response
        }
    except Exception as e:
        print(f"\nError processing item {item_id}: {str(e)[:100]}")
        return None


def main():
    # Step 1: Load input data
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items")

    # Step 2: Assign id = str(index)
    data_with_id = [(item, str(i)) for i, item in enumerate(data)]

    # Step 3: Load completed ids
    completed_ids = load_completed_ids(CHECKPOINT_FILE)

    # Step 4: Filter unprocessed
    work_items = [
        (item, item_id) for item, item_id in data_with_id
        if item_id not in completed_ids
    ]
    print(f"Remaining: {len(work_items)} items to process")
    if not work_items:
        print("All done!")
        return

    # Step 5: Multi-threaded processing with resume
    def worker(pair):
        return process_single_item(pair)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker, pair)
                                   : pair[1] for pair in work_items}

        pbar = tqdm(total=len(work_items), desc="Rewriting via API")
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        f_out.write(json.dumps(
                            result, ensure_ascii=False) + '\n')
                        f_out.flush()
                except Exception as e:
                    item_id = futures[future]
                    print(f"\nThread error for item {item_id}: {e}")
                pbar.update(1)
        pbar.close()

    print(f"\nFinished! Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
