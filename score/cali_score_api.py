#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from api import query_model

# --- 配置 ---
ORIGINAL_FILE = 'RewardTrainer/data/train/skywork_10k_orig_gpt_filtered.json'
BRIDGE_FILE = 'responses/skywork_10k_orig_gpt_filtered_modified.jsonl'
OUTPUT_FILE = 'RewardTrainer/data/train/skywork_10k_orig_gpt_bridge.json'
CHECKPOINT_FILE = OUTPUT_FILE.replace('.json', '.jsonl')  # 临时 checkpoint
API_MODEL = "gpt-4o"
MAX_WORKERS = 30


def load_original_and_bridge():
    with open(ORIGINAL_FILE, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    bridge_entries = []
    with open(BRIDGE_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                bridge_entries.append(json.loads(line.strip()))
    assert len(original_data) == len(bridge_entries), "Data lengths mismatch!"
    print(f"Loaded {len(original_data)} aligned pairs.")
    return original_data, bridge_entries


def build_messages_for_scoring(original_item, bridge_item):
    """构建三轮对话 messages"""
    context = f"""<Start of Context>
##Human: {original_item['prompt']}
<End of Context>"""

    preferred = f"""<Start of Preferred Response>
{original_item['teacher_response']}
<End of Preferred Response>"""

    non_preferred = f"""<Start of Non-Preferred Response>
{original_item['rejected']}
<End of Non-Preferred Response>"""

    instruction_template = """<Start of Instruction>
Given the dialogue context between the human and the assistant, you need to modify the given non-preferred to get a preferred reply from the assistant. Begin your answer with why the given non-preferred response is not preferred compared to the preferred response and how this can be improved. Output your editing by strictly following this format:

<Start of Reasoning>
<End of Reasoning>

<Start of Modified Response>
<End of Modified Response>

You must maintain the original structure of the non-preferred response, ensure the edit distance between the modified response and the non-preferred response is as low as possible.
<End of Instruction>"""

    full_prompt_text = instruction_template + context + preferred + non_preferred

    score_prompt = f"""Now, considering the rejected response was scored {original_item['rejected_score']} out of 10 for quality, helpfulness, and alignment, score the Modified Response in the assistant's reply above on the same 1-10 scale. Output only the numerical score. Do NOT provide any explanation."""

    # 构造三轮 messages
    messages = [
        {"role": "user", "content": full_prompt_text},
        {"role": "assistant", "content": bridge_item['completion']},
        {"role": "user", "content": score_prompt}
    ]
    return messages


def extract_score(text):
    score_match = re.search(r'\b([0-9]+(?:\.[0-9]+)?)\b', text)
    if score_match:
        score = float(score_match.group(1))
        return score
    print(f"Failed to extract score from: {text}")
    return 8.0


def load_completed_ids(checkpoint_file):
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
    print(f"Resuming: {len(completed)} items already scored.")
    return completed


def process_single_pair(index_item):
    idx, original_item, bridge_item = index_item
    try:
        messages = build_messages_for_scoring(original_item, bridge_item)
        response_text = query_model(messages, model_name=API_MODEL)  # ✅ 传入 messages list
        score = extract_score(response_text)

        updated_item = original_item.copy()
        updated_item['chosen'] = bridge_item['response']
        updated_item['chosen_score'] = score
        updated_item['id'] = str(idx)

        return str(idx), updated_item
    except Exception as e:
        print(f"\n❌ Error at index {idx}: {str(e)[:100]}")
        return str(idx), None


def main():
    original_data, bridge_entries = load_original_and_bridge()
    work_items = [(i, original_data[i], bridge_entries[i]) for i in range(len(original_data))]

    completed_ids = load_completed_ids(CHECKPOINT_FILE)
    remaining_items = [item for item in work_items if str(item[0]) not in completed_ids]
    print(f"🔄 Remaining: {len(remaining_items)} items to score")
    if not remaining_items:
        print("🎉 All done!")
        rebuild_final_output()
        return

    def worker(item):
        return process_single_pair(item)

    os.makedirs(os.path.dirname(CHECKPOINT_FILE), exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker, item): str(item[0]) for item in remaining_items}

        pbar = tqdm(total=len(remaining_items), desc="Scoring via API")
        with open(CHECKPOINT_FILE, 'a', encoding='utf-8') as f_out:
            for future in as_completed(futures):
                try:
                    idx_str, result = future.result()
                    if result is not None:
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()
                except Exception as e:
                    idx_str = futures[future]
                    print(f"\nException for index {idx_str}: {e}")
                pbar.update(1)
        pbar.close()

    rebuild_final_output()


def rebuild_final_output():
    """从 checkpoint 重建有序 JSON 数组"""
    if not os.path.exists(CHECKPOINT_FILE):
        print(" No checkpoint found.")
        return

    items_by_id = {}
    with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    item = json.loads(line)
                    items_by_id[item['id']] = item
                except:
                    continue

    sorted_items = []
    idx = 0
    while str(idx) in items_by_id:
        sorted_items.append(items_by_id[str(idx)])
        idx += 1

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(sorted_items, f, ensure_ascii=False, indent=2)

    print(f"Final JSON array saved to {OUTPUT_FILE} ({len(sorted_items)} items)")


if __name__ == "__main__":
    main()