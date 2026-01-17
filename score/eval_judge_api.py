#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from api import query_model

# --- 配置 ---
INPUT_JSONL = "RewardTrainer\data\eval/arabic.jsonl"
ALL_OUTPUT_FILE = "RewardTrainer\data\eval/arabic_all.jsonl"
FILTERED_OUTPUT_FILE = "RewardTrainer\data\eval/arabic_filtered.jsonl"
CHECKPOINT_FILE = ALL_OUTPUT_FILE  # 用 all 文件作 checkpoint
JUDGE_MODEL = "gpt-5"
MAX_WORKERS = 30

SCORING_PROMPT_TEMPLATE = """[Question]
{question}

[The Start of Assistant 1's Answer]
{answer_1}

[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer_2}

[The End of Assistant 2's Answer]

[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. Avoid any potential bias and ensure that the order in which the responses were presented does not affect your judgment. Do NOT provide any explanation.

### Response:"""


def parse_scores(text):
    """返回 (score1, score2) 元组"""
    if not text or not text.strip():
        return 0.0, 0.0
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if len(matches) >= 2:
        try:
            s1 = float(matches[0])
            s2 = float(matches[1])
            return max(0.0, min(10.0, s1)), max(0.0, min(10.0, s2))
        except:
            pass
    return 0.0, 0.0


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
        print(f"⚠️ Warning: failed to load checkpoint: {e}")
    print(f"✅ Resuming: {len(completed)} items already processed.")
    return completed


def process_single_item(item):
    """处理单条：调用 API 并判断一致性"""
    try:
        prompt = item.get("prompt", "").strip()
        chosen = item.get("chosen", "").strip()
        rejected = item.get("rejected", "").strip()

        if not prompt or not chosen or not rejected:
            return item, False, "0 0"

        full_prompt = SCORING_PROMPT_TEMPLATE.format(
            question=prompt,
            answer_1=chosen,
            answer_2=rejected
        )

        response_text = query_model(full_prompt, model_name=JUDGE_MODEL)
        score1, score2 = parse_scores(response_text)
        scores_str = f"{int(score1) if score1.is_integer() else score1} {int(score2) if score2.is_integer() else score2}"

        # 判断是否一致：API 也认为 chosen (Assistant 1) > rejected (Assistant 2)
        is_consistent = score1 > score2

        return item, is_consistent, scores_str

    except Exception as e:
        print(
            f"\n❌ Error processing id {item.get('id', 'unknown')}: {str(e)[:100]}")
        return item, False, "0 0"


def main():
    # Step 1: Load input JSONL
    data = []
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    print(f"✅ Loaded {len(data)} items from {INPUT_JSONL}")

    # Step 2: Load completed ids
    completed_ids = load_completed_ids(CHECKPOINT_FILE)
    remaining_items = [item for item in data if item.get(
        'id') not in completed_ids]
    print(f"🔄 Remaining: {len(remaining_items)} items to process")

    if not remaining_items:
        print("🎉 All done! Proceeding to filter...")
    else:
        # Step 3: Multi-threaded processing
        def worker(item):
            return process_single_item(item)

        os.makedirs(os.path.dirname(ALL_OUTPUT_FILE), exist_ok=True)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(worker, item)
                       for item in remaining_items]

            pbar = tqdm(total=len(remaining_items),
                        desc="Evaluating consistency")
            with open(ALL_OUTPUT_FILE, 'a', encoding='utf-8') as f_all:
                for future in as_completed(futures):
                    try:
                        original_item, is_consistent, scores_str = future.result()
                        # Add new fields
                        output_item = original_item.copy()
                        output_item["scores"] = scores_str
                        output_item["consistent"] = is_consistent
                        f_all.write(json.dumps(
                            output_item, ensure_ascii=False) + '\n')
                        f_all.flush()
                    except Exception as e:
                        print(f"\n⚠️ Thread error: {e}")
                    pbar.update(1)
            pbar.close()

    # Step 4: Load all results and filter consistent ones
    print("🔍 Filtering consistent preferences...")
    consistent_items = []
    with open(ALL_OUTPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if item.get("consistent", False):
                    # 移除内部字段
                    clean_item = {k: v for k, v in item.items() if k not in [
                        "scores", "consistent"]}
                    consistent_items.append(clean_item)

    # Step 5: Save filtered results (same format as input)
    with open(FILTERED_OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for item in consistent_items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ Done!")
    print(f"   - All results (with scores): {ALL_OUTPUT_FILE}")
    print(
        f"   - Consistent only ({len(consistent_items)} items): {FILTERED_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
