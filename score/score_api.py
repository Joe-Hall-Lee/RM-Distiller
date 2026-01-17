#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from api import query_model

# --- 配置 ---
INPUT_FILE = "F:\CS\AI\DistilRM\RewardTrainer\data/train\skywork_10k_gpt_bridge.json"
OUTPUT_FILE = "evaluations/skywork_10k_gpt-4o_bridge_uncalibrated.json.jsonl"
CHECKPOINT_FILE = OUTPUT_FILE
JUDGE_MODEL = "gpt-4o"
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


def parse_scores_to_string(text):
    if not text or not text.strip():
        return "0 0"
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    if len(matches) >= 2:
        try:
            s1 = float(matches[0])
            s2 = float(matches[1])
            s1_str = str(int(s1)) if s1.is_integer() else str(s1)
            s2_str = str(int(s2)) if s2.is_integer() else str(s2)
            return f"{s1_str} {s2_str}"
        except:
            pass
    return "0 0"


def load_completed_ids(checkpoint_file):
    """从输出文件加载已完成的 id（字符串形式的数字）"""
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
                            completed.add(item['id'])  # id 是字符串如 "123"
                    except:
                        continue
    except Exception as e:
        print(f"Warning: failed to load checkpoint: {e}")
    print(f"Resuming: {len(completed)} items already scored.")
    return completed


def process_item(item_and_index):
    item, idx_str = item_and_index
    try:
        question = item.get("prompt", "").strip()
        chosen = item.get("chosen", "").strip()
        rejected = item.get("rejected", "").strip()

        prompt = SCORING_PROMPT_TEMPLATE.format(
            question=question,
            answer_1=chosen,
            answer_2=rejected
        )

        response_text = query_model(prompt, model_name=JUDGE_MODEL)
        scores_str = parse_scores_to_string(response_text)

        return {
            "id": idx_str,  # 字符串形式的索引，如 "0", "1", ...
            "question": question,
            "assistant_1_response": chosen,
            "assistant_2_response": rejected,
            "scores": scores_str
        }

    except Exception as e:
        print(f"\nError processing item {idx_str}: {str(e)[:100]}")
        return None


def main():
    # Step 1: Load input JSON array
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
    print(f"Loaded {len(raw_data)} items from {INPUT_FILE}")

    # Step 2: Assign id = str(index)
    data_with_id = [(item, str(i)) for i, item in enumerate(raw_data)]

    # Step 3: Load completed ids
    completed_ids = load_completed_ids(CHECKPOINT_FILE)

    # Step 4: Filter unprocessed
    work_items = [
        (item, idx_str) for item, idx_str in data_with_id
        if idx_str not in completed_ids
    ]
    print(f"Remaining: {len(work_items)} items to process")

    if not work_items:
        print("All done!")
        return

    # Step 5: Multi-threaded scoring
    def worker(pair):
        return process_item(pair)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(worker, pair) for pair in work_items]

        pbar = tqdm(total=len(work_items), desc="Scoring via API")
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()
                except Exception as e:
                    print(f"\n⚠️ Thread error")
                pbar.update(1)
        pbar.close()

    print(f"\nFinished! Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()