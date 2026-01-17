#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from api import query_model

# --- 配置 ---
ORIGINAL_INPUT_FILE = r"F:/CS\AI\DistilRM/RewardTrainer\data/train\skywork_10k_qwen.json"
SCORED_FILE = r"evaluations/evaluations_gemini.jsonl"
OUTPUT_REGRADED_FILE = r"evaluations/skywork_10k_gemini_regraded_ties.jsonl"  # checkpoint = output
JUDGE_MODEL = "gemini-2.5-pro-nothinking"
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
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance. Try not to give the same score unless the quality of the two responses is exactly the same
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


def is_tie(scores_str):
    parts = scores_str.strip().split()
    if len(parts) == 2:
        try:
            return abs(float(parts[0]) - float(parts[1])) < 1e-5
        except:
            return False
    return False


def load_original_data_by_id():
    with open(ORIGINAL_INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {str(i): data[i] for i in range(len(data))}


def load_tie_items_with_original():
    """加载所有平局项，并附带原始 prompt/chosen/rejected"""
    if not os.path.exists(SCORED_FILE):
        raise FileNotFoundError(f"Scored file not found: {SCORED_FILE}")

    original_map = load_original_data_by_id()
    tie_items = []

    with open(SCORED_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                if 'id' in item and 'scores' in item and is_tie(item['scores']):
                    orig = original_map.get(item['id'])
                    if orig:
                        tie_items.append({
                            'id': item['id'],
                            'prompt': orig['prompt'],
                            'chosen': orig['chosen'],
                            'rejected': orig['rejected']
                        })
            except Exception as e:
                continue
    print(f"✅ Found {len(tie_items)} tie items to regrade.")
    return tie_items


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
    print(f"✅ Resuming: {len(completed)} tie items already regraded.")
    return completed


def process_single_tie(tie_item):
    try:
        prompt = tie_item['prompt'].strip()
        chosen = tie_item['chosen'].strip()
        rejected = tie_item['rejected'].strip()

        if not prompt or not chosen or not rejected:
            return None

        full_prompt = SCORING_PROMPT_TEMPLATE.format(
            question=prompt,
            answer_1=chosen,
            answer_2=rejected
        )

        response_text = query_model(full_prompt, model_name=JUDGE_MODEL)
        scores_str = parse_scores_to_string(response_text)

        return {
            "id": tie_item["id"],
            "question": prompt,
            "assistant_1_response": chosen,
            "assistant_2_response": rejected,
            "scores": scores_str
        }

    except Exception as e:
        print(f"\n❌ Error regrading item {tie_item['id']}: {str(e)[:100]}")
        return None


def main():
    # Step 1: Load all tie items with full context
    tie_items = load_tie_items_with_original()
    if not tie_items:
        print("🎉 No tie items found.")
        return

    # Step 2: Load already regraded ids
    completed_ids = load_completed_ids(OUTPUT_REGRADED_FILE)

    # Step 3: Filter unprocessed
    work_items = [item for item in tie_items if item['id'] not in completed_ids]
    print(f"🔄 Remaining: {len(work_items)} tie items to process")

    if not work_items:
        print("✅ All tie items already regraded!")
        return

    # Step 4: Multi-threaded + append write
    def worker(item):
        return process_single_tie(item)

    os.makedirs(os.path.dirname(OUTPUT_REGRADED_FILE), exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker, item): item['id'] for item in work_items}

        pbar = tqdm(total=len(work_items), desc="Regrading ties via API")
        with open(OUTPUT_REGRADED_FILE, 'a', encoding='utf-8') as f_out:
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                        f_out.flush()  # 确保立即落盘
                except Exception as e:
                    item_id = futures[future]
                    print(f"\n⚠️ Exception for item {item_id}: {e}")
                pbar.update(1)
        pbar.close()

    print(f"\n✅ Finished! Regraded results saved to {OUTPUT_REGRADED_FILE}")


if __name__ == "__main__":
    main()