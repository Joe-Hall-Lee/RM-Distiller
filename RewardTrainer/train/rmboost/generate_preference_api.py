#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import random
import argparse
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from api import query_model

# =============================
# 配置
# =============================

RMBOOST_PROMPT_TEMPLATE = """Your task is to generate another response by EDITING the given response,
so that the new response is {preference} than the given response
with respect to some evaluation aspects.

<task_description>
Below you will first see a guideline with detailed evaluation aspects of responses.
Then, you will be presented with the question and the given response.

You should complete the following steps internally:
Step 1: Select a few aspects from the guideline.
Step 2: Edit the given response so that it becomes {preference}
with respect to the selected aspects.
</task_description>

<guideline>
A high-quality response should:
- Directly and correctly answer the question.
- Be clear, coherent, and logically well-structured.
- Be factually and logically consistent.
- Include all necessary reasoning steps or explanations.

We evaluate responses from the following aspects:
- (Honesty): The assistant should be honest about whether it knows the answer and express its uncertainty explicitly. Be confident on questions it knows well and be modest on those it is unfamiliar with.
- (Truthfulness): The assistant should answer truthfully and be faithful to factual knowledge as well as given contexts, never making up any new facts that are not true or cannot be grounded in the instruction.
- (Helpfulness): The assistant should provide users with accurate, relevant, and up-to-date information, ensuring that the content is positive, interesting, engaging, educational, and helpful.
- (Relevance and Coherence): Whether the response is focused on the question, logically organized, and free of irrelevant or confusing content.
- (Correctness and Faithfulness): Whether the response contains correct reasoning and does not include factual or logical errors.
- (Completeness): Whether the response includes all necessary reasoning steps needed to fully answer the question.
- (Safety and Ethics): Whether the response adheres to ethical guidelines, avoids harmful or biased content, and respects user privacy and dignity.
</guideline>

Below is the question.
<question>
{question}
</question>

Below is the given response.
<given_response>
{response}
</given_response>

Read the question and the given response carefully.
Review the task description and the guideline.

Generate a new response by editing the given response.
The generated response should still look like a plausible model-generated answer.
Do NOT mention the guideline or the fact that you are editing a response.

Do NOT summarize at the end. Do not mention your role. Just output the rewritten response.""".strip()


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_completed_ids(checkpoint_file: str) -> set:
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


def process_single_item(item_with_label_id, model_name: str):
    idx, sample, label = item_with_label_id
    try:
        # 构建 prompt
        prompt = RMBOOST_PROMPT_TEMPLATE.format(
            preference=label,
            question=sample["question"],
            response=sample["response"]
        )
        # 调用 API
        rewritten = query_model(prompt, model_name=model_name).strip()

        original = sample["response"]
        if label == "BETTER":
            chosen = rewritten
            rejected = original
        else:
            chosen = original
            rejected = rewritten

        result = {
            "id": sample.get("id", str(idx)),  # 优先用原始 id，否则用索引
            "prompt": sample["question"],
            "chosen": chosen,
            "rejected": rejected,
        }
        return sample.get("id", str(idx)), result
    except Exception as e:
        error_resp = f"[ERROR: {str(e)[:150]}]"
        result = {
            "id": sample.get("id", str(idx)),
            "prompt": sample["question"],
            "chosen": error_resp,
            "rejected": sample["response"],
        }
        return sample.get("id", str(idx)), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str,
                        default="responses/responses_gpt-4o.jsonl")
    parser.add_argument("--output_json", type=str,
                        default="RewardTrainer/data/train/skywork_10k_rmboost_gpt.json")
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--num_threads", type=int, default=30)
    parser.add_argument("--random_label", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    # Load data
    data = load_jsonl(args.input_jsonl)
    n = len(data)
    print(f"✅ Loaded {n} samples from {args.input_jsonl}")

    # Assign labels
    if args.random_label:
        labels = [random.choice(["BETTER", "WORSE"]) for _ in range(n)]
    else:
        labels = ["BETTER"] * (n // 2) + ["WORSE"] * (n - n // 2)

    # Prepare work items: (index, sample, label)
    work_items = [(i, data[i], labels[i]) for i in range(n)]

    # Checkpoint file (use .jsonl for resume)
    checkpoint_file = args.output_json.replace('.json', '.jsonl')
    completed_ids = load_completed_ids(checkpoint_file)

    # Filter unprocessed (compare by 'id' if exists, else by index string)
    remaining_items = []
    for idx, sample, label in work_items:
        item_id = sample.get("id", str(idx))
        if item_id not in completed_ids:
            remaining_items.append((idx, sample, label))

    print(f"🔄 Total: {len(work_items)}, Remaining: {len(remaining_items)}")
    if not remaining_items:
        print("🎉 All done!")
        rebuild_final_output(checkpoint_file, args.output_json)
        return

    # Multi-threaded processing
    def worker(item):
        return process_single_item(item, args.model_name)

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = {executor.submit(worker, item)
                                   : item[0] for item in remaining_items}

        pbar = tqdm(total=len(remaining_items), desc="RMBoost rewriting")
        with open(checkpoint_file, 'a', encoding='utf-8') as f_out:
            for future in as_completed(futures):
                try:
                    item_id, result = future.result()
                    f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                    f_out.flush()
                except Exception as e:
                    print(f"\n⚠️ Thread error: {e}")
                pbar.update(1)
        pbar.close()

    # Rebuild final JSON array
    rebuild_final_output(checkpoint_file, args.output_json)


def rebuild_final_output(checkpoint_file: str, final_output: str):
    """从 checkpoint .jsonl 重建有序 JSON 数组"""
    if not os.path.exists(checkpoint_file):
        print("❌ Checkpoint not found.")
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

    # 按 id 排序（假设 id 是字符串数字）
    try:
        sorted_items = sorted(items_by_id.values(), key=lambda x: int(x['id']))
    except:
        # 如果 id 不是数字，按原始插入顺序（但通常应是数字）
        sorted_items = list(items_by_id.values())

    # Remove internal fields (e.g., __label)
    final_items = []
    for item in sorted_items:
        clean_item = {k: v for k, v in item.items() if not k.startswith("__")}
        final_items.append(clean_item)

    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(final_items, f, ensure_ascii=False, indent=2)

    print(
        f"✅ Final RMBoost data saved to {final_output} ({len(final_items)} items)")


if __name__ == "__main__":
    main()
