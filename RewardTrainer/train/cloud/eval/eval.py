import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from train.cloud.model import CLoudRewardModel
from train.cloud.inference.api import CLoudAPI


REWARD_BENCH_TO_CATEGORY_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Math": [  # 单独列出数学
        "math-prm",
    ],
    "Code": [  # 单独列出代码
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

###########
# Build eval data
###########

def load_reward_bench(json_path):
    """Loads reward bench data from a JSONL file."""
    data = load_dataset("json", data_files=json_path)["train"]
    eval_data = []
    eval_metadata = []
    for example in data:
        eval_data.append({
            "id": f"{example['id']}-chosen",
            "prompt": example["prompt"],
            "response": example["chosen"]
        })
        eval_data.append({
            "id": f"{example['id']}-rejected",
            "prompt": example["prompt"],
            "response": example["rejected"]
        })
        eval_metadata.append({
            "id": str(example["id"]),
            "subset": example["subset"]
        })
    return eval_data, eval_metadata

###########
# Post-process Scores
###########

def post_process_reward_bench(eval_metadata, rewards):
    # 初始化所有类别（包括 Math 和 Code）
    per_category_scores = {
        "Chat": [],
        "Chat Hard": [],
        "Safety": [],
        "Math": [],
        "Code": []
    }
    
    # 收集各子类别的得分
    for example in eval_metadata:
        id_ = example["id"]
        chosen_reward = rewards[id_ + "-chosen"]
        rejected_reward = rewards[id_ + "-rejected"]
        
        # 检查属于哪个类别
        for category, subsets in REWARD_BENCH_TO_CATEGORY_MAPPING.items():
            if example["subset"] in subsets:
                per_category_scores[category].append(int(chosen_reward > rejected_reward))
                break
    
    # 计算各子类别的准确率
    per_category_scores = {
        category: np.mean(scores) * 100 if scores else 0.0
        for category, scores in per_category_scores.items()
    }
    
    # 计算 Reasoning 类别得分（Math 和 Code 的平均）
    per_category_scores["Reasoning"] = (per_category_scores["Math"] + per_category_scores["Code"]) / 2
    
    # 计算总平均（Chat, Chat Hard, Safety, Reasoning）
    valid_categories = ["Chat", "Chat Hard", "Safety", "Reasoning"]
    per_category_scores["Average"] = np.mean([per_category_scores[cat] for cat in valid_categories])

    # 打印结果
    print("\nReward Bench Scores:")
    print("=" * 40)
    max_category_length = max(len(category) for category in per_category_scores.keys())
    for category in ["Chat", "Chat Hard", "Safety", "Math", "Code", "Reasoning", "Average"]:
        print(f"{category:<{max_category_length}} : {per_category_scores[category]:.2f}%")
    print("=" * 40)

    return per_category_scores

###########
# Scoring
###########

def generate_rewards_hf(model, tokenizer, eval_data, batch_size):
    rewards = {}

    for i in tqdm(range(0, len(eval_data), batch_size)):
        batch = eval_data[i:i+batch_size]

        prompts = [item["prompt"] for item in batch]
        responses = [item["response"] for item in batch]
        ids = [item["id"] for item in batch]

        batch_rewards, _ = model.predict_reward(prompts, responses, tokenizer)

        for id_, reward in zip(ids, batch_rewards):
            rewards[id_] = reward

    return rewards

def generate_rewards_vllm(client, eval_data, num_workers):
    rewards = {}

    def fetch_reward(example):
        critique, reward = client.get_reward(
            example["prompt"],
            example["response"],
        )
        return critique, reward, example["id"]

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(fetch_reward, example): example["id"] for example in eval_data}
        for future in tqdm(as_completed(futures), total=len(eval_data), desc="Generating rewards"):
            critique, reward, id_ = future.result()
            rewards[id_] = reward

    return rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--benchmark_path", type=str, help="Path to the JSONL file containing the reward bench data.")

    # Vllm args
    parser.add_argument("--inference-method", type=str, default="hf", choices=["hf", "vllm"])
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--hosted", action="store_true")

    # HF args
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    eval_data = []
    eval_metadata = []

    if args.benchmark_path:
        eval_data, eval_metadata = load_reward_bench(args.benchmark_path)
    else:
        raise ValueError("benchmark_path must be provided")

    if args.inference_method == "hf":
        if CLoudRewardModel is None or AutoTokenizer is None:
            raise ImportError("CLoudRewardModel and AutoTokenizer are required for HF inference.")
        model = CLoudRewardModel.from_pretrained(args.model_path, device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
        rewards = generate_rewards_hf(model, tokenizer, eval_data, batch_size=args.batch_size)
    elif args.inference_method == "vllm":
        if CLoudAPI is None:
            raise ImportError("CLoudAPI is required for vllm inference.")
        client = CLoudAPI(model=args.model_path, tensor_parallel_size=args.tensor_parallel_size, hosted=args.hosted)
        rewards = generate_rewards_vllm(client, eval_data, num_workers=args.num_workers)

    if eval_metadata and rewards:
        post_process_reward_bench(eval_metadata, rewards)
    else:
        print("Warning: No evaluation data or rewards generated.")