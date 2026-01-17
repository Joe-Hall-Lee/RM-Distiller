#!/usr/bin/env python
# -*- coding: utf-8 -*-

import re
import json
import os
import random
from typing import List, Dict

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm


# =============================
# 配置
# =============================

INPUT_JSONL = "responses/responses_qwen3-14b.jsonl"
OUTPUT_JSON = "RewardTrainer/data/train/skywork_10k_rmboost_qwen.json"

MODEL_NAME = "qwen3-14b"
MODEL_PATH = "models/Qwen3-14B"

MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 1024

RANDOM_LABEL = True
SEED = 42

# =============================
# RMBoost 通用 rewrite prompt
# =============================

RMBOOST_PROMPT_TEMPLATE = """Your task is to generate another response by EDITING the given response, so that the new response is {preference} than the given response with respect to some evaluation aspects.

<task_description>
Below you will first see a guideline with detailed evaluation aspects of responses.
Then, you will be presented with the question and the given response.

You should complete the following steps internally:
Step 1: Select a few aspects from the guideline.
Step 2: Generate another response that is {preference} than the given response in terms of above selected aspects.
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

Read the question and given response carefully. Review the above task description and guideline. Think about how to accomplish the task step by step before you reply. Put your generated response in <response></response> tags."""


# =============================
# 工具函数
# =============================


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def build_prompt(sample: Dict, preference: str, tokenizer) -> str:
    raw_prompt = RMBOOST_PROMPT_TEMPLATE.format(
        preference=preference, question=sample["question"], response=sample["response"]
    )

    messages = [{"role": "user", "content": raw_prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def extract_response(text: str) -> str:
    """
    Robustly extract model-generated response.

    Priority:
    1. Content inside <response>...</response>
    2. Content after <response> if closing tag is missing
    3. Fallback to full text
    """
    if not text:
        return ""

    text = text.strip()

    # Case 1: well-formed <response>...</response>
    pattern_full = re.compile(
        r"<response>(.*?)</response>", re.DOTALL | re.IGNORECASE)
    match = pattern_full.search(text)
    if match:
        return match.group(1).strip()

    # Case 2: only opening <response>
    pattern_open = re.compile(r"<response>(.*)", re.DOTALL | re.IGNORECASE)
    match = pattern_open.search(text)
    if match:
        return match.group(1).strip()

    # Case 3: fallback
    return text


# =============================
# 主流程
# =============================


def main():
    random.seed(SEED)

    # 加载数据
    data = load_jsonl(INPUT_JSONL)
    n = len(data)
    print(f"Loaded {n} samples")

    # 构造偏好标签
    if RANDOM_LABEL:
        labels = [random.choice(["BETTER", "WORSE"]) for _ in range(n)]
    else:
        labels = ["BETTER"] * (n // 2) + ["WORSE"] * (n - n // 2)

    # 加载模型
    llm = LLM(
        model=MODEL_PATH,
        trust_remote_code=True,
        max_model_len=MAX_MODEL_LEN,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, trust_remote_code=True)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_tokens=MAX_NEW_TOKENS,
    )

    # 构造 prompts
    prompts = [
        build_prompt(sample, label, tokenizer) for sample, label in zip(data, labels)
    ]

    # 推理
    print("Generating RMBoost rewritten responses...")
    outputs = llm.generate(prompts, sampling_params)

    # 构造 preference JSON
    pref_data = []

    for idx, (sample, label, output) in enumerate(
        tqdm(zip(data, labels, outputs), total=n)
    ):
        raw_output = output.outputs[0].text
        rewritten = extract_response(raw_output)
        original = sample["response"]

        if label == "BETTER":
            chosen = rewritten
            rejected = original
        else:
            chosen = original
            rejected = rewritten

        pref_data.append(
            {
                "id": str(idx),
                "prompt": sample["question"],
                "chosen": chosen,
                "rejected": rejected,
            }
        )

    # 保存为 JSON（数组）
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(pref_data, f, ensure_ascii=False, indent=2)

    print(f"Saved RMBoost preference data to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
