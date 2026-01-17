#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

# 输入文件路径
DATA_PATH = "data/finance_10k.jsonl"
# 输出目录
OUTPUT_DIR = "responses"
# 候选模型列表
MODEL_PATHS = {
    # "llama-3.1-8b-instruct": "models/Llama-3.1-8B-Instruct",
    # "mistral-7b-instruct-v0.3": "models/Mistral-7B-Instruct-v0.3",
    # "gemma-2-9b-it": "models/gemma-2-9b-it",
    # "vicuna-7b-v1.5": "models/vicuna-7b-v1.5",
    "qwen3-14b": "models/Qwen3-14B",
    # "qwen2.5-3b-instruct": "models/Qwen2.5-3B-Instruct",
}


def load_prompts(file_path):
    """从 JSONL 文件加载 prompts"""
    prompts = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'question' in data:
                        prompts.append(data['question'])
                    else:
                        print(f"警告：跳过无效行，缺少 'question' 字段：{line.strip()}")
                except json.JSONDecodeError as e:
                    print(f"错误：无法解析 JSON 行：{line.strip()}，错误：{e}")
        print(f"成功加载 {len(prompts)} 个 prompts")
        return prompts
    except Exception as e:
        print(f"错误：无法读取文件 {file_path}：{e}")
        return []


def format_and_truncate_prompt(
    prompt: str,
    tokenizer,
    max_model_len: int,
    max_new_tokens: int,
):
    """
    先 apply_chat_template，再按 token 级别截断
    """
    messages = [{"role": "user", "content": prompt}]

    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # 编码成 token
    input_ids = tokenizer(
        formatted,
        add_special_tokens=False,
        return_attention_mask=False,
        return_tensors=None,
    )["input_ids"]

    # 给生成预留空间
    max_prompt_len = max_model_len - max_new_tokens
    if max_prompt_len <= 0:
        raise ValueError("max_model_len 太小，无法生成")

    # 超长则从「左侧」截断（保留问题结尾，金融问答更合理）
    if len(input_ids) > max_prompt_len:
        input_ids = input_ids[-max_prompt_len:]

    # 再 decode 回字符串
    truncated_prompt = tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )

    return truncated_prompt


def generate_responses(model_path, model_name, prompts):
    """使用 vLLM 推理生成 responses"""
    try:
        # 加载模型和 tokenizer
        print(f"加载模型：{model_name}")

        MAX_MODEL_LEN = 8192

        MAX_NEW_TOKENS = 1024
        llm = LLM(model=model_path, trust_remote_code=True,
                  max_model_len=MAX_MODEL_LEN)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True)

        formatted_prompts = [
            format_and_truncate_prompt(
                prompt,
                tokenizer,
                max_model_len=MAX_MODEL_LEN,
                max_new_tokens=MAX_NEW_TOKENS,
            )
            for prompt in prompts
        ]

        # 批量推理
        print(f"开始为 {model_name} 生成 responses……")
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            max_tokens=MAX_NEW_TOKENS,
        )
        outputs = llm.generate(formatted_prompts, sampling_params)

        # 收集结果
        responses = []
        for i, output in enumerate(tqdm(outputs, desc=f"处理 {model_name} 输出")):
            response = output.outputs[0].text.strip()
            responses.append({
                "question": prompts[i],
                "model": model_name,
                "response": response
            })

        # 保存结果
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_file = os.path.join(
            OUTPUT_DIR, f"responses_{model_name.replace('/', '_')}.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for response in responses:
                f.write(json.dumps(response, ensure_ascii=False) + '\n')
        print(f"已保存 {model_name} 的 responses 到 {output_file}")

        return responses

    except Exception as e:
        print(f"错误：{model_name} 推理失败：{e}")
        return []


def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 加载 prompts
    prompts = load_prompts(DATA_PATH)
    if not prompts:
        print("错误：没有加载到任何 prompts，退出")
        return

    # 为每个模型生成 responses
    for model_name, model_path in MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"警告：模型路径 {model_path} 不存在，跳过 {model_name}")
            continue
        generate_responses(model_path, model_name, prompts)


if __name__ == "__main__":
    main()
