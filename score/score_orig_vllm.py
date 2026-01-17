#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
import os
import uuid
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# --- 配置 ---
DATA_PATH = "data/skywork_10k.jsonl"   # 输入文件路径
OUTPUT_FILE = "RewardTrainer/data/train/skywork_10k_orig_qwen.json" # 输出文件路径
TEACHER_MODEL_PATH = "models/Qwen3-14B" # 裁判模型路径

# 评估 Prompt 模板
EVAL_PROMPT_TEMPLATE = """[Question]
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


def load_data(file_path):
    """加载数据，返回列表"""
    data_list = []
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return []

    print(f"正在读取 {file_path} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line.strip())
                # 简单校验必要字段
                if all(k in item for k in ['question', 'chosen', 'rejected']):
                    data_list.append(item)
            except json.JSONDecodeError:
                continue
    
    print(f"成功加载 {len(data_list)} 条数据")
    return data_list


def format_eval_prompt(question, answer_1, answer_2):
    """格式化 Prompt"""
    return EVAL_PROMPT_TEMPLATE.format(question=question, answer_1=answer_1, answer_2=answer_2)


def extract_scores(response_text):
    """
    从模型回复中解析分数。
    假设回复格式类似 "8 2" 或 "9.5 6.0"
    """
    if not response_text:
        return 0.0, 0.0
    
    # 正则匹配浮点数或整数
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", response_text)
    
    if len(matches) >= 2:
        try:
            # 取前两个数字，分别对应 Assistant 1 (Chosen) 和 Assistant 2 (Rejected)
            s1 = float(matches[0])
            s2 = float(matches[1])
            return s1, s2
        except ValueError:
            return 0.0, 0.0
    return 0.0, 0.0


def evaluate_dataset(teacher_model_path, data_list):
    """使用 vLLM 批量评估并生成最终 JSON"""
    
    # 1. 初始化模型
    print(f"正在加载教师模型: {teacher_model_path} ...")
    llm = LLM(model=teacher_model_path, 
              trust_remote_code=True, 
              max_model_len=8192,
              gpu_memory_utilization=0.9,
              tensor_parallel_size=1)

    tokenizer = AutoTokenizer.from_pretrained(teacher_model_path, trust_remote_code=True)

    # 2. 采样参数 (Temperature=0 保证确定性)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=20)

    # 3. 构建 Prompts
    prompts_payload = []
    
    # 用于保留原始数据引用的列表，确保顺序一致
    valid_items = [] 

    print("正在构建 Prompts ...")
    for item in tqdm(data_list):
        question = item['question']
        answer_1 = item['chosen']   # Assistant 1 对应 Chosen
        answer_2 = item['rejected'][0] # Assistant 2 对应 Rejected

        user_content = format_eval_prompt(question, answer_1, answer_2)
        
        messages = [
            {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
            {"role": "user", "content": user_content}
        ]

        # 处理 Chat Template
        try:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
        prompts_payload.append(formatted_prompt)
        valid_items.append(item)

    # 4. 批量推理 (vLLM 核心步骤)
    print(f"开始批量推理 {len(prompts_payload)} 条数据 ...")
    outputs = llm.generate(prompts_payload, sampling_params)

    # 5. 整理结果
    final_results = []
    
    print("正在解析分数并整理结果 ...")
    # zip 确保原始数据和模型输出一一对应 
    for original_item, output in zip(valid_items, outputs):
        generated_text = output.outputs[0].text
        
        # 解析分数
        score1, score2 = extract_scores(generated_text)
        
        # 构造目标格式
        result_entry = {
            "id": str(uuid.uuid4()),  # 生成 UUID
            "prompt": original_item['question'],
            "chosen": original_item['chosen'],
            "rejected": original_item['rejected'][0],
            "chosen_score": score1,   # 对应 Prompt 中的 Assistant 1
            "rejected_score": score2  # 对应 Prompt 中的 Assistant 2
        }
        final_results.append(result_entry)

    # 6. 保存为单个大 JSON 文件
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    print(f"正在保存 {len(final_results)} 条结果到 {OUTPUT_FILE} ...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print("🎉 评估完成！文件已保存。")


def main():
    if not os.path.exists(DATA_PATH):
        print(f"请确认数据文件路径正确: {DATA_PATH}")
        return

    data = load_data(DATA_PATH)
    if not data:
        return

    evaluate_dataset(TEACHER_MODEL_PATH, data)


if __name__ == "__main__":
    main()