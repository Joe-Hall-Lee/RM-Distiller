#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from api import query_model

# --- 配置 ---
DATA_PATH = "data/skywork_10k.jsonl"
OUTPUT_DIR = "responses"

MODEL_NAMES = [
    "gpt-4o",
]

NUM_THREADS = 32


def load_prompts(file_path):
    prompts = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "question" in obj:
                    prompts.append(obj["question"])
                else:
                    prompts.append(None)  # 保留占位
            except json.JSONDecodeError:
                prompts.append(None)
    print(f"成功加载 {len(prompts)} 行")
    return prompts


def load_completed_ids(output_file):
    """从输出文件加载已完成的 id（字符串形式）"""
    completed = set()
    if not os.path.exists(output_file):
        return completed
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
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
        print(f"⚠️ 警告：加载 checkpoint 失败: {e}")
    print(f"✅ 已完成: {len(completed)} 条")
    return completed


def generate_single(prompt_with_id, model_name):
    """处理单条：输入 (id_str, prompt)，输出 (id_str, response)"""
    idx_str, prompt = prompt_with_id
    if prompt is None:
        return idx_str, "[NO_QUESTION]"
    try:
        resp = query_model(prompt, model_name=model_name)
        if isinstance(resp, str):
            resp = resp.strip()
        return idx_str, resp
    except Exception as e:
        return idx_str, f"ERROR: {str(e)[:200]}"


def generate_responses_for_model(model_name, prompts):
    print(f"\n=== 开始生成：{model_name} ===")
    
    # 构建带 id 的任务列表
    tasks = [(str(i), prompt) for i, prompt in enumerate(prompts)]
    
    # 输出文件路径
    output_file = os.path.join(
        OUTPUT_DIR, f"responses_{model_name.replace('/', '_')}.jsonl"
    )
    
    # 加载已完成的 id
    completed_ids = load_completed_ids(output_file)
    
    # 过滤未完成的任务
    remaining_tasks = [task for task in tasks if task[0] not in completed_ids]
    print(f"总任务: {len(tasks)}, 剩余: {len(remaining_tasks)}")

    if not remaining_tasks:
        print("该模型已全部完成！")
        return

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 多线程 + 边跑边存
    def worker(task):
        return generate_single(task, model_name)

    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        futures = {executor.submit(worker, task): task[0] for task in remaining_tasks}

        pbar = tqdm(total=len(remaining_tasks), desc=f"{model_name}")
        with open(output_file, 'a', encoding='utf-8') as f_out:
            for future in as_completed(futures):
                try:
                    idx_str, response = future.result()
                    # 构造输出项
                    entry = {
                        "id": idx_str,               # 原始索引
                        "question": tasks[int(idx_str)][1] or "[MISSING]",
                        "model": model_name,
                        "response": response
                    }
                    f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    f_out.flush()  # 立即落盘
                except Exception as e:
                    idx_str = futures[future]
                    print(f"\n⚠️ 线程异常 (id={idx_str}): {e}")
                pbar.update(1)
        pbar.close()

    print(f"✅ 生成完成: {output_file}")


def main():
    prompts = load_prompts(DATA_PATH)
    for model_name in MODEL_NAMES:
        generate_responses_for_model(model_name, prompts)


if __name__ == "__main__":
    main()