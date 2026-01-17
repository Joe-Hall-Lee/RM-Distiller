import json
import uuid
import re
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from api import query_model
import os

# --- 配置 ---
INPUT_FILE = "data/skywork_10k.jsonl"  # 输入文件名
OUTPUT_FILE = "RewardTrainer/data/train/skywork_10k_orig_gpt.json"
JUDGE_MODEL = "gpt-4o"
MAX_WORKERS = 30  # 并发线程数

# --- 打分 Prompt 模板 ---
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


def parse_scores(response_text):
    """解析分数的函数保持不变"""
    if not response_text:
        return 0.0, 0.0
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", response_text)
    if len(matches) >= 2:
        try:
            s1 = float(matches[0])
            s2 = float(matches[1])
            return s1, s2
        except ValueError:
            return 0.0, 0.0
    else:
        return 0.0, 0.0


def process_single_line(index, line):
    """
    处理单行数据的 Worker 函数。
    这个函数将在独立的线程中运行。
    返回: (index, result_dict) 或 None
    """
    if not line.strip():
        return None

    try:
        item = json.loads(line)
        question = item.get("question", "")
        chosen = item.get("chosen", "")
        # rejected 是一个列表
        rejected = item.get("rejected", "")
        if isinstance(rejected, list) and len(rejected) > 0:
            rejected = rejected[0]
        elif not isinstance(rejected, str):
            # 如果不是列表也不是字符串，跳过
            return None

        prompt = SCORING_PROMPT_TEMPLATE.format(
            question=question, answer_1=chosen, answer_2=rejected
        )

        # 调用 API
        response = query_model(prompt, model_name=JUDGE_MODEL)

        # 解析分数
        score1, score2 = parse_scores(response)

        # 返回 index 和结果字典
        return index, {
            "id": str(uuid.uuid4()),
            "prompt": question,
            "chosen": chosen,
            "rejected": rejected,
            "chosen_score": score1,
            "rejected_score": score2,
        }

    except json.JSONDecodeError:
        # 记录错误并返回 None
        return None
    except Exception as e:
        # 在多线程中捕获异常，防止一个错误中断整个程序
        print(f"\nError processing line {index}: {e}")
        return None


def main():
    # 读取 jsonl 文件
    print(f"正在读取 {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()

    print(f"开始处理 {len(lines)} 条数据，并发数: {MAX_WORKERS}...")

    # 用于存储 (index, result) 元组
    indexed_results = []

    # 使用 ThreadPoolExecutor 创建线程池
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        # 提交所有任务，并将原始索引一起传入
        futures = {
            executor.submit(process_single_line, i, line): i
            for i, line in enumerate(lines)
        }

        # 使用 as_completed 实时获取结果和进度条
        for future in tqdm(as_completed(futures), total=len(lines), desc="Scoring"):
            result = future.result()
            if result:
                # result 是一个元组 (index, result_dict)
                indexed_results.append(result)

    # 关键步骤：根据索引进行排序
    print("处理完成，正在按原始顺序排序结果...")

    # 按照元组的第一个元素（即 index）进行排序
    indexed_results.sort(key=lambda x: x[0])

    # 提取排序后的结果字典
    final_results = [item[1] for item in indexed_results]

    # 保存结果
    print(f"正在保存 {len(final_results)} 条排序后的结果到 {OUTPUT_FILE}...")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        # 使用 indent=4 使输出易读
        json.dump(final_results, f, ensure_ascii=False, indent=4)

    print("完成！")


if __name__ == "__main__":
    main()
