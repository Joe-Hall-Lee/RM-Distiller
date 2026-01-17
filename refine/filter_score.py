import json
import os

# --- 配置 ---
INPUT_FILE = "RewardTrainer\data/train/skywork_10k_gpt-4o_bridge.json"


def analyze_and_filter_scores(file_path):
    """
    统计 chosen_score 与 rejected_score 的关系，将过滤后的数据保存到新文件中。
    """
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    print(f"正在读取文件: {file_path}")

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print("错误：JSON 文件格式不正确，无法加载。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    total_count = len(data)
    reverse_or_tie_count = 0
    strictly_reversed_count = 0
    tie_count = 0

    filtered_data = []

    if total_count == 0:
        print("文件不包含任何数据。")
        return

    for item in data:
        try:
            chosen_score = item["chosen_score"]
            rejected_score = item["rejected_score"]

            # 判断 reversed / tie
            if chosen_score - rejected_score <= 1 or item["chosen"] == item["rejected"]:
                reverse_or_tie_count += 1

                if chosen_score < rejected_score:
                    strictly_reversed_count += 1
                else:
                    tie_count += 1
            else:
                # 只保留正确关系的数据
                filtered_data.append(item)

        except KeyError:
            print("警告：部分数据缺少字段，已跳过该项。")
            continue
        except TypeError:
            print("警告：得分字段类型错误，已跳过该项。")
            continue

    # --- 统计结果 ---
    reverse_or_tie_ratio = (reverse_or_tie_count / total_count) * 100
    judge_accuracy = ((total_count - reverse_or_tie_count) / total_count) * 100

    print("-" * 50)
    print(f"分析结果:")
    print(f"总样本数 (Total Samples): {total_count}")
    print(f"1. 逆转或平局样本数: {reverse_or_tie_count}")
    print(f"   - 逆转数: {strictly_reversed_count}")
    print(f"   - 平局数: {tie_count}")
    print(f"2. 逆转或平局比例: {reverse_or_tie_ratio:.2f}%")
    print(f"3. 裁判模型准确率 (Chosen > Rejected): {judge_accuracy:.2f}%")
    print("-" * 50)

    # --- 保存过滤后的数据 ---
    output_file = file_path.replace(".json", "_filtered_.json")

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json_string = json.dumps(
                filtered_data,
                indent=2
            )

            f.write(json_string)


        print(f"过滤后数据已保存到: {output_file}")
        print(f"过滤后样本数: {len(filtered_data)}")
        print(f"被过滤掉的样本数: {total_count - len(filtered_data)}")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")


if __name__ == "__main__":
    analyze_and_filter_scores(INPUT_FILE)
