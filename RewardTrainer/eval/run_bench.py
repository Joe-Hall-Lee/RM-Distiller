import json
import os
import argparse

from omegaconf import OmegaConf

from module import InferenceModule, VllmModule, HfModule, OpenaiModule, ApiModule


BENCHMARK_IDS = ["rewardbench", "rm-bench", "judgebench",
                 "rmb_pairwise", "evalbiasbench", "ifbench"]


def make_data_row(id: int, instruction: str, response1: str, response2: str, label: int) -> dict:
    return {
        "id": id,
        "instruction": instruction.strip(),
        "response1": response1.strip(),
        "response2": response2.strip(),
        "label": label,
    }


def format_conversation_for_judge(prompt):
    """
    将 prompt 转换为适合 Judge 评估的格式
    - 如果是字符串，直接返回
    - 如果是列表（多轮对话），格式化为可读的对话文本
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and len(prompt) > 0:

        if len(prompt) == 1 and isinstance(prompt[0], dict):
            return prompt[0].get("content", "")
        # 格式化多轮对话为易读文本
        conversation_parts = []
        for turn in prompt:
            if isinstance(turn, dict):
                role = turn.get("role", "user")
                content = turn.get("content", "")
                # 格式化为 "User: xxx" 或 "Assistant: xxx"
                role_label = "User" if role == "user" else "Assistant"
                conversation_parts.append(f"{role_label}: {content}")

        return "\n\n".join(conversation_parts)

    return ""


def get_benchmark_data(benchmark_id: str, data_path) -> dict:
    """output a standardized dataset. only the contents.
    the data structure will be kept until the results."""
    benchmark_set = {}
    assert benchmark_id in BENCHMARK_IDS
    print("Loading benchmark:", benchmark_id)

    if benchmark_id == "rewardbench":
        SUBSET_MAPPING = {
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
            "Math": ["math-prm"],
            "Code": [
                "hep-cpp",
                "hep-go",
                "hep-java",
                "hep-js",
                "hep-python",
                "hep-rust",
            ]
        }
        dataset = []
        with open(os.path.join(data_path, "rewardbench.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]
        for subset_name in ["Chat", "Chat Hard", "Safety", "Math", "Code"]:
            subset = []
            for i, row in enumerate(dataset):
                if dataset[i]["subset"] in SUBSET_MAPPING[subset_name]:
                    subset.append(make_data_row(
                        i, row["prompt"], row["chosen"], row["rejected"], 1))
            benchmark_set[subset_name] = subset
    elif benchmark_id == "judgebench":
        SUBSET_MAPPING = {
            "Knowledge": ["mmlu-pro-biology",
                          "mmlu-pro-business",
                          "mmlu-pro-chemistry",
                          "mmlu-pro-computer science",
                          "mmlu-pro-economics",
                          "mmlu-pro-engineering",
                          "mmlu-pro-health",
                          "mmlu-pro-history",
                          "mmlu-pro-law",
                          "mmlu-pro-math",
                          "mmlu-pro-other",
                          "mmlu-pro-philosophy",
                          "mmlu-pro-physics",
                          "mmlu-pro-psychology"
                          ],
            "Reasoning": ["livebench-reasoning"],
            "Math": ["livebench-math"],
            "Coding": ["livecodebench"],
        }
        dataset = []
        with open(os.path.join(data_path, "judgebench.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]
        for subset_name in ["Knowledge", "Reasoning", "Math", "Coding"]:
            subset = []
            for i, row in enumerate(dataset):
                if dataset[i]["subset"] in SUBSET_MAPPING[subset_name]:
                    subset.append(make_data_row(
                        i, row["prompt"], row["chosen"], row["rejected"], 1))
            benchmark_set[subset_name] = subset
    elif benchmark_id == "rmb_pairwise":
        SUBSET_MAPPING = {
            "Harmlessness": [
                "Violent Crimes",
                "Non-Violent Crimes",
                "Sex-Related Crimes",
                "Child Sexual Exploitation",
                "Specialized Advice",
                "Privacy",
                "Intellectual Property",
                "Indiscriminate Weapons",
                "Hate",
                "Suicide & Self-Harm",
                "Sexual Content",
                "Multi",
            ],
            "Helpfulness": [
                "Brainstorming",
                "Chat",
                "Classification",
                "Closed QA",
                "Code",
                "Generation",
                "Open QA",
                "Reasoning",
                "Rewrite",
                "Role Playing",
                "Summarization",
                "Translation",
            ],
        }
        dataset = []
        with open(os.path.join(data_path, "rmb_pairwise.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]
        for subset_name in ["Helpfulness", "Harmlessness"]:
            subset = []
            for i, row in enumerate(dataset):
                if dataset[i]["subset"] in SUBSET_MAPPING[subset_name]:
                    # 格式化 prompt（处理多轮对话）
                    formatted_prompt = format_conversation_for_judge(
                        row["prompt"])

                    subset.append(make_data_row(
                        i, formatted_prompt, row["chosen"], row["rejected"], 1))
            benchmark_set[subset_name] = subset
    elif benchmark_id == "rm-bench":
        SUBSET_MAPPING = {
            "Chat": [
                "Chat_Easy",
                "Chat_Normal",
                "Chat_Hard",
            ],
            "Code": [
                "Code_Easy",
                "Code_Normal",
                "Code_Hard",
            ],
            "Math": [
                "Math_Easy",
                "Math_Normal",
                "Math_Hard",
            ],
            "Safety": [
                "Safety_Easy",
                "Safety_Normal",
                "Safety_Hard",
            ],
            "Chat_Easy": ["Chat_Easy"],
            "Chat_Normal": ["Chat_Normal"],
            "Chat_Hard": ["Chat_Hard"],
            "Code_Easy": ["Code_Easy"],
            "Code_Normal": ["Code_Normal"],
            "Code_Hard": ["Code_Hard"],
            "Math_Easy": ["Math_Easy"],
            "Math_Normal": ["Math_Normal"],
            "Math_Hard": ["Math_Hard"],
            "Safety_Easy": ["Safety_Easy"],
            "Safety_Normal": ["Safety_Normal"],
            "Safety_Hard": ["Safety_Hard"],
        }
        dataset = []
        with open(os.path.join(data_path, "rm-bench.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]
        for subset_name in ["Chat", "Code", "Math", "Safety"]:
            subset = []
            for i, row in enumerate(dataset):
                if dataset[i]["subset"] in SUBSET_MAPPING[subset_name]:
                    subset.append(make_data_row(
                        i, row["prompt"], row["chosen"], row["rejected"], 1))
            benchmark_set[subset_name] = subset
    elif benchmark_id == "evalbiasbench":
        SUBSET_MAPPING = {
            "Length": ["length bias"],
            "Concreteness": ["concreteness"],
            "Empty Reference": ["empty reference"],
            "Content Continuation": ["content_continuation"],
            "Nested Instruction": ["nested_instruction"],
            "Familiar Knowledge": ["familiar knowledge preference bias"],
        }
        dataset = []
        with open(os.path.join(data_path, "evalbiasbench.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]
        for subset_name in ["Length", "Concreteness", "Empty Reference", "Content Continuation", "Nested Instruction", "Familiar Knowledge"]:
            subset = []
            for i, row in enumerate(dataset):
                if dataset[i]["subset"] in SUBSET_MAPPING[subset_name]:
                    subset.append(make_data_row(
                        i, row["prompt"], row["chosen"], row["rejected"], 1))
            benchmark_set[subset_name] = subset
    elif benchmark_id == "ifbench":
        SUBSET_MAPPING = {
            "IFBench": ["ifbench"],
        }
        dataset = []
        with open(os.path.join(data_path, "ifbench.jsonl"), "r") as fin:
            lines = [line.strip() for line in fin.readlines()]
            dataset = [json.loads(line) for line in lines]
        for subset_name in ["IFBench"]:
            subset = []
            for i, row in enumerate(dataset):
                if dataset[i]["subset"] in SUBSET_MAPPING[subset_name]:
                    subset.append(make_data_row(
                        i, row["prompt"], row["chosen"], row["rejected"], 1))
            benchmark_set[subset_name] = subset
    else:
        raise ValueError(benchmark_id)

    return benchmark_set


def add_inference(benchmark_data: dict, module: InferenceModule) -> None:
    """all common logic for benchmarking. 
    apply swap, apply prompt template, apply chat template, for all subsets in benchmark data.
    run inference and update on benchmark_data"""
    conversation_list = []

    for subset_name, subset_data in benchmark_data.items():

        for row in subset_data:
            # 如果是 pointwise 模式，则为每个 response 分别生成对话
            if module.pointwise:
                for point_idx in [1, 2]:
                    conversation_list.append(module.make_conversation(
                        row["instruction"], row["response1"], row["response2"], False, point_idx))
            else:
                # pairwise
                for swap in [False, True]:
                    conversation_list.append(module.make_conversation(
                        row["instruction"], row["response1"], row["response2"], swap))

    generated_texts = module.generate(conversation_list)

    index = 0
    for subset_name, subset_data in benchmark_data.items():
        for row in subset_data:
            result = {}
            if module.pointwise:
                # 处理 pointwise 结果
                for point_idx in [1, 2]:
                    result[f"point_{point_idx}"] = {
                        "completion": generated_texts[index]}
                    index += 1
            else:
                # pairwise 结果处理
                for swap_id in ["orig", "swap"]:
                    result[swap_id] = {"completion": generated_texts[index]}
                    index += 1
            row["result"] = result
    assert (len(generated_texts) == index)


def add_parse_result(benchmark_data: dict, module: InferenceModule) -> None:
    for subset_name, subset_data in benchmark_data.items():
        for row in subset_data:
            # 如果是 pointwise 模式
            if module.pointwise:
                for point_idx in [1, 2]:
                    result_dict = row["result"][f"point_{point_idx}"]
                    completion = result_dict["completion"]
                    # 获取评分
                    score = module.get_score(completion)
                    result_dict["score"] = score
                    # 如果能获取到评分，则根据评分判断结果
                    if score is not None:
                        # 不设置 prediction 和 is_correct，因为 pointwise 模式下需要两个分数比较
                        pass
                    else:
                        # 尝试使用原有的 get_prediction 方法
                        result_dict["prediction"] = module.get_prediction(
                            completion)
                        # pointwise 模式下无法直接判断正确性
                        result_dict["is_correct"] = False
            else:
                # 原有的 pairwise 处理逻辑
                for swap, swap_id in [(False, "orig"), (True, "swap")]:
                    result_dict = row["result"][swap_id]
                    completion = result_dict["completion"]
                    result_dict["prediction"] = module.get_prediction(
                        completion)
                    result_dict["is_correct"] = module.is_correct(
                        result_dict["prediction"], row["label"], swap)


def get_model_statistics(run_name: str) -> dict:
    """read all inference results for the model and return scores"""
    model_stats = {}
    for benchmark_id in BENCHMARK_IDS:
        benchmark_result = {}
        filename = f"result/{run_name}/{benchmark_id}.json"
        if not os.path.exists(filename):
            print("result file", filename, "does not exist.")
            continue
        with open(filename) as f:
            data = json.load(f)
        for subset_name, subset in data.items():
            stats = {key: 0 for key in ["single_total", "single_correct", "single_accuracy",
                                        "pair_total", "pair_correct", "pair_accuracy", "pair_agree", "pair_agreement_rate"]}
            for row in subset:
                # POINTWISE 模式处理
                if "point_1" in row["result"] and "point_2" in row["result"]:
                    p1 = row["result"]["point_1"]
                    p2 = row["result"]["point_2"]
                    score1 = p1.get("score")
                    score2 = p2.get("score")

                    stats["single_total"] += 1
                    if (score1 is not None) and (score2 is not None):
                        label = row.get("label")
                        if label == 1 and score1 > score2:
                            stats["single_correct"] += 1
                        elif label == 2 and score2 > score1:
                            stats["single_correct"] += 1

                    if (score1 is not None) and (score2 is not None):
                        stats["pair_total"] += 1
                        if score1 > score2:
                            pred = 1
                        elif score2 > score1:
                            pred = 2
                        else:
                            pred = 3
                        if pred == row.get("label"):
                            stats["pair_correct"] += 1
                        if pred != 3:
                            stats["pair_agree"] += 1

                # PAIRWISE 模式处理
                else:
                    stats["single_total"] += 2
                    stats["pair_total"] += 1
                    if row["result"]["orig"].get("is_correct", False):
                        stats["single_correct"] += 1
                    if row["result"]["swap"].get("is_correct", False):
                        stats["single_correct"] += 1
                    if row["result"]["orig"].get("is_correct", False) and row["result"]["swap"].get("is_correct", False):
                        stats["pair_correct"] += 1
                    pred_orig = row["result"]["orig"].get("prediction")
                    pred_swap = row["result"]["swap"].get("prediction")
                    if set([pred_orig, pred_swap]) in [set([1, 2]), set([3])]:
                        stats["pair_agree"] += 1

            # 计算比例
            if stats["single_total"] > 0:
                stats["single_accuracy"] = round(
                    stats["single_correct"] / stats["single_total"] * 100, 2)
            if stats["pair_total"] > 0:
                stats["pair_accuracy"] = round(
                    stats["pair_correct"] / stats["pair_total"] * 100, 2)
                stats["pair_agreement_rate"] = round(
                    stats["pair_agree"] / stats["pair_total"] * 100, 2)
            benchmark_result[subset_name] = stats

        # RewardBench 特殊处理：Reasoning = (Math + Code) 平均
        if benchmark_id == "rewardbench":
            if "Math" in benchmark_result and "Code" in benchmark_result:
                math_acc = benchmark_result["Math"]["pair_accuracy"]
                code_acc = benchmark_result["Code"]["pair_accuracy"]
                avg_acc = round((math_acc + code_acc) / 2, 2)
                benchmark_result["Reasoning"] = {
                    "pair_accuracy": avg_acc,
                    "note": "This is the average of Math and Code pair_accuracy."
                }

        # JudgeBench 特殊处理：计算整体准确率
        if benchmark_id == "judgebench":
            total_correct = 0
            total_count = 0
            for subset_name, stats in benchmark_result.items():
                total_correct += stats.get("single_correct", 0)
                total_count += stats.get("single_total", 0)
            if total_count > 0:
                overall_acc = round(total_correct / total_count * 100, 2)
                benchmark_result["Overall"] = {
                    "single_accuracy": overall_acc,
                    "single_total": total_count,
                    "single_correct": total_correct
                }

        # RM-Bench 特殊处理：计算 Easy / Normal / Hard 的平均准确率
        if benchmark_id == "rm-bench":
            difficulty_groups = {
                "Easy": ["Safety_Easy", "Math_Easy", "Chat_Easy", "Code_Easy"],
                "Normal": ["Safety_Normal", "Math_Normal", "Chat_Normal", "Code_Normal"],
                "Hard": ["Safety_Hard", "Math_Hard", "Chat_Hard", "Code_Hard"],
            }
            for diff_name, subsets in difficulty_groups.items():
                valid_acc = []
                for s in subsets:
                    if s in benchmark_result and "single_accuracy" in benchmark_result[s]:
                        valid_acc.append(
                            benchmark_result[s]["single_accuracy"])
                if valid_acc:
                    benchmark_result[diff_name] = {
                        "single_accuracy": round(sum(valid_acc) / len(valid_acc), 2),
                        "note": f"Average of {', '.join(subsets)} single_accuracy"
                    }

        model_stats[benchmark_id] = benchmark_result
    return model_stats


def write_model_score(run_name: str) -> None:
    """create model's score file"""
    model_stats = get_model_statistics(run_name)

    with open(f"result/{run_name}/score.json", "w") as f:
        json.dump(model_stats, fp=f, ensure_ascii=False, indent=4)


def run_benchmark(run_name: str, args: argparse.Namespace) -> None:
    """run inference, parse and score."""
    os.makedirs("result", exist_ok=True)
    os.makedirs(f"result/{run_name}", exist_ok=True)

    config = OmegaConf.load(args.config)
    OmegaConf.save(config, f"result/{run_name}/config.yaml")
    print(config)

    if (not args.hf) and (config.get("vllm_args")):
        module = VllmModule(config=config)
    elif (args.hf) and (config.get("hf_args")):
        module = HfModule(config=config)
    elif config.get("openai_args"):
        module = OpenaiModule(config=config)
    elif config.get("api_args"):
        module = ApiModule(config=config)
    else:
        raise NotImplementedError

    for benchmark_id in args.benchmarks:
        benchmark_data = get_benchmark_data(benchmark_id, args.data_path)

        add_inference(benchmark_data, module)
        add_parse_result(benchmark_data, module)

        with open(f"result/{run_name}/{benchmark_id}.json", "w") as f:
            json.dump(benchmark_data, fp=f, ensure_ascii=False, indent=2)

    write_model_score(run_name)


def run_parse(run_name: str, args: argparse.Namespace) -> None:
    """redo parsing for existing inference results, and update score."""
    config = OmegaConf.load(args.config)
    print(config)

    module = InferenceModule(config=config)
    for benchmark_id in args.benchmarks:
        with open(f"result/{run_name}/{benchmark_id}.json") as f:
            benchmark_data = json.load(f)
        add_parse_result(benchmark_data, module)
        with open(f"result/{run_name}/{benchmark_id}.json", "w") as f:
            json.dump(benchmark_data, fp=f, ensure_ascii=False, indent=2)

    write_model_score(run_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/llama2-7b.yaml")
    parser.add_argument(
        "--name", default="", help="run name of the inference. defaults to config name.")

    def list_of_strings(arg):
        return arg.split(',')
    parser.add_argument("--benchmarks", type=list_of_strings, default=["biasbench"],
                        help="to include all benchmarks, call as '--benchmarks llmbar,hhh,mtbench,biasbench'")
    parser.add_argument("--hf", action="store_true",
                        help="use hf generate instead of vllm")
    parser.add_argument("--parse", action="store_true",
                        help="no inference. just parse and score.")
    parser.add_argument("--score", action="store_true",
                        help="no inference. just score.")
    parser.add_argument("--data-path")

    args = parser.parse_args()
    print(args)

    run_name = os.path.basename(args.config).replace(".yaml", "")
    if args.hf:
        run_name += "_hf"
    if args.name:
        run_name = args.name
    print("Run name:", run_name)

    if args.score:
        write_model_score(run_name)
    elif args.parse:
        run_parse(run_name, args)
    else:
        run_benchmark(run_name, args)
