import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str,
                        required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output_file", type=str,
                        required=True, help="输出 JSONL 文件路径")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    # 加载模型和分词器
    model = LLM(
        model=args.model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, trust_remote_code=True)

    # 读取 JSONL 数据
    data = []
    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    # 准备所有推理的 Prompts
    all_prompts = []
    num_items = len(data)

    for item in data:
        # 构建生成 critique 的 prompt 模板
        def make_critique_prompt(resp):
            return f"""User: {item['prompt']}
Chatbot: {resp}
Please provide a critique of the last response in one short paragraph. Your critique should be concise, specific, insightful and to the point. Aspects you should consider are: (1) Helpfulness. A good response should precisely/closely answer the user's request. (2) Correctness. A good response should be honest and factually correct."""

        # 处理 chosen 和 rejected
        p_chosen = make_critique_prompt(item['chosen'])
        p_rejected = make_critique_prompt(item['rejected'])

        # 应用聊天模板
        text_chosen = tokenizer.apply_chat_template(
            [{"role": "user", "content": p_chosen}],
            tokenize=False,
            add_generation_prompt=True, enable_thinking=False
        )
        text_rejected = tokenizer.apply_chat_template(
            [{"role": "user", "content": p_rejected}],
            tokenize=False,
            add_generation_prompt=True, enable_thinking=False
        )

        all_prompts.append(text_chosen)
        all_prompts.append(text_rejected)

    # 批处理生成
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
    )

    print(f"正在为 {num_items} 条数据（共 {len(all_prompts)} 个回复）生成 critique...")
    outputs = model.generate(all_prompts, sampling_params)

    # 拼接结果并保存为 JSONL
    print(f"正在保存结果到 {args.output_file}...")

    # 确保输出目录存在
    os.makedirs(os.path.dirname(
        os.path.abspath(args.output_file)), exist_ok=True)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for i in range(num_items):
            # 获取对应的输出索引（all_prompts 是 [c0, r0, c1, r1, ...] 结构）
            # 所以 2*i 是 chosen 的 critique，2*i + 1 是 rejected 的 critique
            critique_chosen = outputs[2*i].outputs[0].text.strip()
            critique_rejected = outputs[2*i + 1].outputs[0].text.strip()

            # 拼接
            data[i]['chosen'] = data[i]['chosen'] + "\n\n" + critique_chosen
            data[i]['rejected'] = data[i]['rejected'] + \
                "\n\n" + critique_rejected

            # 写入一行 JSONL
            f.write(json.dumps(data[i], ensure_ascii=False) + '\n')

    print("完成！")


if __name__ == "__main__":
    main()
