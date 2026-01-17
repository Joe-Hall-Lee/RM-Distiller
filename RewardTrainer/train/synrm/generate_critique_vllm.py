import json
import argparse
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-14B")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = parser.parse_args()

    # Load model
    model = LLM(
        model=args.model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    # Load data
    with open(args.input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Prepare prompts for batch inference
    chosen_prompts = []
    rejected_prompts = []
    indices = []

    for i, item in enumerate(data):
        critique_prompt_chosen = f"""User: {item['prompt']}
Chatbot: {item['chosen']}
Please provide a critique of the last response in one short paragraph. Your critique should be concise, specific, insightful and to the point. Aspects you should consider are: (1) Helpfulness. A good response should precisely/closely answer the user's request. (2) Correctness. A good response should be honest and factually correct."""

        critique_prompt_rejected = f"""User: {item['prompt']}
Chatbot: {item['rejected']}
Please provide a critique of the last response in one short paragraph. Your critique should be concise, specific, insightful and to the point. Aspects you should consider are: (1) Helpfulness. A good response should precisely/closely answer the user's request. (2) Correctness. A good response should be honest and factually correct."""

        messages_chosen = [{"role": "user", "content": critique_prompt_chosen}]
        messages_rejected = [{"role": "user", "content": critique_prompt_rejected}]

        input_text_chosen = tokenizer.apply_chat_template(messages_chosen, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        input_text_rejected = tokenizer.apply_chat_template(messages_rejected, tokenize=False, add_generation_prompt=True, enable_thinking=False)

        chosen_prompts.append(input_text_chosen)
        rejected_prompts.append(input_text_rejected)
        indices.append(i)

    # Batch generate for chosen
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        min_p=0.0,
        max_tokens=1024,
    )

    print("Generating critiques for chosen responses...")
    chosen_outputs = model.generate(chosen_prompts, sampling_params)

    print("Generating critiques for rejected responses...")
    rejected_outputs = model.generate(rejected_prompts, sampling_params)

    # Add critiques to data
    for i, idx in enumerate(indices):
        data[idx]['chosen'] = data[idx]['chosen'] + "\n\n" + chosen_outputs[i].outputs[0].text.strip()
        data[idx]['rejected'] = data[idx]['rejected'] + "\n\n" + rejected_outputs[i].outputs[0].text.strip()

    # Save data
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print("Done!")

if __name__ == "__main__":
    main()
