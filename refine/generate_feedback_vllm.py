import os
import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ==== File paths ====
input_file = 'RewardTrainer/data/train/skywork_10k_qwen.json'
output_file = 'feedback/skywork_10k_qwen_feedback.jsonl'

# ==== Model ====
model_name = 'models/Qwen3-14B'

tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, trust_remote_code=True, max_model_len=8192)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    max_tokens=2048,
)

# ==== PROMPT ====
prompt_template = """You are an expert evaluator. You will be given two candidate responses to the same input, together with their numeric scores.
Your job is to carefully compare the two responses across semantics, correctness, completeness, reasoning quality, factuality, style, logical consistency, safety or any other relevant aspects.

During comparison, consider the following questions:
1. Are the two responses inherently different?
2. Where are they different?
3. What causes these differences? (e.g., misunderstanding, missing information, reasoning errors, stylistic choices, hallucinations…)

After contrasting, you should generate a checklist based on these differences between two responses. You should carefully consider each discrepancy and the reasons behind it, summarizing them into a few checking instructions in the checklist. This checklist can guide others to re-examine the input and these responses.

Make sure your output strictly follows:

[Output Format]
- Differences: {{A detailed comparison between the chosen and rejected responses}}
- Checklist: {{ChecklistItem1, ChecklistItem2, ChecklistItem3, …}}

[Input]
- Input: {input_text}
- Chosen Response: {chosen}
- Rejected Response: {rejected}
- Chosen Score: {chosen_score}
- Rejected Score: {rejected_score}
"""

# ==== Load input ====
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ==== Prepare all prompts ====
formatted_prompts = []
items = []

for item in data:
    text = prompt_template.format(
        input_text=item["prompt"],
        chosen=item["chosen"],
        rejected=item["rejected"],
        chosen_score=item["chosen_score"],
        rejected_score=item["rejected_score"],
    )

    messages = [{"role": "user", "content": text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    formatted_prompts.append(prompt)
    items.append(item)

# ==== Generate ====
outputs = llm.generate(formatted_prompts, sampling_params)

# ==== Output JSONL ====


def extract_field(tag, text):
    """
    Extract content like:
    - Differences: {...}
    - Checklist: {...}
    """
    pattern = fr'{tag}:\s*(.*?)\s*(?=(Checklist:|$))' if tag == "Differences" else fr'{tag}:\s*(.*)'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


# Ensure output directory exists
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created directory: {output_dir}")


with open(output_file, "w", encoding="utf-8") as f:
    for i, output in enumerate(outputs):
        item = items[i]
        full_text = output.outputs[0].text.strip()

        differences = extract_field("Differences", full_text)
        checklist = extract_field("Checklist", full_text)

        entry = {
            "question": item["prompt"],
            "chosen": item["chosen"],
            "rejected": item["rejected"],
            "chosen_score": item["chosen_score"],
            "rejected_score": item["rejected_score"],
            "completion": full_text,
            "differences": differences,
            "checklist": checklist,
        }

        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"Done. Saved to {output_file}")
