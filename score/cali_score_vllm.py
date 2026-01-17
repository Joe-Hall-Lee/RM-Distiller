import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# File paths
original_file = '/root/autodl-tmp/DistilRM/RewardTrainer/data/train/skywork_10k_orig_qwen_with_teacher_filtered_inconsistence.json'
bridge_file = '/root/autodl-tmp/DistilRM/responses/skywork_10k_orig_qwen_filtered_modified_inconsistent.jsonl'
output_file = '/root/autodl-tmp/DistilRM/RewardTrainer/data/train/skywork_10k_orig_qwen_filtered_bridge_inconsistent.json'

# Model configuration
model_name = '/root/autodl-tmp/DistilRM/models/Qwen3-14B'

# Initialize tokenizer and LLM
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(model=model_name, trust_remote_code=True, max_model_len=8192)

# Sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0.0,
    max_tokens=20,
)

# Load original JSON array
with open(original_file, 'r', encoding='utf-8') as f:
    original_data_all = json.load(f)

# 提前过滤
original_data = [
    item for item in original_data_all
    if item.get("chosen_score", 0) > item.get("rejected_score", 0) and item.get("chosen", 0) != item.get("rejected", 0)
]

print(
    f"Original total: {len(original_data_all)}, "
    f"After filter (chosen > rejected): {len(original_data)}"
)

# Load bridge JSONL
bridge_entries = []
with open(bridge_file, 'r', encoding='utf-8') as f:
    for line in f:
        bridge_entries.append(json.loads(line.strip()))

# Assume len(original_data) == len(bridge_entries), match by index
assert len(original_data) == len(bridge_entries), (
    f"Data lengths mismatch after filtering! "
    f"orig={len(original_data)}, bridge={len(bridge_entries)}"
)

# Prepare batch prompts for scoring
formatted_prompts = []
items = []
for i, item in enumerate(original_data):
    bridge_item = bridge_entries[i]

    # Construct context
    context = f"""<Start of Context>
##Human: {item['prompt']}
<End of Context>"""

    # Preferred response (using teacher_response as preferred)
    preferred = f"""<Start of Preferred Response>
{item['teacher_response']}
<End of Preferred Response>"""

    # Non-preferred response (rejected)
    non_preferred = f"""<Start of Non-Preferred Response>
{item['rejected']}
<End of Non-Preferred Response>"""

    # Full prompt text (instruction + context + preferred + non-preferred)
    instruction_template = """<Start of Instruction>
Given the dialogue context between the human and the assistant, you need to modify the given non-preferred to get a preferred reply from the assistant. Begin your answer with why the given non-preferred response is not preferred compared to the preferred response and how this can be improved. Output your editing by strictly following this format:

<Start of Reasoning>
<End of Reasoning>

<Start of Modified Response>
<End of Modified Response>

You must maintain the original structure of the non-preferred response, ensure the edit distance between the modified response and the non-preferred response is as low as possible.
<End of Instruction>"""

    full_prompt_text = instruction_template + context + preferred + non_preferred

    # Score prompt: ask to score the modified response, referencing rejected score
    score_prompt = f"""Now, considering the rejected response was scored {item['rejected_score']} out of 10 for quality, helpfulness, and alignment, score the Modified Response in the assistant's reply above on the same 1-10 scale. Output only the numerical score. Do NOT provide any explanation."""

    # Construct messages: user (full_prompt), assistant (completion), user (score_prompt)
    messages = [
        {"role": "user", "content": full_prompt_text},
        {"role": "assistant", "content": bridge_item['completion']},
        {"role": "user", "content": score_prompt}
    ]

    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    formatted_prompts.append(formatted_prompt)
    items.append((item, bridge_item))

# Generate scores in batch
outputs = llm.generate(formatted_prompts, sampling_params)

# Process outputs
scored_data = []
for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text.strip()
    original_item, bridge_item = items[i]

    # Extract numerical score using regex
    score_match = re.search(r'\b([0-9]+(\.[0-9]+)?)\b', generated_text)
    if score_match:
        new_chosen_score = float(score_match.group(1))
        # Clamp to 0-10
        new_chosen_score = max(0.0, min(10.0, new_chosen_score))
    else:
        # Fallback: default to average or warn
        new_chosen_score = 8.0  # Midpoint bias towards improvement
        print(
            f"Warning: Could not extract score for id {original_item['id']}, using 8.0")

    # Update original item
    updated_item = original_item.copy()
    # Replace chosen with modified
    updated_item['chosen'] = bridge_item['response']
    updated_item['chosen_score'] = new_chosen_score  # New score

    scored_data.append(updated_item)

# Save to JSON array
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(scored_data, f, ensure_ascii=False, indent=2)

print(f"Processing complete. Output saved to {output_file}")
