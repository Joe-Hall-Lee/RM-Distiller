import json
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# File paths
input_file = '/root/autodl-tmp/DistilRM/RewardTrainer/data/train/skywork_10k_orig_qwen_with_teacher.json'
output_file = '/root/autodl-tmp/DistilRM/responses/skywork_10k_orig_qwen_filtered_modified_inconsistent.jsonl'

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
    max_tokens=2048,
)

# Fixed instruction template
instruction_template = """<Start of Instruction>
Given the dialogue context between the human and the assistant, you need to modify the given non-preferred to get a preferred reply from the assistant. Begin your answer with why the given non-preferred response is not preferred compared to the preferred response and how this can be improved. Output your editing by strictly following this format:

<Start of Reasoning>
<End of Reasoning>

<Start of Modified Response>
<End of Modified Response>

You must maintain the original structure of the non-preferred response, ensure the edit distance between the modified response and the non-preferred response is as low as possible.
<End of Instruction>"""

# Load the JSON array
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

filtered_data = []
for item in data:
        filtered_data.append(item)

print(f"Total loaded: {len(data)}, kept after filtering: {len(filtered_data)}")

# Prepare batch prompts
formatted_prompts = []
items = []

for item in filtered_data:
    # Construct context
    context = f"""<Start of Context>
##Human: {item['prompt']}
<End of Context>"""

    # Preferred response
    preferred = f"""<Start of Preferred Response>
{item['teacher_response']}
<End of Preferred Response>"""

    # Non-preferred response (rejected)
    non_preferred = f"""<Start of Non-Preferred Response>
{item['rejected']}
<End of Non-Preferred Response>"""

    # Full prompt text
    full_prompt_text = instruction_template + context + preferred + non_preferred

    # Construct messages for chat template
    messages = [
        {"role": "user", "content": full_prompt_text}
    ]

    # Apply chat template
    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    formatted_prompts.append(formatted_prompt)
    items.append(item)

# Generate responses in batch
outputs = llm.generate(formatted_prompts, sampling_params)

# Process outputs
modified_entries = []
for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text.strip()
    item = items[i]

    # Extract the modified response using regex
    match = re.search(
        r'<Start of Modified Response>\s*(.*?)\s*<End of Modified Response>',
        generated_text,
        re.DOTALL
    )

    if not match:
        # Try with '/' in the end tag
        match = re.search(
            r'<Start of Modified Response>\s*(.*?)\s*</End of Modified Response>',
            generated_text,
            re.DOTALL
        )

    if match:
        modified_response = match.group(1).strip()
    else:
        # Fallback
        modified_response = generated_text.split(
            '<End of Reasoning>')[-1].strip() if '<End of Reasoning>' in generated_text else generated_text
        print(
            f"Warning: Could not extract modified response for id {item.get('id', 'unknown')}, using full generation.")

    # Create entry with specified fields
    entry = {
        "question": item['prompt'],
        "model": model_name,
        "completion": generated_text,
        "response": modified_response
    }
    modified_entries.append(entry)

# Save to jsonl
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in modified_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Processing complete. Output saved to {output_file}")
