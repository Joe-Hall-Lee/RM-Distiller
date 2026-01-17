# refine_vllm_fixed.py
# -*- coding: utf-8 -*-
import os
import json
import re
import uuid
import traceback
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# ==== 文件路径配置 ====
input_file = '/root/autodl-tmp/DistilRM/feedback/skywork_10k_qwen_feedback.jsonl'
output_file = 'revised/skywork_10k_qwen_revised_rejected.json'

# ==== 模型配置 ====
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

# ============= 新 Prompt 模板（修改 rejected，尽量最小改动） =============
prompt_template = """You are an expert reviser. 
Your task is to **refine the REJECTED response**, making it clearer, more accurate, more aligned with the CHECKLIST, 
while **making the minimum number of changes necessary**. 

Guidelines:
- Use CHOSEN, SCORES, DIFFERENCES, and CHECKLIST as informative signals.
- Improve REJECTED so it satisfies the CHECKLIST better.
- Do NOT rewrite everything—only modify what is needed.
- Output ONLY the following XML format:

<SCORE>a float score evaluating the improved REJECTED</SCORE>
<REVISED>
(the minimally revised REJECTED response)
</REVISED>

====== INPUT ======

<INPUT>
{input_text}
</INPUT>

<CHOSEN>
{chosen_text}
</CHOSEN>

<CHOSEN_SCORE>
{chosen_score}
</CHOSEN_SCORE>

<REJECTED>
{rejected_text}
</REJECTED>

<REJECTED_SCORE>
{rejected_score}
</REJECTED_SCORE>

<DIFFERENCES>
{differences}
</DIFFERENCES>

<CHECKLIST>
{checklist}
</CHECKLIST>

Now output ONLY:

<SCORE>...</SCORE>
<REVISED>...</REVISED>
"""


# ==== XML 标签解析函数 ====
def extract_tag(text, tag):
    pattern = rf"<{tag}>([\s\S]*?)</{tag}>"
    m = re.search(pattern, text)
    if not m:
        print(f"Warning: 未找到标签 <{tag}>")
        print(text.strip()[:150])
        return None
    return m.group(1).strip()


def parse_xml_revision(output_text):
    """
    解析 <SCORE> 与 <REVISED>
    """
    score_raw = extract_tag(output_text, "SCORE")
    revised = extract_tag(output_text, "REVISED")

    try:
        score = float(score_raw) if score_raw is not None else None
    except:
        score = None

    return {
        "revised": revised,
        "score": score,
    }


# ==== 主流程 ====
def main():
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 加载反馈 JSONL
    feedback_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                feedback_data.append(json.loads(line))

    # 创建 prompts
    prompts = []
    for item in feedback_data:
        prompt = prompt_template.format(
            input_text=item.get("question", ""),
            chosen_text=item.get("chosen", ""),
            chosen_score=item.get("chosen_score", 0),
            rejected_text=item.get("rejected", ""),
            rejected_score=item.get("rejected_score", 0),
            differences=item.get("differences", ""),
            checklist=item.get("checklist", "")
        )
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompts.append(formatted_prompt)

    # 调用模型
    outputs = llm.generate(prompts, sampling_params)

    # 处理结果
    results = []
    for i, item in enumerate(feedback_data):
        try:
            output_text = outputs[i].outputs[0].text.strip()
        except:
            output_text = ""

        parsed = parse_xml_revision(output_text)

        revised_rejected = item.get("rejected", "")
        revised_score = item.get("rejected_score", 0)

        if parsed:
            if parsed["revised"] is not None:
                revised_rejected = parsed["revised"]
            if parsed["score"] is not None:
                revised_score = parsed["score"]

        # 注意：这里保存的是 revised_rejected
        result = {
            "id": str(uuid.uuid4()),
            "prompt": item.get("question", ""),
            "chosen": item.get("chosen", ""),
            "rejected": revised_rejected,
            "chosen_score": item.get("chosen_score", 0),
            "rejected_score": revised_score,
            "teacher_output": output_text
        }

        results.append(result)

    # 写出结果
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=4)

    print(f"修订完成! 输出保存到: {output_file}")
    print(f"共处理 {len(results)} 条记录")


if __name__ == "__main__":
    main()
