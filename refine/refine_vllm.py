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
input_file = '/root/autodl-tmp/DistilRM/feedback/skywork_10k_qwen_feedback.jsonl'  # 输入反馈文件
output_file = 'revised/skywork_10k_qwen_revised.json'  # 输出文件

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


prompt_template = """You are an expert reviser. Your task is to **actively improve** the CHOSEN response—not just fix errors, but refine clarity, precision, completeness, and alignment with the CHECKLIST. Even if the CHOSEN seems acceptable, you must look for opportunities to enhance it using the guidance from the CHECKLIST and the contrast with the REJECTED response.

Critical instructions:
- **Do not assume the CHOSEN is perfect.** Treat every response as improvable unless it flawlessly satisfies every item in the CHECKlist.
- If the CHECKLIST suggests any room for improvement, you **must revise**.
- The presence of DIFFERENCES between CHOSEN and REJECTED often reveals issues. Make the revised chosen response better than the original chosen response, and avoid the mistake of rejected response.
- Only output `<CHANGED>false</CHANGED>` if the CHOSEN already **exactly and fully** meets all checklist criteria with no possible enhancement. This should be extremely rare.

You MUST output ONLY the following XML format—nothing else:

<CHANGED>true or false</CHANGED>
<SCORE>a float reflecting the quality of the revised response</SCORE>
<REVISED>
(the fully revised CHOSEN content, incorporating all necessary improvements)
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

Now output ONLY the XML tags:
<CHANGED>...</CHANGED>
<SCORE>...</SCORE>
<REVISED>...</REVISED>
"""


# ==== XML 标签解析函数 ====
def extract_tag(text, tag):
    """提取 <TAG>...</TAG> 内容"""
    pattern = rf"<{tag}>([\s\S]*?)</{tag}>"
    m = re.search(pattern, text)
    if not m:
        print(f"Warning: 未找到标签 <{tag}>")
        print(text.strip()[:100])
        return None
    return m.group(1).strip()


def parse_xml_revision(output_text):
    """
    从模型输出解析 changed / score / revised 三个字段
    """
    changed_raw = extract_tag(output_text, "CHANGED")
    score_raw = extract_tag(output_text, "SCORE")
    revised = extract_tag(output_text, "REVISED")

    changed = (changed_raw is not None and changed_raw.lower() == "true")

    try:
        score = float(score_raw) if score_raw is not None else None
    except:
        score = None

    return {
        "changed": changed,
        "revised": revised,
        "score": score,
    }


# ==== 主处理流程 ====
def main():
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 加载反馈数据
    feedback_data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                feedback_data.append(json.loads(line))

    # 准备 prompts
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

    # 调模型
    outputs = llm.generate(prompts, sampling_params)

    # 处理结果
    results = []
    for i, item in enumerate(feedback_data):
        try:
            output_text = outputs[i].outputs[0].text.strip()
        except:
            output_text = ""

        parsed = parse_xml_revision(output_text)

        revised_chosen = item.get("chosen", "")
        revised_score = item.get("chosen_score", 0)
        changed_flag = False

        if parsed:
            if parsed["revised"] is not None:
                revised_chosen = parsed["revised"]
            if parsed["score"] is not None:
                revised_score = parsed["score"]
            changed_flag = parsed["changed"]

        result = {
            "id": str(uuid.uuid4()),
            "prompt": item.get("question", ""),
            "chosen": revised_chosen,
            "rejected": item.get("rejected", ""),
            "chosen_score": revised_score,
            "rejected_score": item.get("rejected_score", 0),
            "teacher_output": output_text,
            "changed": changed_flag
        }

        results.append(result)

    # 保存
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(results, f_out, ensure_ascii=False, indent=4)

    print(f"修订完成! 输出保存到: {output_file}")
    print(f"共处理 {len(results)} 条记录")


if __name__ == "__main__":
    main()
