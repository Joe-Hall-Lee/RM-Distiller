system = """You are a helpful and precise assistant for checking the quality of the answer."""

user = """Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user prompt displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, level of detail, and ethicality of the response. Be as objective as possible. Please rate the response with a score from 1 to 10, without further explanation.

[Question]
{input}

[The Assistant's Answer]
{output}

Please output only a value from 1 to 10 on a single line. Do NOT provide any explanation or additional text."""

# 匹配单个整数或小数评分（1-10）
output_pattern = {
    1: r"(?:^|\n)\s*(10|[1-9](?:\.\d+)?|0\.\d+)\s*(?:$|\n)",
}
