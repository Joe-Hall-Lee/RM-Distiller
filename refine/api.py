import openai
import os
from dotenv import load_dotenv

load_dotenv()  # 加载 .env 文件中的变量


def query_model(prompt, model_name):
    """
    向指定的大模型发送请求并获取回复。

    此函数支持两种形式的输入:
    1.  一个字符串 (str): 将被视为单个用户问题。
    2.  一个列表 (list): 将被视为完整的对话历史消息列表。

    Args:
        prompt (str or list): 要发送给模型的问题字符串或消息列表。
        model_name (str): 要调用的模型名称。

    Returns:
        str: 模型的回复内容，如果发生错误则返回错误信息的字符串。
    """
    openai.api_key = os.getenv("API_KEY")
    openai.base_url = "https://yunwu.ai/v1/"

    messages = []
    if isinstance(prompt, str):
        # 如果输入是字符串，保持旧的兼容性，将其包装成消息列表
        messages = [
            {"role": "user", "content": prompt}
        ]
    elif isinstance(prompt, list):
        # 如果输入已经是列表，直接使用它作为消息历史
        messages = prompt
    else:
        # 对无效的输入类型返回错误
        return "Error: a prompt must be a string or a list of messages."

    try:
        response = openai.chat.completions.create(
            model=model_name,
            messages=messages
        )
        # print(response)
        return response.choices[0].message.content
    except Exception as e:
        return str(e)


# 问题
question = """[Question]
The polynomial $$P(x)=(1+x+x^2+\\ldots+x^{17})^2-x^{17}$$has 34 complex zeros of the form $z_k=r_k\\left[\\cos(2\\pi\\alpha_k)\n+i\\sin(2\\pi\\alpha_k)\\right]$, $k=1,2,3,\\ldots,34$, with $0<\\alpha_1\\le\\alpha_2\\le\\alpha_3\\le\\dots\\le\\alpha_{34}<1$ and $r_k>0$. Find $\\alpha_1+\\alpha_2+\\alpha_3+\\alpha_4+\\alpha_5.$",

[The Start of Assistant 1's Answer]
## Step 1:  The problem asks us to find the sum of the first five unique values of $\\alpha$ that satisfy the given polynomial equation.\n## Step 2:  We are given the polynomial $P(x) = (1+x+x^2+\\ldots+x^{17})^2 - x^{17}$ and are asked to find the sum of the first five unique values of $\\alpha$ that satisfy the equation $z_k=r_k\\left[\\cos(2\\pi\\alpha_k)+i\\sin(2\\pi\\alpha_k)\\right]$.\n## Step 3:  To start, we can simplify the expression for $P(x)$ by using the formula for the sum of a geometric series: $1+x+x^2+\\ldots+x^{17}=\\frac{x^{18}-1}{x-1}$.\n## Step 4:  Substituting this into the expression for $P(x)$, we get $P(x)=\\left(\\frac{x^{18}-1}{x-1}\\right)^2-x^{17}$.\n## Step 5:  We can then simplify this expression to obtain $P(x)=\\frac{(x^{19}-1)(x^{17}-1)}{(x-1)^2}$.\n## Step 6:  The roots of $P(x)$ are the roots of $x^{19}-1$ and $x^{17}-1$ excluding $x=1$.\n## Step 7:  The roots of $x^{19}-1$ are the 19th roots of unity, and the roots of $x^{17}-1$ are the 17th roots of unity.\n## Step 8:  The 19th roots of unity are given by $\\cos\\left(\\frac{2\\pi k}{19}\\right)+i\\sin\\left(\\frac{2\\pi k}{19}\\right)$ for $k=0,1,2,\\ldots,18$, and the 17th roots of unity are given by $\\cos\\left(\\frac{2\\pi k}{17}\\right)+i\\sin\\left(\\frac{2\\pi k}{17}\\right)$ for $k=0,1,2,\\ldots,16$.\n## Step 9:  We need to exclude the root $x=1$ from both sets of roots, so we remove the root $\\cos(0)+i\\sin(0)=1$ from the 19th roots of unity and the root $\\cos(0)+i\\sin(0)=1$ from the 17th roots of unity.\n## Step 10:  The remaining roots of $x^{19}-1$ are $\\cos\\left(\\frac{2\\pi k}{19}\\right)+i\\sin\\left(\\frac{2\\pi k}{19}\\right)$ for $k=1,2,\\ldots,18$, and the remaining roots of $x^{17}-1$ are $\\cos\\left(\\frac{2\\pi k}{17}\\right)+i\\sin\\left(\\frac{2\\pi k}{17}\\right)$ for $k=1,2,\\ldots,16$.\n## Step 11:  We need to find the sum of the first five unique values of $\\alpha$ that satisfy the equation $z_k=r_k\\left[\\cos(2\\pi\\alpha_k)+i\\sin(2\\pi\\alpha_k)\\right]$.\n## Step 12:  The first five unique values of $\\alpha$ are $\\frac{1}{19},\\frac{1}{17},\\frac{2}{19},\\frac{2}{17},\\frac{3}{19}$.\n## Step 13:  Therefore, the sum of the first five unique values of $\\alpha$ is $\\frac{1}{19}+\\frac{1}{17}+\\frac{2}{19}+\\frac{2}{17}+\\frac{3}{19}$.\n## Step 14:  We can find a common denominator to add these fractions together: $\\frac{1}{19}+\\frac{1}{17}+\\frac{2}{19}+\\frac{2}{17}+\\frac{3}{19}=\\frac{1\\cdot17+1\\cdot19+2\\cdot17+2\\cdot19+3\\cdot17}{19\\cdot17}$.\n## Step 15:  Simplifying the numerator, we get $\\frac{17+19+34+38+51}{323}$.\n## Step 16:  Combining the terms in the numerator, we get $\\frac{159}{323}$.\n## Step 17:  Therefore, the sum of the first five unique values of $\\alpha$ is $\\frac{159}{323}$.\n\nThe final answer is: $\\boxed{\\frac{159}{323}}$",

[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
Here's how to solve this problem using the properties of roots of polynomials and complex numbers:\n\n**Understanding the Problem**\n\n* **Geometric Interpretation:** The roots of the polynomial are complex numbers represented in polar form.  The angle  $2\\pi\\alpha_k$ represents the argument (angle) of the complex number $z_k$ in the complex plane.\n* **Roots of Unity:** Notice that the expression $(1+x+x^2+\\ldots+x^{17})$ is a geometric series.  Its sum is  $\\frac{1-x^{18}}{1-x}$. This suggests that the roots of this part of the polynomial might be related to the roots of unity.\n\n**Solution**\n\n1. **Roots of the Geometric Series:** The roots of the geometric series are the complex numbers that satisfy:\n   \n   $$\\frac{1-x^{18}}{1-x} = 0$$\n\n   This equation is satisfied when $1-x^{18} = 0$.  Therefore, the roots of the geometric series are the 18th roots of unity:\n\n   $$x_j = \\operatorname{cis}\\left(\\frac{2\\pi j}{18}\\right) = \\operatorname{cis}\\left(\\frac{\\pi j}{9}\\right)$$ \n   where $j = 0, 1, 2, \\ldots, 17$.\n\n2. **Roots of the Polynomial:**  The polynomial $P(x)$ has 34 roots. Since the geometric series part contributes 18 roots, the remaining 16 roots must come from the $-x^{17}$ term.  These roots are simply the 17th roots of unity:\n\n   $$x_j = \\operatorname{cis}\\left(\\frac{2\\pi j}{17}\\right)$$\n   where $j = 0, 1, 2, \\ldots, 16$.\n\n3. **Finding the Sum:**  The problem asks for $\\alpha_1 + \\alpha_2 + \\alpha_3 + \\alpha_4 + \\alpha_5$.  Since the roots are arranged in increasing order of their angles, we need to find the angles corresponding to the first five roots of the polynomial.\n\n   * The first five roots of the geometric series are:\n      * $\\operatorname{cis}(0)$\n      * $\\operatorname{cis}\\left(\\frac{\\pi}{9}\\right)$\n      * $\\operatorname{cis}\\left(\\frac{2\\pi}{9}\\right)$\n      * $\\operatorname{cis}\\left(\\frac{3\\pi}{9}\\right) = \\operatorname{cis}\\left(\\frac{\\pi}{3}\\right)$\n      * $\\operatorname{cis}\\left(\\frac{4\\pi}{9}\\right)$\n\n   * The first five roots of the $-x^{17}$ term are:\n      * $\\operatorname{cis}(0)$\n      * $\\operatorname{cis}\\left(\\frac{2\\pi}{17}\\right)$\n      * $\\operatorname{cis}\\left(\\frac{4\\pi}{17}\\right)$\n      * $\\operatorname{cis}\\left(\\frac{6\\pi}{17}\\right)$\n      * $\\operatorname{cis}\\left(\\frac{8\\pi}{17}\\right)$\n\n   *  Notice that $\\operatorname{cis}(0)$ appears in both lists.  \n\n   Therefore, the first five roots of $P(x)$ are:\n   * $\\operatorname{cis}(0)$\n   * $\\operatorname{cis}\\left(\\frac{\\pi}{9}\\right)$\n   * $\\operatorname{cis}\\left(\\frac{2\\pi}{9}\\right)$\n   * $\\operatorname{cis}\\left(\\frac{\\pi}{3}\\right)$\n   * $\\operatorname{cis}\\left(\\frac{4\\pi}{9}\\right)$\n\n   The sum of the angles corresponding to these roots is:\n   $$0 + \\frac{\\pi}{9} + \\frac{2\\pi}{9} + \\frac{\\pi}{3} + \\frac{4\\pi}{9} = \\boxed{\\frac{10\\pi}{9}}$$",

[The End of Assistant 2's Answer]

[System]
We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.
Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.
Please output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. Avoid any potential bias and ensure that the order in which the responses were presented does not affect your judgment. Do NOT provide any explanation.

### Response:"""

# answer = query_model(question, model_name="gemini-2.5-pro-nothinking")
# print(answer)
