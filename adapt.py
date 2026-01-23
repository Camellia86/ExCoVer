import os
import json
import base64
import time
from openai import OpenAI

def build_concise_prompt_c(similar_intent_rules): #需要补充应该带入的变量
    """严格按照指定格式构建提示词"""
    # 处理现有的混淆意图判定规则表
    similar_intent_display = similar_intent_rules

    prompt = f"""
你需要对给你的混淆意图判定规则表进行调整，要求保留核心逻辑和你认为没问题的案例，将整体调整到适合你理解并拿来应用的程度，使其能够帮助你区分混淆意图，而非影响你判断。

输出格式要求
严格按照以下格式输出，严禁包含任何解释性文字或开场白，务必注意省略号和判定规则只是必要的填充，不是一定要填充，是可为空的：

混淆意图判定规则：
[意图A vs 意图B]：判定规则内容
...

输入信息定义

• 混淆意图判定规则表：
{similar_intent_display}

"""

    return prompt