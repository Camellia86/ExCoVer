import os
import json
import base64
import time
from openai import OpenAI

def build_concise_prompt_o(error_samples, similar_intent_rules):
    """
    批量构建optimizer提示词

    Args:
        error_samples: 失败样本列表，每个元素为dict：
            {
                'context': '...',
                'sticker_text': '...',
                'true_intent': '...',
                'true_sentiment': '...',
                'learner_response': '...'
            }
        fine_grained_features: 现有的修正情感的细粒度特征表
        similar_intent_rules: 现有的相近意图判定规则表
    """
    # 处理现有的相近意图判定规则表
    similar_intent_display = similar_intent_rules

    # 构建批量样本展示文本
    samples_display = ""
    for idx, sample in enumerate(error_samples, 1):
        context = sample.get('context', '')
        sticker_text = sample.get('sticker_text', '')
        true_intent = sample.get('true_intent', '')
        true_sentiment = sample.get('true_sentiment', '')
        learner_response = sample.get('learner_response', '')

        # 处理上下文格式
        if context and context.strip():
            context_display = "，".join([line.strip() for line in context.split('\t') if line.strip()])
        else:
            context_display = "无"

        # 处理表情包文本
        sticker_text_display = sticker_text if sticker_text and sticker_text.strip() else "无"

        samples_display += f"""
【样本 {idx}】
• 上下文：{context_display}
• 表情包内文本：{sticker_text_display}
• 正确的意图和情感倾向标签：意图={true_intent}，情感倾向={true_sentiment}
• Learner的回复：{learner_response}
"""

    # 严格按照指定格式构建提示词
    prompt = f"""
角色与任务设定

你是一个社交媒体语义识别优化专家。你的核心任务是分析 Learner（识别模型）在处理"聊天记录+表情包"时的推理失误，通过溯源错误根因，提炼出混淆意图判定规则，并动态更新现有的知识库。

注意：本任务遇到的情感是情感倾向，只有无明显情感(neutral)、积极(positive)和消极(netural)。代理意图属于开集合，可为空，不止是预定的意图标签。

第一步：意图根因诊断（内心思考过程，不输出）
逻辑主次-对于该样本来说，错误意图也可能是辅助意图，Learner是否将"辅助意图"误判为了"核心意图"？
意图相近-learner预测的意图是否和正确意图相近？(如意图“抱怨”和“妥协”相近)

第二步：跨样本通用特征提取
从以下{len(error_samples)}个失败样本中，逐样本提取意图判定规则，然后检查这些规则，是否是**不与其他样本矛盾**的，优先选择能覆盖多个样本的规律，而非针对单个样本的特殊情况。

第三步：知识库更新准则
混淆意图判定规则：
针对 Learner 混淆的两个意图，提炼出明确的判别准则。
主次逻辑判定：若两个意图兼容，需注明判定主次的方法（例如："这是我朋友John，他足球踢的特别好"，包含介绍意图和表扬意图，但主要是介绍哥哥，因此是介绍意图，给出判定为介绍的原因）。格式：[意图A vs 意图B - 判定规则]
相近意图判定：若两意图意思相近，注意指出区分他们的关键代理意图或方法（例如："生活好难，我真没办法"，包含抱怨意图和妥协意图，但"没办法"推导出的代理意图"无奈"更贴近妥协，因此意图是妥协；例如“你作业能借我抄一下吗”，包含询问和求助意图，但用户更想得到作业来抄，因此是求助意图）。格式：[意图A vs 意图B - 判定规则]

第四步：维护规则（去重与合并）
覆盖升级：若旧规则中有同类型规则，且旧规则和新规则意思相近，则只补充新规则上的新的判别逻辑（也可能新旧意思一样，没有新的逻辑需要补充），若旧规则在当前案例中失效，则剔除旧规则中与现有案例矛盾的逻辑，并补充上新规则的逻辑，保证判定规则的简洁与凝练，但不失去关键细节，尤其是能够区分不同的意图的关键代理意图。
注意，意图不包括3种情感倾向，且你给出的需判定的意图要严格遵循给你的样本中出现的正确意图标签和模型错误的预测标签，不能改变意图名称。以此形成混淆意图判定规则。

输出格式要求：
严格按照以下格式输出，禁止任何多余的开场白或解释文字，务必注意省略号和判定规则只是必要的填充，不是一定要填充，是可为空的。且混淆意图判定规则只可以写意图标签，不可以写情感倾向标签：

混淆意图判定规则：
[意图A vs 意图B]：判定规则内容
...

任务输入：

【批量失败样本分析】
{samples_display}

【当前知识库】

• 待更新的混淆意图判定规则表：
{similar_intent_display}

"""

    return prompt