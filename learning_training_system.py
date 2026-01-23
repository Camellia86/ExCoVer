import os
import json
import base64
import time
import re
from typing import Dict, Tuple, List, Any, Optional
from openai import OpenAI
from learner_prompt import build_concise_prompt_l
from optimizer_prompt import build_concise_prompt_o
from regularizer_prompt import build_concise_prompt_r
from adapt import build_concise_prompt_c
from a_result3 import extract_labels_from_response

# 初始化客户端 (Learner & Optimizer)
client = OpenAI(
    api_key="Your API_KEY",
    base_url="API_URL",
)

# 初始化Regularizer客户端 (使用Gemini-3)
regularizer_client = OpenAI(
    api_key="Your API_KEY",
    base_url="API_URL",
)

# 意图标签映射 (中文)
INTENT_MAPPING = {
    "抱怨": 0, "表扬": 1, "同意，认同": 2, "妥协": 3, "询问": 4,
    "开玩笑，说着玩": 5, "拒绝": 6, "告知，通知": 7, "求助": 8, "问候": 9,
    "嘲讽": 10, "介绍": 11, "猜测，估计": 12, "离开": 13, "建议": 14,
    "炫耀": 15, "批评": 16, "道谢": 17, "安慰": 18, "道歉": 19
}

INTENT_ID_TO_NAME = {v: k for k, v in INTENT_MAPPING.items()}

# 英文意图标签到中文的映射
INTENT_EN_TO_CN = {
    "Complain": "抱怨",
    "Praise": "表扬",
    "Agree": "同意，认同",
    "Compromise": "妥协",
    "Query": "询问",
    "Joke": "开玩笑，说着玩",
    "Oppose": "拒绝",
    "Inform": "告知，通知",
    "Ask for help": "求助",
    "Greet": "问候",
    "Taunt": "嘲讽",
    "Introduce": "介绍",
    "Guess": "猜测，估计",
    "Leave": "离开",
    "Advise": "建议",
    "Flaunt": "炫耀",
    "Criticize": "批评",
    "Thank": "道谢",
    "Comfort": "安慰",
    "Apologize": "道歉"
}

# ID到英文标签的反向映射
INTENT_ID_TO_EN = {v: k for k, v in {
    "Complain": 0, "Praise": 1, "Agree": 2, "Compromise": 3, "Query": 4,
    "Joke": 5, "Oppose": 6, "Inform": 7, "Ask for help": 8, "Greet": 9,
    "Taunt": 10, "Introduce": 11, "Guess": 12, "Leave": 13, "Advise": 14,
    "Flaunt": 15, "Criticize": 16, "Thank": 17, "Comfort": 18, "Apologize": 19
}.items()}

SENTIMENT_MAPPING = {
    "无明显情绪": 0, "积极": 1, "消极": 2
}

SENTIMENT_ID_TO_NAME = {v: k for k, v in SENTIMENT_MAPPING.items()}

# 情感标签ID映射
SENTIMENT_EN_TO_ID = {
    "Neutral": 0,
    "Positive": 1,
    "Negative": 2
}


class LearningTrainingSystem:
    """语言化学习训练系统"""

    def __init__(self, train_json_path: str, base_image_path: str = "", batch_size: int = 1):
        """
        初始化学习训练系统

        Args:
            train_json_path: 训练数据JSON文件路径
            base_image_path: 图片基础路径
            batch_size: 批处理大小（默认=1表示立即更新，>1表示累积batch_size个样本后更新）
        """
        self.train_json_path = train_json_path
        self.base_image_path = base_image_path

        # Batch处理配置
        self.batch_size = batch_size
        print(f"[INIT DEBUG] 接收到的 batch_size 参数: {batch_size}")
        print(f"[INIT DEBUG] self.batch_size 已设置为: {self.batch_size}")
        self.batch_sample_count = 0  # 当前batch中的样本计数
        self.batch_errors = []  # 当前batch中的错误样本
        self.batch_count = 0  # 已完成的batch计数

        # 加载训练数据
        self.train_data = self._load_train_data()

        # 初始化规则表
        self.similar_intent_rules = self._load_similar_intent_rules()

        # 训练统计
        self.training_stats = {
            "total_samples": len(self.train_data),
            "correct_count": 0,
            "error_count": 0,
            "optimizer_calls": 0,
            "regularizer_calls": 0,
            "cleaner_calls": 0
        }

    def _load_train_data(self) -> List[Dict]:
        """加载训练数据"""
        try:
            with open(self.train_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"成功加载 {len(data)} 条训练数据")
            return data
        except Exception as e:
            print(f"加载训练数据失败: {e}")
            return []

    def _load_similar_intent_rules(self) -> str:
        """加载混淆意图判定规则表"""
        rules_file = "Similar Intent Determination Rules.txt"
        try:
            if os.path.exists(rules_file):
                with open(rules_file, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"加载规则表失败: {e}")
        return ""

    def _save_agent_response_json(self, sample_id: int, agent_name: str, response: str,
                                   item: Dict = None, pred_intent: str = "", pred_sentiment: int = -1,
                                   true_intent: str = "", true_sentiment: int = -1):
        """将Agent的响应保存到完整的json文件（包含样本信息）"""
        try:
            # 创建agent_responses目录（如果不存在）
            os.makedirs("agent_responses", exist_ok=True)

            # 构建文件名：agent_responses/sample_{id}_{agent}.json
            json_file = f"agent_responses/sample_{sample_id:05d}_{agent_name}.json"

            # 构建完整的响应数据（包含样本信息）
            response_data = {
                "sample_id": sample_id,
                "agent": agent_name,
                "context": item.get('context', '') if item else "",
                "sticker": item.get('sticker', '') if item else "",
                "sticker_text": item.get('sticker_text', '') if item else "",
                "true_intent": true_intent,
                "true_sentiment": true_sentiment,
                "pred_intent": pred_intent if pred_intent else "",
                "pred_sentiment": pred_sentiment if pred_sentiment != -1 else "",
                "agent_response": response,
                "timestamp": time.time()
            }

            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(response_data, f, ensure_ascii=False, indent=2)

            print(f"✓ {agent_name} 响应已保存: {json_file}")
            return json_file

        except Exception as e:
            print(f"❌ 保存 {agent_name} 响应失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _save_feature_and_rules(self, index: int):
        """保存规则表（定期检查点）"""
        rules_file = f"Similar Intent Determination Rules_step{index}.txt"

        try:
            # 从主文件复制当前内容作为检查点
            with open("Similar Intent Determination Rules.txt", 'r', encoding='utf-8') as f:
                content = f.read()
            with open(rules_file, 'w', encoding='utf-8') as f:
                f.write(content)

            print(f"✓ 已保存第 {index} 步的规则表检查点")
        except Exception as e:
            print(f"保存规则表检查点失败: {e}")

    def local_img_to_url(self, img_path: str) -> str:
        """将本地图片转换为Base64 URL"""
        try:
            if not os.path.exists(img_path):
                return ""
            ext = os.path.splitext(img_path)[1].lstrip('.')
            with open(img_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return f"data:image/{ext};base64,{b64}"
        except Exception as e:
            print(f"转换图片失败: {e}")
            return ""

    def call_learner(self, item: Dict) -> Tuple[str, str, int]:
        """
        调用learner获取预测结果

        Returns:
            (model_response, pred_intent, pred_sentiment)
            - pred_intent: 英文意图标签（如'Praise'）
            - pred_sentiment: 情感数字标签（0-2）
        """
        try:
            context = item.get('context', '')
            sticker_text = item.get('sticker_text', '')

            # 构建提示词
            prompt = build_concise_prompt_l(
                context,
                sticker_text,
                self.similar_intent_rules
            )

            # 构建图片URL（如果有）
            sticker_id = item.get('sticker', '')
            image_content = []

            if self.base_image_path and sticker_id:
                img_path_png = os.path.join(self.base_image_path, f"{sticker_id}.png")
                img_path_webp = os.path.join(self.base_image_path, f"{sticker_id}.webp")

                img_path = None
                if os.path.exists(img_path_png):
                    img_path = img_path_png
                elif os.path.exists(img_path_webp):
                    img_path = img_path_webp

                if img_path:
                    img_url = self.local_img_to_url(img_path)
                    if img_url:
                        image_content.append({
                            "type": "image_url",
                            "image_url": {"url": img_url}
                        })

            # 添加文本内容
            image_content.append({
                "type": "text",
                "text": prompt
            })

            # 调用API
            completion = client.chat.completions.create(
                model="doubao-seed-1-6-vision-250815",
                messages=[
                    {
                        "role": "user",
                        "content": image_content
                    }
                ],
                extra_body={
                    'enable_thinking': False,
                    "thinking_budget": 8192
                }
            )

            # 提取响应内容
            model_response = completion.choices[0].message.content or ""

            # 提取标签（返回ID）
            pred_intent_id, pred_sentiment = extract_labels_from_response(model_response)

            # 转换意图ID为英文标签
            pred_intent = INTENT_ID_TO_EN.get(pred_intent_id, '')

            return model_response, pred_intent, pred_sentiment

        except Exception as e:
            print(f"调用learner失败: {e}")
            return "", "", -1

    def call_optimizer(self, error_samples: list):
        """
        批量调用optimizer更新特征表和规则表

        Args:
            error_samples: 失败样本列表，每个元素为dict：
                {
                    'item': {...},  # 原始样本数据
                    'model_response': '...',  # Learner的回复
                    'pred_intent': '...',  # Learner的预测意图
                    'pred_sentiment': int,  # Learner的预测情感
                    'true_intent': '...',  # 正确意图
                    'true_sentiment': int  # 正确情感
                }
        """
        if not error_samples:
            print("⚠️  没有失败样本，跳过optimizer")
            return ""

        try:
            # 构建optimizer需要的样本格式
            formatted_samples = []
            for sample in error_samples:
                item = sample['item']
                context = item.get('context', '')
                sticker_text = item.get('sticker_text', '')
                true_intent = sample['true_intent']
                true_sentiment = sample['true_sentiment']
                learner_response = sample['model_response']

                # 转换英文标签为中文用于prompt
                true_intent_cn = INTENT_EN_TO_CN.get(true_intent, true_intent)
                true_sentiment_name = SENTIMENT_ID_TO_NAME.get(true_sentiment, f"情感{true_sentiment}")

                formatted_samples.append({
                    'context': context,
                    'sticker_text': sticker_text,
                    'true_intent': true_intent_cn,
                    'true_sentiment': true_sentiment_name,
                    'learner_response': learner_response
                })

            # 构建优化器提示词（批量）
            prompt = build_concise_prompt_o(
                formatted_samples,
                self.similar_intent_rules
            )

            # 调用API
            print(f"🔧 调用Optimizer处理{len(error_samples)}个失败样本...")
            completion = client.chat.completions.create(
                model="doubao-seed-1-6-vision-250815",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                extra_body={
                    'enable_thinking': True,
                    "thinking_budget": 8192
                }
            )

            optimizer_response = completion.choices[0].message.content or ""

            # 注意：Optimizer返回建议后，由调用者在train()中立即调用_update_tables_from_response()来提取内容
            # 然后Regularizer处理已更新的表
            self.training_stats["optimizer_calls"] += 1

            return optimizer_response

        except Exception as e:
            print(f"❌ 调用optimizer失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def call_regularizer(self, is_global: bool = False):
        """调用regularizer验证和优化表

        Args:
            is_global: 如果为True，则处理全局的表文件；如果为False，处理当前内存中的表
        """
        try:
            # 如果是全局处理，先从文件读取表
            if is_global:
                try:
                    with open("Similar Intent Determination Rules.txt", 'r', encoding='utf-8') as f:
                        rules_to_process = f.read()
                    print("\n🔄 全局Regularizer: 从文件读取表进行处理...")
                except Exception as e:
                    print(f"❌ 读取全局表文件失败: {e}")
                    return ""
            else:
                # 使用内存中的表（可能已被Optimizer更新）
                rules_to_process = self.similar_intent_rules

            # 构建提示词（不需要optimizer_response，regularizer直接处理当前表状态）
            prompt = build_concise_prompt_r(
                rules_to_process
            )

            # 调用API (使用Gemini-3)
            completion = regularizer_client.chat.completions.create(
                model="deepseek-v3-2-251201",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            regularizer_response = completion.choices[0].message.content or ""

            # 更新表
            if is_global:
                # 全局regularizer: 提取表但不调用_save_tables_to_files（避免冗余）
                # 在外层直接处理文件保存
                self._update_tables_from_response(regularizer_response, "regularizer_global")
            else:
                self._update_tables_from_response(regularizer_response, "regularizer")
            self.training_stats["regularizer_calls"] += 1

            return regularizer_response

        except Exception as e:
            print(f"调用regularizer失败: {e}")
            return ""

    def call_cleaner(self):
        """调用cleaner对regularizer优化后的表进行进一步清理"""
        try:
            # 使用内存中的表（已被Regularizer优化）
            rules_to_process = self.similar_intent_rules

            # 构建提示词
            prompt = build_concise_prompt_c(
                rules_to_process
            )

            # 调用API (使用Doubao客户端，同Optimizer，开启思考)
            print(f"🧹 调用Cleaner对表进行进一步清理...")
            completion = client.chat.completions.create(
                model="doubao-seed-1-6-vision-250815",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                extra_body={
                    'enable_thinking': True,
                    "thinking_budget": 8192
                }
            )

            cleaner_response = completion.choices[0].message.content or ""

            # 从Cleaner响应中提取表内容并更新
            if cleaner_response:
                self._update_tables_from_response(cleaner_response, "cleaner")
                self.training_stats["cleaner_calls"] += 1

            return cleaner_response

        except Exception as e:
            print(f"❌ 调用cleaner失败: {e}")
            import traceback
            traceback.print_exc()
            return ""

    def _update_tables_from_response(self, response: str, source: str):
        """从响应中提取更新的表 - 基于明确的section headers"""
        try:
            print(f"\n🔍 [{source}] 开始解析响应...")

            # 提取混淆意图判定规则部分 - 从"混淆意图判定规则："开始到末尾
            rules_pattern = r"混淆意图判定规则：(.*)"
            rules_match = re.search(rules_pattern, response, re.DOTALL)

            rules_extracted = False
            if rules_match:
                updated_rules = rules_match.group(1).strip()
                if updated_rules:  # 防止空更新
                    self.similar_intent_rules = updated_rules
                    rules_extracted = True
                    print(f"  ✓ 混淆意图判定规则: 匹配成功 ({len(updated_rules)} 字符)")
                else:
                    print(f"  ⚠ 混淆意图判定规则: 匹配但内容为空")
            else:
                print(f"  ✗ 混淆意图判定规则: 未找到")

            # 保存更新后的表到txt文件
            if rules_extracted:
                # 如果是Optimizer的输出，保存到ablation文件和主表文件
                if source == "optimizer":
                    self._save_ablation_tables()
                    self._save_tables_to_files()
                elif source == "regularizer_global":
                    self._save_global_regularizer_tables()
                elif source == "cleaner":
                    self._save_cleaner_tables()
                # # 其他情况（样本级regularizer）：追加到文件
                # else:
                #     self._save_tables_to_files()
                print(f"\n✅ [{source}] 表格已成功提取并保存")
            else:
                print(f"\n⚠️  [{source}] 未能提取任何表格内容")

        except Exception as e:
            print(f"❌ 更新表格失败 ({source}): {e}")
            import traceback
            traceback.print_exc()

    def _save_ablation_tables(self):
        """将Optimizer提取的表保存到ablation文件中"""
        try:
            ablation_rules_file = "ablation-规则.txt"

            # 保存到ablation规则文件
            with open(ablation_rules_file, "a", encoding="utf-8") as f:
                f.write(self.similar_intent_rules)
            print(f"    📊 Ablation规则表已保存: {ablation_rules_file} ({len(self.similar_intent_rules)} 字符)")

        except Exception as e:
            print(f"❌ 保存ablation表到文件失败: {e}")

    def _save_tables_to_files(self):
        """将当前的表保存到txt文件中"""
        try:
            rules_file = "Similar Intent Determination Rules.txt"

            # 保存混淆意图判定规则表
            with open(rules_file, "a", encoding="utf-8") as f:
                f.write(self.similar_intent_rules)
            print(f"    💾 规则表已保存: {rules_file} ({len(self.similar_intent_rules)} 字符)")

        except Exception as e:
            print(f"❌ 保存表到文件失败: {e}")
    def _save_global_regularizer_tables(self):
        """直接覆盖主表文件（全局Regularizer输出）- 不产生冗余"""
        try:
            rules_file = "Similar Intent Determination Rules.txt"
            # 直接覆盖混淆意图判定规则表
            with open(rules_file, "w", encoding="utf-8") as f:
                f.write(self.similar_intent_rules)
            print(f"    ✅ 规则表已覆盖（全局Regularizer）: {rules_file}")

        except Exception as e:
            print(f"❌ 保存全局Regularizer表到文件失败: {e}")

    def _save_cleaner_tables(self):
        """直接覆盖主表文件（Cleaner输出）"""
        try:
            rules_file = "Similar Intent Determination Rules.txt"
            # 直接覆盖混淆意图判定规则表
            with open(rules_file, "w", encoding="utf-8") as f:
                f.write(self.similar_intent_rules)
            print(f"    ✨ 规则表已清理并覆盖（Cleaner）: {rules_file}")

        except Exception as e:
            print(f"❌ 保存Cleaner表到文件失败: {e}")

    def train(self, max_samples: Optional[int] = None, save_interval: int = 50, resume_from: int = 0):
        """
        执行训练循环

        Args:
            max_samples: 最多训练样本数（None表示全部）
            save_interval: 每多少个批次保存一次检查点（当batch_size>1时）或每多少个样本（当batch_size=1时）
            resume_from: 从第几个样本开始（0表示从头开始）
        """
        samples_to_process = self.train_data[resume_from:max_samples] if max_samples else self.train_data[resume_from:]

        # 如果是续训，重新加载已有的表内容
        if resume_from > 0:
            print(f"\n{'='*60}")
            print(f"📂 从第 {resume_from + 1} 个样本继续训练...")
            print(f"   重新加载已有的规则表...")
            print(f"{'='*60}\n")
            self.similar_intent_rules = self._load_similar_intent_rules()

        # 初始化batch相关变量
        self.batch_sample_count = 0

        # 调试：验证batch_size
        print(f"\n{'='*60}")
        print(f"[TRAIN DEBUG] 训练开始前的batch_size: {self.batch_size}")
        print(f"{'='*60}\n")

        for idx, item in enumerate(samples_to_process):
            # 计算全局样本索引（考虑续训）
            global_idx = resume_from + idx
            print(f"\n{'='*60}")
            print(f"处理进度: {global_idx + 1}/{len(self.train_data)}")
            print(f"{'='*60}")

            # 获取真实标签（支持新数据格式）
            # 新格式：multimodal_intent_label (英文), multimodal_sentiment_label (数字)
            # 旧格式：intent (数字), sentiment (数字)

            # 优先使用新格式（英文标签）
            if 'multimodal_intent_label' in item:
                true_intent = item.get('multimodal_intent_label', '')  # 英文标签
            else:
                # 兼容旧格式（数字ID）
                true_intent_id = item.get('intent', -1)
                true_intent = INTENT_ID_TO_EN.get(true_intent_id, '')

            # 获取情感标签
            if 'multimodal_sentiment_label' in item:
                true_sentiment = item.get('multimodal_sentiment_label', -1)
            else:
                true_sentiment = item.get('sentiment', -1)

            # 验证标签有效性
            if not true_intent or true_sentiment == -1:
                print(f"⚠ 缺少真实标签 (intent={true_intent}, sentiment={true_sentiment})，跳过此样本")
                continue

            # 调用learner
            print("📚 调用 Learner...")
            model_response, pred_intent, pred_sentiment = self.call_learner(item)

            if not model_response:
                print("❌ Learner 返回空响应")
                continue

            # 保存Learner响应到json（包含完整的样本信息）
            self._save_agent_response_json(
                idx + 1, "learner", model_response,
                item=item,
                pred_intent=pred_intent,
                pred_sentiment=pred_sentiment,
                true_intent=true_intent,
                true_sentiment=true_sentiment
            )

            # 判断是否正确
            is_correct = (pred_intent == true_intent)

            if is_correct:
                print("✅ 预测正确！")
                self.training_stats["correct_count"] += 1
            else:
                print("❌ 预测错误！")
                print(f"   预测: 意图={pred_intent}, 情感={pred_sentiment}")
                print(f"   真实: 意图={true_intent}, 情感={true_sentiment}")
                self.training_stats["error_count"] += 1

                # 将失败样本添加到当前batch的错误列表
                self.batch_errors.append({
                    'item': item,
                    'model_response': model_response,
                    'pred_intent': pred_intent,
                    'pred_sentiment': pred_sentiment,
                    'true_intent': true_intent,
                    'true_sentiment': true_sentiment
                })

            # 增加batch计数
            self.batch_sample_count += 1

            # 调试：打印batch状态
            print(f"[DEBUG] batch_size={self.batch_size}, batch_sample_count={self.batch_sample_count}, is_correct={is_correct}")
            print(f"[DEBUG] 条件检查 -> batch_size > 1? {self.batch_size > 1}  |  batch_sample_count >= batch_size? {self.batch_sample_count >= self.batch_size}")

            # 当batch满时，处理错误样本并刷新到主表
            if self.batch_size > 1 and self.batch_sample_count >= self.batch_size:
                print(f"[DEBUG] ✓ 进入批处理分支")
                print(f"\n{'='*60}")
                print(f"🔄 Batch 已满({self.batch_sample_count}个样本)，准备处理...")
                print(f"{'='*60}")

                # 如果这个batch内有失败样本，调用optimizer和全局regularizer
                if self.batch_errors:
                    print(f"📊 发现 {len(self.batch_errors)} 个失败样本，调用Optimizer...")
                    optimizer_response = self.call_optimizer(self.batch_errors)

                    # 保存Optimizer响应到json
                    if optimizer_response:
                        batch_json_file = f"agent_responses/batch_{self.batch_count + 1}_optimizer.json"
                        try:
                            os.makedirs("agent_responses", exist_ok=True)
                            response_data = {
                                "batch": self.batch_count + 1,
                                "error_samples_count": len(self.batch_errors),
                                "type": "optimizer",
                                "optimizer_response": optimizer_response,
                                "timestamp": time.time()
                            }
                            with open(batch_json_file, 'a', encoding='utf-8') as f:
                                json.dump(response_data, f, ensure_ascii=False, indent=2)
                            print(f"✓ Optimizer响应已保存: {batch_json_file}")
                        except Exception as e:
                            print(f"⚠️  保存Optimizer响应失败: {e}")

                    # 从Optimizer响应中提取表内容，更新内存变量和ablation文件
                    if optimizer_response:
                        print("📊 从Optimizer响应中提取表内容...")
                        self._update_tables_from_response(optimizer_response, "optimizer")

                    # 每次Optimizer后立即调用全局Regularizer
                    if optimizer_response:
                        print("🌍 每次Optimizer后，立即调用全局Regularizer进行优化...")
                        global_regularizer_response = self.call_regularizer(is_global=True)

                        # 处理全局Regularizer的响应
                        if global_regularizer_response:
                            print("📊 从全局Regularizer响应中提取表内容...")
                            self._update_tables_from_response(global_regularizer_response, "regularizer_global")

                            # 保存全局Regularizer响应到json
                            batch_json_file = f"agent_responses/batch_{self.batch_count + 1}_global_regularizer.json"
                            try:
                                os.makedirs("agent_responses", exist_ok=True)
                                response_data = {
                                    "batch": self.batch_count + 1,
                                    "error_samples_count": len(self.batch_errors),
                                    "type": "global_regularizer",
                                    "regularizer_response": global_regularizer_response,
                                    "timestamp": time.time()
                                }
                                with open(batch_json_file, 'a', encoding='utf-8') as f:
                                    json.dump(response_data, f, ensure_ascii=False, indent=2)
                                print(f"✓ 全局Regularizer响应已保存: {batch_json_file}")
                            except Exception as e:
                                print(f"⚠️  保存全局Regularizer响应失败: {e}")

                            # Regularizer后调用Cleaner进行进一步清理
                            cleaner_response = self.call_cleaner()

                            # 处理Cleaner的响应
                            if cleaner_response:
                                print("📊 从Cleaner响应中提取表内容...")
                                self._update_tables_from_response(cleaner_response, "cleaner")

                                # 保存Cleaner响应到json
                                batch_json_file = f"agent_responses/batch_{self.batch_count + 1}_cleaner.json"
                                try:
                                    os.makedirs("agent_responses", exist_ok=True)
                                    response_data = {
                                        "batch": self.batch_count + 1,
                                        "error_samples_count": len(self.batch_errors),
                                        "type": "cleaner",
                                        "cleaner_response": cleaner_response,
                                        "timestamp": time.time()
                                    }
                                    with open(batch_json_file, 'a', encoding='utf-8') as f:
                                        json.dump(response_data, f, ensure_ascii=False, indent=2)
                                    print(f"✓ Cleaner响应已保存: {batch_json_file}")
                                except Exception as e:
                                    print(f"⚠️  保存Cleaner响应失败: {e}")

                    # 清空错误列表
                    self.batch_errors = []
                else:
                    print(f"✓ 这个Batch中所有样本都预测正确，跳过Optimizer和Regularizer")

                self.batch_sample_count = 0

            # batch_size == 1时的原有逻辑
            elif self.batch_size == 1 and not is_correct:
                print(f"[DEBUG] ✓ 进入逐样本处理分支")
                print(f"🔧 调用 Optimizer...")
                optimizer_response = self.call_optimizer(
                    [{
                        'item': item,
                        'model_response': model_response,
                        'pred_intent': pred_intent,
                        'pred_sentiment': pred_sentiment,
                        'true_intent': true_intent,
                        'true_sentiment': true_sentiment
                    }]
                )

                # 保存Optimizer响应到json（包含完整的样本信息）
                if optimizer_response:
                    self._save_agent_response_json(
                        idx + 1, "optimizer", optimizer_response,
                        item=item,
                        pred_intent=pred_intent,
                        pred_sentiment=pred_sentiment,
                        true_intent=true_intent,
                        true_sentiment=true_sentiment
                    )

                if optimizer_response:
                    # 从Optimizer响应中提取表内容，更新内存变量和ablation文件
                    print("📊 从Optimizer响应中提取表内容...")
                    self._update_tables_from_response(optimizer_response, "optimizer")

                    # 每次Optimizer后立即调用全局Regularizer
                    print("🌍 每次Optimizer后，立即调用全局Regularizer进行优化...")
                    global_regularizer_response = self.call_regularizer(is_global=True)

                    # 保存全局Regularizer响应到json（包含完整的样本信息）
                    if global_regularizer_response:
                        self._save_agent_response_json(
                            idx + 1, "global_regularizer", global_regularizer_response,
                            item=item,
                            pred_intent=pred_intent,
                            pred_sentiment=pred_sentiment,
                            true_intent=true_intent,
                            true_sentiment=true_sentiment
                        )

                        # 从全局Regularizer响应中提取表内容
                        print("📊 从全局Regularizer响应中提取表内容...")
                        self._update_tables_from_response(global_regularizer_response, "regularizer_global")

                        # Regularizer后调用Cleaner进行进一步清理
                        cleaner_response = self.call_cleaner()

                        # 保存Cleaner响应到json（包含完整的样本信息）
                        if cleaner_response:
                            self._save_agent_response_json(
                                idx + 1, "cleaner", cleaner_response,
                                item=item,
                                pred_intent=pred_intent,
                                pred_sentiment=pred_sentiment,
                                true_intent=true_intent,
                                true_sentiment=true_sentiment
                            )

                            # 从Cleaner响应中提取表内容
                            print("📊 从Cleaner响应中提取表内容...")
                            self._update_tables_from_response(cleaner_response, "cleaner")

            # 定期保存检查点
            # 无论batch_size多少，都是每save_interval个样本保存一次
            # 注：全局Regularizer已在每次Optimizer后调用过，此处只保存checkpoint文件
            should_checkpoint = (global_idx + 1) % save_interval == 0

            if should_checkpoint:
                print(f"\n{'='*60}")
                print(f"💾 第 {global_idx + 1} 个样本检查点保存...")
                print(f"{'='*60}")

                # 保存检查点（基于已通过全局Regularizer优化的表）
                self._save_feature_and_rules(global_idx + 1)
                print(f"✅ 第 {global_idx + 1} 步检查点完成")

            # 添加延迟避免API限制
            time.sleep(1)

        # 如果还有未刷新的batch，处理并刷新它
        if self.batch_size > 1 and self.batch_sample_count > 0:
            print(f"\n{'='*60}")
            print(f"🔄 训练结束，处理剩余的 {self.batch_sample_count} 个样本...")
            print(f"{'='*60}")

            # 处理剩余batch中的错误样本
            if self.batch_errors:
                print(f"📊 发现 {len(self.batch_errors)} 个失败样本，调用Optimizer...")
                optimizer_response = self.call_optimizer(self.batch_errors)

                if optimizer_response:
                    # 从Optimizer响应中提取表内容
                    print("📊 从Optimizer响应中提取表内容...")
                    self._update_tables_from_response(optimizer_response, "optimizer")

                    # 每次Optimizer后立即调用全局Regularizer
                    print("🌍 每次Optimizer后，立即调用全局Regularizer进行优化...")
                    global_regularizer_response = self.call_regularizer(is_global=True)

                    # 从全局Regularizer响应中提取表内容
                    if global_regularizer_response:
                        print("📊 从全局Regularizer响应中提取表内容...")
                        self._update_tables_from_response(global_regularizer_response, "regularizer_global")

                        # Regularizer后调用Cleaner进行进一步清理
                        cleaner_response = self.call_cleaner()

                        # 从Cleaner响应中提取表内容
                        if cleaner_response:
                            print("📊 从Cleaner响应中提取表内容...")
                            self._update_tables_from_response(cleaner_response, "cleaner")

                self.batch_errors = []
            else:
                print(f"✓ 剩余Batch中所有样本都预测正确，跳过Optimizer和Regularizer")

        # 最终保存
        print(f"\n💾 保存最终的表格...")
        self._save_feature_and_rules("final")

        # 打印训练统计
        self._print_training_stats()

    def _print_training_stats(self):
        """打印训练统计信息"""
        print(f"\n{'='*60}")
        print("🎓 训练完成！统计信息：")
        print(f"{'='*60}")
        print(f"总样本数: {self.training_stats['total_samples']}")
        print(f"正确预测: {self.training_stats['correct_count']}")
        print(f"错误预测: {self.training_stats['error_count']}")
        print(f"Optimizer 调用次数: {self.training_stats['optimizer_calls']}")
        print(f"Regularizer 调用次数: {self.training_stats['regularizer_calls']}")
        print(f"Cleaner 调用次数: {self.training_stats['cleaner_calls']}")

        if self.training_stats['total_samples'] > 0:
            accuracy = self.training_stats['correct_count'] / self.training_stats['total_samples']
            print(f"准确率: {accuracy:.2%}")
        print(f"{'='*60}\n")


if __name__ == "__main__":
    # 配置参数
    TRAIN_JSON_PATH = "train.json"
    BASE_IMAGE_PATH = "/root/autodl-fs/stickers"
    BATCH_SIZE = 2  # ← 在这里调整 batch_size（1表示逐样本，>1表示批处理）

    # 调试：确认main部分变量
    print(f"\n{'='*60}")
    print(f"[MAIN DEBUG] BATCH_SIZE 已设置为: {BATCH_SIZE}")
    print(f"[MAIN DEBUG] 准备传递给 LearningTrainingSystem: batch_size={BATCH_SIZE}")
    print(f"{'='*60}\n")

    # 初始化训练系统
    print("🚀 初始化学习训练系统...")
    system = LearningTrainingSystem(TRAIN_JSON_PATH, BASE_IMAGE_PATH, batch_size=BATCH_SIZE)

    # 开始训练
    print("\n🎓 开始训练...")
    system.train(save_interval=20, resume_from = 0)

    print("\n✅ 学习训练系统执行完成！")