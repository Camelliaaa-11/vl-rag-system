import json
import logging
import os
from typing import Any, Dict, List, Optional

import requests

from config import Config
from rag.retriever import MuseumRetriever

logger = logging.getLogger("LLM")

try:
    import ollama  # type: ignore
except Exception:
    ollama = None


class LLMService:
    """语言系统 - 负责 RAG 注入、提示词管理与大模型调用。"""

    def __init__(self, model_name: str = Config.LLM_MODEL_NAME):
        self.provider = Config.LLM_PROVIDER.lower().strip() or "deepseek"
        self.model_name = model_name
        self.retriever = MuseumRetriever()
        self.prompt_templates: Dict[str, str] = {}
        self.model_status = "ready"
        logger.info(
            "🧠 [INIT] 初始化 LLM 服务: provider=%s, model=%s",
            self.provider,
            self.model_name,
        )
        self._load_prompt_templates()

    def _load_prompt_templates(self):
        """加载 prompts 目录下的模板文件。"""
        try:
            for filename in os.listdir(Config.PROMPTS_DIR):
                if filename.endswith(".md"):
                    template_name = filename.replace(".md", "")
                    self.prompt_templates[template_name] = self._load_prompt(filename)
            logger.info("📝 [PROMPT] 提示词模板加载完成: %s", list(self.prompt_templates.keys()))
        except Exception as exc:
            logger.warning("⚠️ [PROMPT] 加载提示词模板失败: %s", exc)

    def _load_prompt(self, filename: str, **kwargs) -> str:
        """从 prompts 目录加载并格式化提示词。"""
        path = Config.PROMPTS_DIR / filename
        try:
            with open(path, "r", encoding="utf-8") as file:
                template = file.read()
            if kwargs:
                return template.format(**kwargs)
            return template
        except Exception as exc:
            logger.warning("⚠️ [PROMPT] 无法加载提示词 %s: %s", filename, exc)
            if "system" in filename:
                return "你是一个导览机器人。"
            return kwargs.get("question", "请分析这张图片")

    def _build_messages(
        self,
        prompts: List[str],
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """检索上下文并组装最终消息。"""
        question = " ".join(prompt.strip() for prompt in prompts if prompt and prompt.strip())
        if not question:
            question = "请介绍这个展品。"

        system_prompt = self.prompt_templates.get("system_prompt", "你是一个导览机器人。")
        intent_info = self._analyze_intent(question, history)
        topic_type = intent_info["topic_type"]
        topic_subject = intent_info["topic_subject"]
        retrieval_top_k = intent_info.get("retrieval_top_k", Config.RETRIEVAL_TOP_K)

        retrieval_query = ""
        context = ""
        prompt_body = ""

        if topic_type in {"knowledge", "knowledge_followup", "knowledge_recommendation"}:
            retrieval_query = self._build_retrieval_query(question, history, intent_info)
            logger.info("🔍 [RAG] 检索问题: %s", retrieval_query)
            context = self.retriever.retrieve(retrieval_query, top_k=retrieval_top_k)
            if topic_type == "knowledge_recommendation":
                prompt_body = self._build_recommendation_prompt(question, context, topic_subject)
            else:
                prompt_body = self._load_prompt(
                    "identify_prompt.md",
                    context=context,
                    question=question,
                )
        else:
            prompt_body = self._build_smalltalk_prompt(question, history, topic_subject)

        full_prompt = question
        if history:
            recent_history = history[-4:]
            history_text = "\n".join(
                f"{item.get('role', 'user')}: {item.get('content', '')}" for item in recent_history
            )
            full_prompt = f"对话历史：\n{history_text}\n\n当前问题：{question}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt_body + "\n\n" + full_prompt},
        ]
        return {
            "question": question,
            "topic_type": topic_type,
            "topic_subject": topic_subject,
            "retrieval_top_k": retrieval_top_k,
            "retrieval_query": retrieval_query,
            "context": context,
            "system_prompt": system_prompt,
            "messages": messages,
        }

    def _analyze_intent(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, str]:
        """轻量意图分析：区分知识问答、知识追问、闲聊。"""
        normalized_question = question.strip()
        previous_user = ""
        previous_assistant = ""
        if history:
            user_msgs = [
                item.get("content", "").strip()
                for item in history[-6:]
                if item.get("role") == "user" and item.get("content", "").strip()
            ]
            assistant_msgs = [
                item.get("content", "").strip()
                for item in history[-6:]
                if item.get("role") == "assistant" and item.get("content", "").strip()
            ]
            previous_user = user_msgs[-1] if user_msgs else ""
            previous_assistant = assistant_msgs[-1] if assistant_msgs else ""

        smalltalk_keywords = [
            "你好", "你是谁", "你叫什么", "谢谢", "再见", "哈哈", "真有意思", "你会不会",
            "你觉得", "你声音", "你真", "辛苦", "累不累", "聊天", "陪我", "心情", "天气",
        ]
        summary_keywords = [
            "总结", "回顾", "梳理", "概括", "汇总", "整理一下", "总结一下", "回顾一下",
            "总结我们", "目前的谈话", "刚才聊了什么", "用列表", "一一总结",
        ]
        recommendation_keywords = [
            "介绍一个", "推荐一个", "随便介绍", "随便来一个", "挑一个", "选一个",
            "给我讲一个", "来一个展品", "说一个展品", "讲一个展品",
        ]
        knowledge_keywords = [
            "作品", "展品", "作者", "设计", "展区", "灵感", "动机", "技术", "特点",
            "背景", "介绍", "是什么", "为什么", "怎么实现", "材料", "创作", "理念",
        ]
        follow_up_markers = [
            "它", "这个", "这件", "那个", "那它", "那这个", "继续", "再说", "再讲",
            "详细", "具体", "展开", "补充", "还有", "然后", "为什么", "怎么", "再",
        ]

        looks_smalltalk = any(keyword in normalized_question for keyword in smalltalk_keywords)
        looks_summary = any(keyword in normalized_question for keyword in summary_keywords)
        looks_recommendation = any(keyword in normalized_question for keyword in recommendation_keywords)
        looks_knowledge = any(keyword in normalized_question for keyword in knowledge_keywords)
        looks_follow_up = any(marker in normalized_question for marker in follow_up_markers)
        short_question = len(normalized_question) <= 18

        previous_topic_subject = self._extract_topic_subject(previous_user or previous_assistant)

        recommendation_subject = self._extract_recommendation_preference(normalized_question)
        explicit_subject = self._extract_topic_subject(normalized_question)
        if previous_topic_subject and self._is_generic_followup_question(normalized_question):
            explicit_subject = previous_topic_subject

        if looks_summary:
            return {
                "topic_type": "smalltalk",
                "topic_subject": "会话总结",
                "retrieval_top_k": 0,
            }

        if looks_recommendation:
            return {
                "topic_type": "knowledge_recommendation",
                "topic_subject": recommendation_subject or "展品推荐",
                "retrieval_top_k": 5,
            }

        if looks_smalltalk and not looks_knowledge:
            return {
                "topic_type": "smalltalk",
                "topic_subject": previous_topic_subject or "机器人交流",
                "retrieval_top_k": 0,
            }

        if looks_knowledge:
            return {
                "topic_type": "knowledge",
                "topic_subject": explicit_subject or previous_topic_subject,
                "retrieval_top_k": Config.RETRIEVAL_TOP_K,
            }

        if explicit_subject and previous_topic_subject and explicit_subject != previous_topic_subject:
            return {
                "topic_type": "knowledge",
                "topic_subject": explicit_subject,
                "retrieval_top_k": Config.RETRIEVAL_TOP_K,
            }

        if short_question and looks_follow_up and previous_topic_subject:
            return {
                "topic_type": "knowledge_followup",
                "topic_subject": previous_topic_subject,
                "retrieval_top_k": Config.RETRIEVAL_TOP_K,
            }

        if previous_topic_subject and (looks_follow_up or short_question):
            return {
                "topic_type": "knowledge_followup",
                "topic_subject": previous_topic_subject,
                "retrieval_top_k": Config.RETRIEVAL_TOP_K,
            }

        return {
            "topic_type": "smalltalk",
            "topic_subject": previous_topic_subject or "机器人交流",
            "retrieval_top_k": 0,
        }

    def _extract_topic_subject(self, text: str) -> str:
        """从问题或回答中提取一个粗粒度主题对象。"""
        normalized = (text or "").strip()
        if not normalized:
            return ""

        quote_pairs = [("《", "》"), ("“", "”"), ("\"", "\"")]
        for left, right in quote_pairs:
            if left in normalized and right in normalized:
                start = normalized.find(left)
                end = normalized.find(right, start + 1)
                if start != -1 and end != -1 and end > start + 1:
                    return normalized[start + 1:end].strip()

        alias_markers = ["灵视", "永栖所", "bytebunny", "萨曼鼓", "萨满鼓", "虚拟偶像"]
        for marker in alias_markers:
            if marker.lower() in normalized.lower():
                return marker

        markers = ["作品", "展品", "作者", "设计", "展区", "技术", "灵感", "动机"]
        if any(marker in normalized for marker in markers):
            return normalized[:24]

        return ""

    def _is_generic_followup_question(self, text: str) -> bool:
        """判断是否为依赖上下文的泛化追问。"""
        normalized = (text or "").strip()
        if not normalized:
            return False

        quote_pairs = [("《", "》"), ("“", "”"), ("\"", "\"")]
        for left, right in quote_pairs:
            if left in normalized and right in normalized:
                return False

        alias_markers = ["灵视", "永栖所", "bytebunny", "萨曼鼓", "萨满鼓", "虚拟偶像"]
        if any(marker.lower() in normalized.lower() for marker in alias_markers):
            return False

        generic_markers = [
            "这个", "这件", "它", "那个", "那件", "那它",
            "有什么", "是什么", "为什么", "怎么", "背景", "理念", "技术", "作者",
        ]
        return len(normalized) <= 24 and any(marker in normalized for marker in generic_markers)

    def _extract_recommendation_preference(self, text: str) -> str:
        """从推荐式问题中提取偏好条件。"""
        normalized = (text or "").strip()
        if not normalized:
            return ""

        preference_keywords = [
            "实用", "互动", "传统文化", "环保", "科技", "未来", "儿童", "空间",
            "材料", "交互", "可持续", "工业设计", "艺术与科技", "环境设计",
        ]
        matched = [keyword for keyword in preference_keywords if keyword in normalized]
        if matched:
            return "、".join(matched)
        return ""

    def _build_retrieval_query(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        intent_info: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        为 RAG 构造更稳的检索 query。

        目标：
        - 当前问题足够完整时，直接用当前问题检索
        - 当前问题较短、包含代词或明显追问语气时，拼接最近主题
        """
        normalized_question = question.strip()
        if not history:
            return normalized_question

        recent_history = [item for item in history[-6:] if item.get("content")]
        recent_user_questions = [
            item["content"].strip()
            for item in recent_history
            if item.get("role") == "user" and item.get("content", "").strip()
        ]
        recent_assistant_answers = [
            item["content"].strip()
            for item in recent_history
            if item.get("role") == "assistant" and item.get("content", "").strip()
        ]

        previous_user = recent_user_questions[-1] if recent_user_questions else ""
        previous_assistant = recent_assistant_answers[-1] if recent_assistant_answers else ""

        follow_up_markers = [
            "它", "这个", "这个作品", "这件作品", "这个展品", "那个", "那它", "那这个",
            "继续", "再说", "再讲", "详细", "具体", "展开", "补充", "还有", "然后",
            "为什么", "怎么", "那", "再", "请继续", "请具体说说",
        ]
        question_is_short = len(normalized_question) <= 18
        looks_like_follow_up = any(marker in normalized_question for marker in follow_up_markers)

        topic_subject = (intent_info or {}).get("topic_subject", "").strip()

        if (intent_info or {}).get("topic_type") == "knowledge_recommendation":
            recommendation_parts: List[str] = []
            if topic_subject and topic_subject != "展品推荐":
                recommendation_parts.append(f"筛选偏好：{topic_subject}")
            recommendation_parts.append(f"用户需求：{normalized_question}")
            recommendation_parts.append("请优先召回最符合偏好、适合被介绍的一件或几件展品")
            return "\n".join(recommendation_parts)

        if not question_is_short and not looks_like_follow_up and not topic_subject:
            return normalized_question

        topic_parts: List[str] = []
        if topic_subject:
            topic_parts.append(f"当前话题对象：{topic_subject}")
        if previous_user and previous_user != normalized_question:
            topic_parts.append(f"上一轮用户问题：{previous_user}")
        if previous_assistant:
            topic_parts.append(f"上一轮回答摘要：{previous_assistant[:120]}")
        topic_parts.append(f"当前追问：{normalized_question}")

        retrieval_query = "\n".join(topic_parts).strip()
        return retrieval_query or normalized_question

    def _build_recommendation_prompt(
        self,
        question: str,
        context: str,
        topic_subject: str = "",
    ) -> str:
        """推荐/挑选型知识问答提示词，读取 prompts/recommendation_prompt.md。"""
        return self._load_prompt(
            "recommendation_prompt.md",
            topic_subject=topic_subject or "无明确偏好",
            context=context or "无可用参考信息",
            question=question,
        )

    def _build_smalltalk_prompt(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
        topic_subject: str = "",
    ) -> str:
        """闲聊场景下不走 RAG，读取 prompts/smalltalk_prompt.md。"""
        history_hint = ""
        if history:
            recent = history[-4:]
            history_hint = "\n".join(
                f"{item.get('role', 'user')}: {item.get('content', '')}" for item in recent
            )

        return self._load_prompt(
            "smalltalk_prompt.md",
            topic_subject=topic_subject or "无",
            history_hint=history_hint or "无",
            question=question,
        )

    def _call_deepseek(self, messages: List[Dict[str, str]]) -> str:
        """通过 DeepSeek Chat Completions API 获取回答。"""
        if not Config.DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY 未配置，无法调用 DeepSeek API")

        url = Config.DEEPSEEK_BASE_URL.rstrip("/") + Config.DEEPSEEK_CHAT_PATH
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {Config.DEEPSEEK_API_KEY}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            url,
            headers=headers,
            data=json.dumps(payload),
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"DeepSeek 返回内容异常: {data}")

        message = choices[0].get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        if not content:
            raise RuntimeError("DeepSeek 返回空内容")
        return content

    def _call_ollama(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """可选的本地 Ollama 回退。"""
        if ollama is None:
            raise RuntimeError("ollama 未安装或不可用")

        response = ollama.chat(model=Config.OLLAMA_MODEL_NAME, messages=messages, stream=False)
        message = response.get("message", {})
        content = message.get("content", "")
        if content:
            return content

        fallback = ollama.generate(
            model=Config.OLLAMA_MODEL_NAME,
            system=system_prompt,
            prompt=messages[-1]["content"],
            stream=False,
        )
        content = fallback.get("response", "")
        if not content:
            raise RuntimeError("Ollama 返回空内容")
        return content

    def _generate_answer(
        self,
        prompts: List[str],
        history: Optional[List[Dict[str, str]]] = None,
        image_data: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        统一生成入口。

        当前主路径只做文本问答，image_data 仅保留接口兼容。
        """
        del image_data
        prepared = self._build_messages(prompts, history)
        self.model_status = "generating"

        try:
            if self.provider == "deepseek":
                logger.info("🤖 [LLM] 调用 DeepSeek API")
                answer = self._call_deepseek(prepared["messages"])
            elif self.provider == "ollama":
                logger.info("🤖 [LLM] 调用 Ollama")
                answer = self._call_ollama(prepared["messages"], prepared["system_prompt"])
            else:
                raise RuntimeError(f"不支持的 LLM_PROVIDER: {self.provider}")

            self.model_status = "ready"
            return {
                "answer": answer,
                "context": prepared["context"],
                "question": prepared["question"],
                "topic_type": prepared["topic_type"],
                "topic_subject": prepared["topic_subject"],
                "retrieval_query": prepared["retrieval_query"],
                "provider": self.provider,
                "model_name": self.model_name,
                "success": True,
            }
        except Exception as exc:
            logger.error("❌ [LLM] 模型调用失败: %s", exc)
            self.model_status = "error"

            if self.provider != "ollama" and ollama is not None:
                try:
                    logger.warning("🔄 [LLM FALLBACK] DeepSeek 失败，尝试 Ollama 回退")
                    answer = self._call_ollama(prepared["messages"], prepared["system_prompt"])
                    self.model_status = "ready"
                    return {
                        "answer": answer,
                        "context": prepared["context"],
                        "question": prepared["question"],
                        "topic_type": prepared["topic_type"],
                        "topic_subject": prepared["topic_subject"],
                        "retrieval_query": prepared["retrieval_query"],
                        "provider": "ollama_fallback",
                        "model_name": Config.OLLAMA_MODEL_NAME,
                        "success": True,
                    }
                except Exception as fallback_exc:
                    logger.error("❌ [LLM] Ollama 回退失败: %s", fallback_exc)

            self.model_status = "ready"
            return {
                "answer": "当前模型服务不可用，请稍后重试。",
                "context": prepared["context"],
                "question": prepared["question"],
                "topic_type": prepared["topic_type"],
                "topic_subject": prepared["topic_subject"],
                "retrieval_query": prepared["retrieval_query"],
                "provider": self.provider,
                "model_name": self.model_name,
                "success": False,
                "error": str(exc),
            }

    def generate_stream(
        self,
        prompts: List[str],
        history: Optional[List[Dict[str, str]]] = None,
        image_data: Optional[bytes] = None,
    ):
        """兼容旧接口，内部先获取完整回答，再逐字符输出。"""
        result = self._generate_answer(prompts, history, image_data)
        for char in result["answer"]:
            yield char

    def generate_sync(
        self,
        prompts: List[str],
        image_data: Optional[bytes] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """同步生成文本响应。"""
        return self._generate_answer(prompts, history, image_data)

    def load_model(self, model_config: Dict[str, Any]):
        """更新模型配置。"""
        try:
            self.model_name = model_config.get("model_name", self.model_name)
            self.provider = model_config.get("provider", self.provider)
            logger.info("🔧 [LLM] 加载模型配置: provider=%s, model=%s", self.provider, self.model_name)
            self.model_status = "ready"
            return True
        except Exception as exc:
            logger.error("❌ [LLM] 模型配置更新失败: %s", exc)
            self.model_status = "error"
            return False

    def get_model_status(self):
        """获取当前模型状态。"""
        return {
            "status": self.model_status,
            "provider": self.provider,
            "model_name": self.model_name,
            "resources": {
                "memory_usage": "N/A",
                "cpu_usage": "N/A",
            },
        }

    def update_prompt_template(self, template_name: str, content: str):
        """更新提示词模板。"""
        try:
            self.prompt_templates[template_name] = content
            template_path = Config.PROMPTS_DIR / f"{template_name}.md"
            with open(template_path, "w", encoding="utf-8") as file:
                file.write(content)
            logger.info("🔄 [PROMPT] 提示词模板更新成功: %s", template_name)
            return True
        except Exception as exc:
            logger.error("❌ [PROMPT] 提示词模板更新失败: %s", exc)
            return False

    def inject_context(self, context_data: Dict[str, Any]):
        """保留的兼容接口。"""
        logger.info("📚 [LLM] 注入背景知识: keys=%s", list(context_data.keys()))
        return True

    def generate_response_stream(
        self,
        image_data: bytes,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """兼容旧接口。"""
        return self.generate_stream([question], history, image_data)

    def generate_response_sync(
        self,
        image_data: bytes,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """兼容旧接口。"""
        return self.generate_sync([question], image_data, history)
