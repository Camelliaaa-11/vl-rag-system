import os
import ollama
import logging
from rag.retriever import MuseumRetriever
from config import Config

logger = logging.getLogger("LLM")

class LLMService:
    def __init__(self, model_name: str = Config.LLM_MODEL_NAME):
        logger.info("🔥 [INIT] 正在初始化 LLM 服务 (Qwen-VL)...")
        self.retriever = MuseumRetriever()
        self.model_name = model_name

    def _load_prompt(self, filename: str, **kwargs):
        """从 prompts 目录加载并格式化提示词"""
        path = Config.PROMPTS_DIR / filename
        try:
            with open(path, "r", encoding="utf-8") as f:
                template = f.read()
            return template.format(**kwargs)
        except Exception as e:
            logger.warning("⚠️ [PROMPT] 无法加载提示词 %s: %s", filename, e)
            if "system" in filename:
                return "你是一个导览机器人。"
            return kwargs.get('question', '请分析这张图片')

    def generate_response_stream(self, image_data: bytes, question: str, history: list = None):
        """
        整合了 RAG、历史记录文本化与异常回退机制的流式接口
        """
        # 1. RAG 检索
        logger.info("🔍 [RAG] 正在检索知识库: %s", question)
        context = self.retriever.retrieve(question, top_k=Config.RETRIEVAL_TOP_K)
        
        # 2. 加载双提示词
        system_prompt = self._load_prompt("system_prompt.md")
        identify_prompt = self._load_prompt("identify_prompt.md", context=context, question=question)

        # 3. 对话历史处理 (参考 qwen_vl.py 逻辑：将记录转化为文本块注入)
        full_prompt = question
        if history:
            # 仅取最近 4 条记录以防超出模型窗口
            history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history[-4:]])
            full_prompt = f"对话历史：\n{history_text}\n\n当前问题：{question}"

        # 4. 组装输入 (暂不考虑图片传输逻辑的精细化，保持现有钩子)
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': identify_prompt + "\n\n" + full_prompt}
        ]

        # 5. 执行调用与回退逻辑
        try:
            logger.info("🤖 [LLM] 开始请求流式响应 (Chat 模式)...")
            response = ollama.chat(model=self.model_name, messages=messages, stream=True)
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    if content:
                        yield content
        except Exception as e:
            logger.warning("🔄 [LLM FALLBACK] 流式调用失败，尝试非流式生成模式: %s", e)
            try:
                # 尝试回退到 generate 接口
                fallback_response = ollama.generate(
                    model=self.model_name,
                    system=system_prompt,
                    prompt=identify_prompt + "\n\n" + full_prompt,
                    stream=False
                )
                answer = fallback_response.get('response', '')
                if answer:
                    for char in answer:
                        yield char
            except Exception as e2:
                logger.error("❌ [LLM CRITICAL ERROR] 连回退模式也失败了: %s", e2)
                yield "哎呀，信号闪断了，您可以再试一次吗？"

    def generate_response_sync(self, image_data: bytes, question: str, history: list = None):
        """同步响应接口"""
        full_answer = ""
        for chunk in self.generate_response_stream(image_data, question, history):
            full_answer += chunk
        return {"answer": full_answer, "success": True}
