import os
import ollama
import logging
import requests
from typing import List, Dict, Any, Optional
from rag.retriever import MuseumRetriever
from config import Config

logger = logging.getLogger("LLM")

class LLMService:
    """语言系统 - 负责DeepSeek大模型的底层调用、流式输出管理及提示词注入"""
    def __init__(self, model_name: str = Config.LLM_MODEL_NAME):
        logger.info("🧠 [INIT] 正在初始化 LLM 服务 (DeepSeek)...")
        self.retriever = MuseumRetriever()
        self.model_name = model_name
        self.prompt_templates = {}
        self.model_status = "ready"
        self._load_prompt_templates()
    
    def _load_prompt_templates(self):
        """加载提示词模板"""
        try:
            prompt_files = os.listdir(Config.PROMPTS_DIR)
            for file in prompt_files:
                if file.endswith('.md'):
                    template_name = file.replace('.md', '')
                    self.prompt_templates[template_name] = self._load_prompt(file)
            logger.info("📝 [PROMPT] 提示词模板加载完成: %s", list(self.prompt_templates.keys()))
        except Exception as e:
            logger.warning("⚠️ [PROMPT] 加载提示词模板失败: %s", e)
    
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
    
    def generate_stream(self, prompts: List[str], history: Optional[List[Dict[str, str]]] = None, image_data: Optional[bytes] = None):
        """
        流式生成文本响应，支持多模态输入
        """
        # 1. RAG 检索
        question = " ".join(prompts)
        logger.info("🔍 [RAG] 正在检索知识库: %s", question)
        context = self.retriever.retrieve(question, top_k=Config.RETRIEVAL_TOP_K)
        
        # 2. 加载提示词
        system_prompt = self.prompt_templates.get("system_prompt", "你是一个导览机器人。")
        identify_prompt = self._load_prompt("identify_prompt.md", context=context, question=question)

        # 3. 对话历史处理
        full_prompt = question
        if history:
            # 仅取最近 4 条记录以防超出模型窗口
            history_text = "\n".join([f"{h['role']}: {h['content']}" for h in history[-4:]])
            full_prompt = f"对话历史：\n{history_text}\n\n当前问题：{question}"

        # 4. 组装输入
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': identify_prompt + "\n\n" + full_prompt}
        ]

        # 5. 执行调用与回退逻辑
        try:
            self.model_status = "generating"
            logger.info("🤖 [LLM] 开始请求流式响应 (Chat 模式)...")
            response = ollama.chat(model=self.model_name, messages=messages, stream=True)
            for chunk in response:
                if 'message' in chunk and 'content' in chunk['message']:
                    content = chunk['message']['content']
                    if content:
                        yield content
            self.model_status = "ready"
        except Exception as e:
            self.model_status = "error"
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
                self.model_status = "ready"
            except Exception as e2:
                logger.error("❌ [LLM CRITICAL ERROR] 连回退模式也失败了: %s", e2)
                yield "哎呀，信号闪断了，您可以再试一次吗？"
                self.model_status = "ready"
    
    def generate_sync(self, prompts: List[str], image_data: Optional[bytes] = None):
        """
        同步生成文本响应
        """
        full_answer = ""
        for chunk in self.generate_stream(prompts, None, image_data):
            full_answer += chunk
        return {"answer": full_answer, "success": True}
    
    def load_model(self, model_config: Dict[str, Any]):
        """
        加载指定配置的DeepSeek模型
        """
        try:
            self.model_name = model_config.get("model_name", self.model_name)
            logger.info("🔧 [LLM] 加载模型: %s", self.model_name)
            # 这里可以添加模型加载逻辑
            self.model_status = "ready"
            return True
        except Exception as e:
            logger.error("❌ [LLM] 模型加载失败: %s", e)
            self.model_status = "error"
            return False
    
    def get_model_status(self):
        """
        获取当前模型的状态与资源使用情况
        """
        return {
            "status": self.model_status,
            "model_name": self.model_name,
            "resources": {
                "memory_usage": "N/A",  # 实际项目中可以添加内存使用监控
                "cpu_usage": "N/A"     # 实际项目中可以添加CPU使用监控
            }
        }
    
    def update_prompt_template(self, template_name: str, content: str):
        """
        更新提示词模板
        """
        try:
            self.prompt_templates[template_name] = content
            # 保存到文件
            template_path = Config.PROMPTS_DIR / f"{template_name}.md"
            with open(template_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info("🔄 [PROMPT] 提示词模板更新成功: %s", template_name)
            return True
        except Exception as e:
            logger.error("❌ [PROMPT] 提示词模板更新失败: %s", e)
            return False
    
    def inject_context(self, context_data: Dict[str, Any]):
        """
        注入背景知识到提示词中
        """
        # 这里可以实现背景知识注入逻辑
        logger.info("📚 [LLM] 注入背景知识")
        return True

    def generate_response_stream(self, image_data: bytes, question: str, history: list = None):
        """
        兼容旧接口
        """
        return self.generate_stream([question], history, image_data)

    def generate_response_sync(self, image_data: bytes, question: str, history: list = None):
        """
        兼容旧接口
        """
        return self.generate_sync([question], image_data)

