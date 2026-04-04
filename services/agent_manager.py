from typing import Dict, Any, Optional
from agents.intro_agent import IntroAgent
from agents.chat_agent import ChatAgent
from agents.smalltalk_agent import SmallTalkAgent

class AgentManager:
    """
    负责根据交互场景（Context）路由并选择最合适的 Agent。
    """
    def __init__(self):
        self.agents = {
            "intro": IntroAgent(),
            "chat": ChatAgent(),
            "smalltalk": SmallTalkAgent()
        }

    def select_agent(self, query: str, context: Optional[Dict[str, Any]] = None):
        """
        根据 query 和 context 分发任务给合适的 Agent。
        """
        # 这里的路由逻辑可以根据“关键词识别”、“视觉环境属性”或“意图分类模型”来实现。
        # 临时演示：简单逻辑
        if "介绍" in query or "看" in query:
            return self.agents["intro"]
        elif len(query) > 10: # 假设长句子是深度对话
            return self.agents["chat"]
        else:
            return self.agents["smalltalk"]
