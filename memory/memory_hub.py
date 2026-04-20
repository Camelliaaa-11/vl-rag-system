"""记忆总线 (Memory Hub)：对外唯一入口。"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .insight_archive import InsightArchive
from .insight_extractor import InsightExtractor, LLMCaller
from .models import ChatTurn, InsightEntry, UserGroupProfile
from .short_term_memory import ShortTermMemory
from .user_group_profiles import UserGroupProfiles

logger = logging.getLogger("Memory.Hub")


@dataclass
class RecallResult:
    """recall() 的统一返回结构。"""

    raw_history: List[Dict[str, str]] = field(default_factory=list)
    insights: List[InsightEntry] = field(default_factory=list)
    user_group: Optional[UserGroupProfile] = None
    combined_context: str = ""

    def to_dict(self) -> Dict:
        return {
            "raw_history": list(self.raw_history),
            "insights": [insight.dict() for insight in self.insights],
            "user_group": self.user_group.dict() if self.user_group else None,
            "combined_context": self.combined_context,
        }


class MemoryHub:
    """
    统一协调 ShortTermMemory / InsightArchive / UserGroupProfiles。

    使用方式：
        hub = MemoryHub()                                  # 只读 + 存储
        hub.attach_extractor(InsightExtractor(llm_caller)) # 可选: 启用内省

        hub.record_turn(session_id, "user", "这是什么作品？")
        hub.record_turn(session_id, "assistant", "这是《永栖所》……")

        result = hub.recall("《永栖所》的作者是谁", user_id="u1", session_id="s1")
        await hub.reflect_on_conversation(session_id="s1", user_id="u1")
    """

    def __init__(
        self,
        short_term: Optional[ShortTermMemory] = None,
        insight_archive: Optional[InsightArchive] = None,
        user_groups: Optional[UserGroupProfiles] = None,
        extractor: Optional[InsightExtractor] = None,
    ):
        self.short_term = short_term or ShortTermMemory()
        self.insights = insight_archive or InsightArchive()
        self.user_groups = user_groups or UserGroupProfiles()
        self.extractor: Optional[InsightExtractor] = extractor
        logger.info("🧠 [HUB] 初始化完成")

    def attach_extractor(
        self,
        extractor_or_caller,
        **extractor_kwargs,
    ) -> None:
        """允许传入 InsightExtractor，或直接传 LLMCaller 自动构造一个。"""
        if isinstance(extractor_or_caller, InsightExtractor):
            self.extractor = extractor_or_caller
        else:
            self.extractor = InsightExtractor(
                llm_caller=extractor_or_caller,
                **extractor_kwargs,
            )
        logger.info("🪞 [HUB] 已绑定 insight extractor")

    def record_turn(self, session_id: str, role: str, content: str) -> None:
        self.short_term.add_chat_history(session_id, role, content)

    def clear_session(self, session_id: str) -> bool:
        return self.short_term.clear_chat_history(session_id)

    def recall(
        self,
        query: str,
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        top_k: int = 3,
        user_features: Optional[Dict] = None,
        history_tail: int = 6,
    ) -> RecallResult:
        raw_history: List[Dict[str, str]] = []
        if session_id:
            raw_history = self.short_term.get_raw_history(session_id)[-history_tail:]

        insights = self.insights.search_by_text(
            query=query,
            top_k=top_k,
            user_id=user_id if user_id and user_id != "anonymous" else None,
        )

        group: Optional[UserGroupProfile] = None
        if user_features:
            group_id = self.user_groups.match_group(user_features)
            group = self.user_groups.get_group_config(group_id)
        else:
            group = self.user_groups.get_group_config("general_public")

        combined = self._compose_context(raw_history, insights, group)
        logger.info(
            "🔁 [HUB] recall user=%s session=%s history=%d insights=%d group=%s",
            user_id,
            session_id,
            len(raw_history),
            len(insights),
            group.group_id if group else "-",
        )
        return RecallResult(
            raw_history=raw_history,
            insights=insights,
            user_group=group,
            combined_context=combined,
        )

    def _compose_context(
        self,
        history: List[Dict[str, str]],
        insights: List[InsightEntry],
        group: Optional[UserGroupProfile],
    ) -> str:
        blocks: List[str] = []
        if group:
            blocks.append(
                f"[用户群体] {group.category_name} / 审美:{group.aesthetic_pref} / 沟通:{group.communication_pref}"
            )
        if insights:
            lines = [
                f"- ({insight.topic or '见解'}) {insight.content}"
                for insight in insights
            ]
            blocks.append("[过往见解]\n" + "\n".join(lines))
        if history:
            lines = [f"{turn.get('role', 'user')}: {turn.get('content', '')}" for turn in history]
            blocks.append("[近期对话]\n" + "\n".join(lines))
        return "\n\n".join(blocks)

    async def reflect_on_conversation(
        self,
        session_id: str,
        user_id: str = "anonymous",
        topic_subject: str = "",
    ) -> List[InsightEntry]:
        if self.extractor is None:
            logger.warning("⚠️ [HUB] 未绑定 extractor，跳过内省")
            return []
        turns = self.short_term.get_turns(session_id)
        if not turns:
            return []
        entries = await asyncio.to_thread(
            self.extractor.extract,
            turns,
            user_id,
            session_id,
            topic_subject,
        )
        committed: List[InsightEntry] = []
        for entry in entries:
            if await self.insights.commit_insight(entry):
                committed.append(entry)
        return committed

    def reflect_on_conversation_sync(
        self,
        session_id: str,
        user_id: str = "anonymous",
        topic_subject: str = "",
    ) -> List[InsightEntry]:
        if self.extractor is None:
            logger.warning("⚠️ [HUB] 未绑定 extractor，跳过内省")
            return []
        turns = self.short_term.get_turns(session_id)
        if not turns:
            return []
        entries = self.extractor.extract(turns, user_id, session_id, topic_subject)
        committed: List[InsightEntry] = []
        for entry in entries:
            if self.insights.commit_insight_sync(entry):
                committed.append(entry)
        return committed

    def sync_persistence(self) -> None:
        self.short_term.sync_persistence()
        self.user_groups.sync_persistence()
        logger.info("💾 [HUB] sync_persistence 完成")

    def get_stats(self) -> Dict:
        return {
            "short_term_sessions": len(self.short_term.list_sessions()),
            "insight_archive": self.insights.get_stats(),
            "user_groups": self.user_groups.list_all_groups(),
            "extractor_bound": self.extractor is not None,
        }
