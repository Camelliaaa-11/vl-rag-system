"""Memory hub: single public entry for chat memory flows."""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from config import Config

from .insight_archive import InsightArchive
from .insight_extractor import InsightExtractor
from .models import InsightEntry, UserGroupProfile
from .short_term_memory import ShortTermMemory
from .user_group_profiles import UserGroupProfiles

logger = logging.getLogger("Memory.Hub")


def _model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


@dataclass
class RecallResult:
    raw_history: List[Dict[str, str]] = field(default_factory=list)
    insights: List[InsightEntry] = field(default_factory=list)
    user_group: Optional[UserGroupProfile] = None
    combined_context: str = ""

    def to_dict(self) -> Dict:
        return {
            "raw_history": list(self.raw_history),
            "insights": [_model_to_dict(insight) for insight in self.insights],
            "user_group": _model_to_dict(self.user_group) if self.user_group else None,
            "combined_context": self.combined_context,
        }


class MemoryHub:
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
        self._reflection_state_path = Path(Config.MEMORY_DIR) / "reflection_state.json"
        self._reflection_state = self._load_reflection_state()
        logger.info("Memory hub ready")

    def _load_reflection_state(self) -> Dict[str, int]:
        try:
            if self._reflection_state_path.exists():
                raw = json.loads(self._reflection_state_path.read_text(encoding="utf-8"))
                return {str(k): int(v) for k, v in raw.items()}
        except Exception as exc:
            logger.warning("Failed to load reflection state: %s", exc)
        return {}

    def _save_reflection_state(self) -> None:
        try:
            self._reflection_state_path.write_text(
                json.dumps(self._reflection_state, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save reflection state: %s", exc)

    def attach_extractor(self, extractor_or_caller, **extractor_kwargs) -> None:
        if isinstance(extractor_or_caller, InsightExtractor):
            self.extractor = extractor_or_caller
        else:
            self.extractor = InsightExtractor(
                llm_caller=extractor_or_caller,
                **extractor_kwargs,
            )
        logger.info("Insight extractor attached")

    def record_turn(self, session_id: str, role: str, content: str) -> None:
        self.short_term.add_chat_history(session_id, role, content)

    def clear_session(self, session_id: str) -> bool:
        self._reflection_state.pop(session_id, None)
        self._save_reflection_state()
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

        if user_features:
            group_id = self.user_groups.match_group(user_features)
            group = self.user_groups.get_group_config(group_id)
        else:
            group = self.user_groups.get_group_config("general_public")

        combined = self._compose_context(raw_history, insights, group)
        logger.info(
            "recall user=%s session=%s history=%d insights=%d group=%s",
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
            lines = [f"- ({insight.topic or '见解'}) {insight.content}" for insight in insights]
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
            logger.warning("Extractor not attached, skip reflection")
            return []

        turns = self.short_term.get_turns(session_id)
        start_index = self._reflection_state.get(session_id, 0)
        pending_turns = turns[start_index:]
        if len(pending_turns) < 2:
            logger.info(
                "[MEMORY] skip reflection session=%s pending_turns=%d reason=not_enough_new_turns",
                session_id,
                len(pending_turns),
            )
            return []

        logger.info(
            "[MEMORY] reflect session=%s user=%s new_turns=%d total_turns=%d topic=%s",
            session_id,
            user_id,
            len(pending_turns),
            len(turns),
            topic_subject or "-",
        )

        entries = await asyncio.to_thread(
            self.extractor.extract,
            pending_turns,
            user_id,
            session_id,
            topic_subject,
        )
        committed: List[InsightEntry] = []
        for entry in entries:
            if await self.insights.commit_insight(entry):
                committed.append(entry)

        self._reflection_state[session_id] = len(turns)
        self._save_reflection_state()
        logger.info(
            "[MEMORY] reflection finished session=%s extracted=%d committed=%d",
            session_id,
            len(entries),
            len(committed),
        )
        return committed

    def reflect_on_conversation_sync(
        self,
        session_id: str,
        user_id: str = "anonymous",
        topic_subject: str = "",
    ) -> List[InsightEntry]:
        if self.extractor is None:
            logger.warning("Extractor not attached, skip reflection")
            return []

        turns = self.short_term.get_turns(session_id)
        start_index = self._reflection_state.get(session_id, 0)
        pending_turns = turns[start_index:]
        if len(pending_turns) < 2:
            logger.info(
                "[MEMORY] skip reflection sync session=%s pending_turns=%d reason=not_enough_new_turns",
                session_id,
                len(pending_turns),
            )
            return []

        logger.info(
            "[MEMORY] reflect sync session=%s user=%s new_turns=%d total_turns=%d topic=%s",
            session_id,
            user_id,
            len(pending_turns),
            len(turns),
            topic_subject or "-",
        )

        entries = self.extractor.extract(pending_turns, user_id, session_id, topic_subject)
        committed: List[InsightEntry] = []
        for entry in entries:
            if self.insights.commit_insight_sync(entry):
                committed.append(entry)

        self._reflection_state[session_id] = len(turns)
        self._save_reflection_state()
        logger.info(
            "[MEMORY] reflection sync finished session=%s extracted=%d committed=%d",
            session_id,
            len(entries),
            len(committed),
        )
        return committed

    def sync_persistence(self) -> None:
        self.short_term.sync_persistence()
        self.user_groups.sync_persistence()
        self._save_reflection_state()
        logger.info("Memory persistence synced")

    def get_stats(self) -> Dict:
        return {
            "short_term_sessions": len(self.short_term.list_sessions()),
            "insight_archive": self.insights.get_stats(),
            "user_groups": self.user_groups.list_all_groups(),
            "extractor_bound": self.extractor is not None,
        }
