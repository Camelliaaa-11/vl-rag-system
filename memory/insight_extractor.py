"""Conversation to insight extraction via LLM."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from config import Config

from .models import ChatTurn, InsightEntry

logger = logging.getLogger("Memory.Extractor")

LLMCaller = Callable[[List[dict]], str]
DEFAULT_PROMPT_FILE = "insight_extraction.md"


def _preview(text: str, limit: int = 240) -> str:
    normalized = " ".join((text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3] + "..."


class InsightExtractor:
    def __init__(
        self,
        llm_caller: LLMCaller,
        prompt_path: Optional[Path] = None,
        max_turns_per_extract: int = 10,
    ):
        self.llm_caller = llm_caller
        self.prompt_path = Path(prompt_path or (Config.PROMPTS_DIR / DEFAULT_PROMPT_FILE))
        self.max_turns_per_extract = max_turns_per_extract
        self._template_cache: Optional[str] = None

    def _load_template(self) -> str:
        if self._template_cache is not None:
            return self._template_cache
        try:
            self._template_cache = self.prompt_path.read_text(encoding="utf-8")
        except Exception as exc:
            logger.error("[EXTRACTOR] load prompt failed %s: %s", self.prompt_path, exc)
            self._template_cache = (
                "请从对话里提炼用户见解，以 JSON 数组返回，字段包含 topic/content/key_entities。\n"
                "对话:\n---\n{conversation}\n---\n已知话题:{topic_subject}"
            )
        return self._template_cache

    def _format_conversation(self, turns: Iterable[ChatTurn]) -> str:
        lines: List[str] = []
        for turn in list(turns)[-self.max_turns_per_extract:]:
            role = "用户" if turn.role == "user" else "机器人"
            lines.append(f"{role}: {turn.content}")
        return "\n".join(lines)

    def _parse_llm_output(self, raw: str) -> List[dict]:
        if not raw:
            return []
        text = raw.strip()

        fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            array_match = re.search(r"\[.*\]", text, flags=re.DOTALL)
            if not array_match:
                logger.warning("[EXTRACTOR] cannot parse llm output: %s", _preview(text, 200))
                return []
            try:
                parsed = json.loads(array_match.group(0))
            except json.JSONDecodeError as exc:
                logger.warning("[EXTRACTOR] json parse failed: %s", exc)
                return []

        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]

    def _build_messages(self, conversation_text: str, topic_subject: str) -> List[dict]:
        template = self._load_template()
        user_prompt = template.format(
            conversation=conversation_text or "（空）",
            topic_subject=topic_subject or "无",
        )
        return [
            {"role": "system", "content": "你是沉稳专业的对话内省助手。"},
            {"role": "user", "content": user_prompt},
        ]

    def extract(
        self,
        turns: List[ChatTurn],
        user_id: str = "anonymous",
        session_id: Optional[str] = None,
        topic_subject: str = "",
    ) -> List[InsightEntry]:
        if not turns:
            return []

        conversation_text = self._format_conversation(turns)
        messages = self._build_messages(conversation_text, topic_subject)
        logger.info(
            "[EXTRACTOR] start user=%s session=%s turns=%d topic=%s conversation=%s",
            user_id,
            session_id,
            len(turns),
            topic_subject or "-",
            _preview(conversation_text),
        )

        try:
            raw = self.llm_caller(messages)
        except Exception as exc:
            logger.error("[EXTRACTOR] llm call failed: %s", exc)
            return []

        logger.info("[EXTRACTOR] raw output: %s", _preview(raw, 320))
        items = self._parse_llm_output(raw)

        entries: List[InsightEntry] = []
        for item in items:
            content = (item.get("content") or "").strip()
            if not content:
                continue
            entries.append(
                InsightEntry(
                    user_id=user_id,
                    session_id=session_id,
                    topic=(item.get("topic") or "").strip()[:24],
                    content=content,
                    key_entities=[
                        str(tag).strip()
                        for tag in (item.get("key_entities") or [])
                        if str(tag).strip()
                    ][:5],
                )
            )

        if not entries:
            logger.info("[EXTRACTOR] no insight extracted user=%s session=%s", user_id, session_id)
            return []

        logger.info("[EXTRACTOR] extracted %d insights user=%s session=%s", len(entries), user_id, session_id)
        for index, entry in enumerate(entries, start=1):
            logger.info(
                "[EXTRACTOR] insight #%d topic=%s content=%s entities=%s",
                index,
                entry.topic or "-",
                entry.content,
                entry.key_entities,
            )
        return entries
