"""记忆系统的数据模型定义。"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    """短时记忆中的一条对话轮次。"""

    role: str
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    def to_message(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


class UserGroupProfile(BaseModel):
    """用户群体画像。"""

    group_id: str
    category_name: str
    aesthetic_pref: str = ""
    communication_pref: str = ""
    typical_tags: List[str] = Field(default_factory=list)
    response_style: Dict[str, str] = Field(default_factory=dict)


class InsightEntry(BaseModel):
    """LLM 提炼出的结构化见解。"""

    insight_id: str = Field(default_factory=lambda: uuid4().hex)
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    topic: str = ""
    content: str
    key_entities: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    embedding: Optional[List[float]] = None

    def to_chroma_metadata(self) -> Dict[str, str]:
        return {
            "insight_id": self.insight_id,
            "user_id": self.user_id,
            "session_id": self.session_id or "",
            "topic": self.topic,
            "key_entities": ",".join(self.key_entities),
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_chroma(cls, document: str, metadata: Dict[str, str]) -> "InsightEntry":
        raw_entities = metadata.get("key_entities", "")
        entities = [item for item in raw_entities.split(",") if item]
        ts = metadata.get("timestamp")
        try:
            timestamp = datetime.fromisoformat(ts) if ts else datetime.utcnow()
        except ValueError:
            timestamp = datetime.utcnow()
        return cls(
            insight_id=metadata.get("insight_id") or uuid4().hex,
            user_id=metadata.get("user_id", "anonymous"),
            session_id=metadata.get("session_id") or None,
            topic=metadata.get("topic", ""),
            content=document,
            key_entities=entities,
            timestamp=timestamp,
        )
