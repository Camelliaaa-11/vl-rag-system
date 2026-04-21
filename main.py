import asyncio
import datetime
import json
import logging
from contextlib import suppress
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from config import Config
from memory import MemoryHub
from memory.models import UserGroupProfile
from services.llm_service import LLMService
from services.tts_service import TTSService

logger = logging.getLogger("Backend")

app = FastAPI()


def _model_to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


model_inference: Optional[LLMService] = None
try:
    model_inference = LLMService()
except SystemExit as exc:
    logger.warning("[BOOT] LLMService init failed code=%s", exc.code)
except Exception as exc:
    logger.warning("[BOOT] LLMService init error: %s", exc)


memory_hub = MemoryHub()
tts_service = TTSService()


def _llm_caller_for_memory(messages: List[Dict[str, str]]) -> str:
    if model_inference is None:
        raise RuntimeError("LLMService 未初始化，无法执行 memory reflection")
    if model_inference.provider == "deepseek":
        return model_inference._call_deepseek(messages)
    if model_inference.provider == "ollama":
        system_prompt = next((m["content"] for m in messages if m.get("role") == "system"), "")
        return model_inference._call_ollama(messages, system_prompt)
    raise RuntimeError(f"unsupported LLM provider for extractor: {model_inference.provider}")


async def _run_memory_reflection(session_id: str, user_id: str, topic_subject: str) -> None:
    with suppress(Exception):
        committed = await memory_hub.reflect_on_conversation(
            session_id=session_id,
            user_id=user_id,
            topic_subject=topic_subject,
        )
        if committed:
            logger.info(
                "[MEMORY] async reflection committed session=%s count=%d",
                session_id,
                len(committed),
            )


if model_inference is not None:
    try:
        memory_hub.attach_extractor(_llm_caller_for_memory)
    except Exception as exc:
        logger.warning("[MEMORY] attach extractor failed: %s", exc)
else:
    logger.info("[MEMORY] extractor unavailable because LLMService is not ready")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_frontend_dir = Config.BASE_DIR / "frontend"
if _frontend_dir.exists():
    app.mount(
        "/frontend",
        StaticFiles(directory=str(_frontend_dir), html=True),
        name="frontend",
    )


@app.get("/")
def root():
    index_path = _frontend_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "backend is running", "status": "ok"}


@app.get("/memory-test")
def memory_test_page():
    page = _frontend_dir / "memory_test.html"
    if page.exists():
        return FileResponse(str(page))
    raise HTTPException(status_code=404, detail="memory_test.html not found")


@app.get("/health")
def health_check():
    return {"backend": "healthy", "rag": "healthy", "model": "healthy"}


@app.get("/status")
def status_check():
    return {"backend": "healthy", "rag": "healthy", "model": "healthy"}


@app.post("/chat")
async def chat_endpoint(
    image: UploadFile = File(None),
    question: str = Form(None),
    history: str = Form(None),
    session_id: str = Form(None),
    user_id: str = Form("anonymous"),
    user_features: str = Form(None),
):
    logger.info("=== incoming /chat request ===")

    image_data = None
    current_question = question if question else "请识别这张图中的展品信息"
    current_session_id = (session_id or "").strip() or f"session_{uuid4().hex[:12]}"
    current_user_id = (user_id or "anonymous").strip() or "anonymous"
    parsed_history = None
    parsed_user_features = None

    if image:
        try:
            image_data = await image.read()
            logger.info("image loaded: %d bytes", len(image_data))
        except Exception as exc:
            logger.error("read image failed: %s", exc)

    if history:
        try:
            loaded_history = json.loads(history)
            if isinstance(loaded_history, list):
                parsed_history = [
                    {
                        "role": item.get("role", "user"),
                        "content": item.get("content", ""),
                    }
                    for item in loaded_history
                    if isinstance(item, dict) and item.get("content")
                ]
                logger.info("parsed request history: %d turns", len(parsed_history))
        except Exception as exc:
            logger.warning("parse history failed, ignore request history: %s", exc)

    if user_features:
        try:
            loaded_features = json.loads(user_features)
            if isinstance(loaded_features, dict):
                parsed_user_features = loaded_features
        except Exception as exc:
            logger.warning("parse user_features failed, ignore request user_features: %s", exc)

    if model_inference is None:
        logger.error("[CHAT] LLMService not initialized")
        return {
            "status": "error",
            "message": "后端 LLMService 尚未初始化，可能是 RAG 向量库未构建。",
        }

    try:
        memory_hub.record_turn(current_session_id, "user", current_question)
        recall_result = memory_hub.recall(
            query=current_question,
            user_id=current_user_id,
            session_id=current_session_id,
            top_k=3,
            user_features=parsed_user_features,
            history_tail=8,
        )
        merged_history = recall_result.raw_history or parsed_history

        result = model_inference.generate_response_sync(
            image_data,
            current_question,
            history=merged_history,
            memory_context=recall_result.combined_context,
            memory_profile=_model_to_dict(recall_result.user_group) if recall_result.user_group else None,
        )

        memory_hub.record_turn(current_session_id, "assistant", result["answer"])
        asyncio.create_task(
            _run_memory_reflection(
                session_id=current_session_id,
                user_id=current_user_id,
                topic_subject=result.get("topic_subject", ""),
            )
        )

        logger.info("chat completed, answer length=%d", len(result["answer"]))
        return {
            "status": "success",
            "data": {
                "answer": result["answer"],
                "context": result.get("context", ""),
                "topic_type": result.get("topic_type", ""),
                "topic_subject": result.get("topic_subject", ""),
                "retrieval_query": result.get("retrieval_query", ""),
                "memory_context": result.get("memory_context", ""),
                "memory_group": _model_to_dict(recall_result.user_group) if recall_result.user_group else None,
                "memory_insight_count": len(recall_result.insights),
                "session_id": current_session_id,
                "user_id": current_user_id,
                "provider": result.get("provider", ""),
                "model_name": result.get("model_name", ""),
                "confidence": "high",
            },
            "message": "处理成功",
            "timestamp": datetime.datetime.now().isoformat(),
        }
    except Exception as exc:
        logger.exception("chat failed: %s", exc)
        return {"status": "error", "message": f"模型处理异常: {str(exc)}"}


class TurnIn(BaseModel):
    session_id: str = Field(..., description="会话 ID")
    role: str = Field("user", description="user / assistant / system")
    content: str


class RecallIn(BaseModel):
    query: str
    user_id: str = "anonymous"
    session_id: Optional[str] = None
    top_k: int = 3
    user_features: Optional[Dict[str, Any]] = None
    history_tail: int = 6


class ReflectIn(BaseModel):
    session_id: str
    user_id: str = "anonymous"
    topic_subject: str = ""


class GroupMatchIn(BaseModel):
    user_features: Dict[str, Any]


class GroupProfileIn(BaseModel):
    group_id: str
    category_name: str
    aesthetic_pref: str = ""
    communication_pref: str = ""
    typical_tags: List[str] = Field(default_factory=list)
    response_style: Dict[str, str] = Field(default_factory=dict)


class TTSSynthesizeIn(BaseModel):
    text: str
    voice: Optional[str] = None


class TTSConfigIn(BaseModel):
    voice: Optional[str] = None
    speed: Optional[float] = None
    volume: Optional[float] = None
    pitch: Optional[float] = None


@app.get("/memory/stats")
def memory_stats():
    return {"status": "ok", "data": memory_hub.get_stats()}


@app.post("/memory/turn")
def memory_add_turn(payload: TurnIn):
    memory_hub.record_turn(payload.session_id, payload.role, payload.content)
    return {
        "status": "ok",
        "count": memory_hub.short_term.get_history_count(payload.session_id),
    }


@app.get("/memory/sessions")
def memory_list_sessions():
    return {"status": "ok", "data": memory_hub.short_term.list_sessions()}


@app.get("/memory/history/{session_id}")
def memory_history(session_id: str):
    return {
        "status": "ok",
        "session_id": session_id,
        "history": memory_hub.short_term.get_raw_history(session_id),
        "count": memory_hub.short_term.get_history_count(session_id),
    }


@app.delete("/memory/session/{session_id}")
def memory_clear_session(session_id: str):
    ok = memory_hub.clear_session(session_id)
    return {"status": "ok" if ok else "error", "session_id": session_id}


@app.post("/memory/recall")
def memory_recall(payload: RecallIn):
    result = memory_hub.recall(
        query=payload.query,
        user_id=payload.user_id,
        session_id=payload.session_id,
        top_k=payload.top_k,
        user_features=payload.user_features,
        history_tail=payload.history_tail,
    )
    return {"status": "ok", "data": result.to_dict()}


@app.post("/memory/reflect")
async def memory_reflect(payload: ReflectIn):
    if memory_hub.extractor is None:
        raise HTTPException(status_code=503, detail="insight extractor 未绑定")
    try:
        committed = await memory_hub.reflect_on_conversation(
            session_id=payload.session_id,
            user_id=payload.user_id,
            topic_subject=payload.topic_subject,
        )
    except Exception as exc:
        logger.exception("[MEMORY] reflect failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "status": "ok",
        "committed": [_model_to_dict(entry) for entry in committed],
        "count": len(committed),
    }


@app.get("/memory/insights/{user_id}")
def memory_list_insights(user_id: str, limit: int = 50):
    entries = memory_hub.insights.list_user_insights(user_id, limit=limit)
    return {
        "status": "ok",
        "user_id": user_id,
        "count": len(entries),
        "data": [_model_to_dict(entry) for entry in entries],
    }


@app.delete("/memory/insight/{insight_id}")
def memory_delete_insight(insight_id: str):
    ok = memory_hub.insights.delete_insight(insight_id)
    return {"status": "ok" if ok else "error", "insight_id": insight_id}


@app.get("/memory/groups")
def memory_list_groups():
    return {
        "status": "ok",
        "data": [_model_to_dict(p) for p in memory_hub.user_groups.list_all_profiles()],
    }


@app.get("/memory/group/{group_id}")
def memory_get_group(group_id: str):
    profile = memory_hub.user_groups.get_group_config(group_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"group not found: {group_id}")
    return {"status": "ok", "data": _model_to_dict(profile)}


@app.post("/memory/group/match")
def memory_match_group(payload: GroupMatchIn):
    group_id = memory_hub.user_groups.match_group(payload.user_features)
    profile = memory_hub.user_groups.get_group_config(group_id)
    return {
        "status": "ok",
        "group_id": group_id,
        "data": _model_to_dict(profile) if profile else None,
    }


@app.post("/memory/group")
def memory_save_group(payload: GroupProfileIn):
    profile = UserGroupProfile(**_model_to_dict(payload))
    memory_hub.user_groups.save_group_profile(profile)
    return {"status": "ok", "data": _model_to_dict(profile)}


@app.get("/api/tts/status")
def tts_status():
    return {"status": "ok", "data": tts_service.get_status()}


@app.post("/api/tts/config")
def tts_config(payload: TTSConfigIn):
    config = {k: v for k, v in _model_to_dict(payload).items() if v is not None}
    tts_service.set_config(config)
    return {"status": "ok", "data": tts_service.get_status()}


@app.post("/api/tts/synthesize")
def tts_synthesize(payload: TTSSynthesizeIn):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="text is required")

    output_path = tts_service.synthesize(payload.text.strip(), voice_type=payload.voice)
    if not output_path:
        raise HTTPException(status_code=500, detail="TTS synthesis failed")

    audio_file = Path(output_path)
    media_type = "audio/wav" if audio_file.suffix.lower() == ".wav" else "audio/mpeg"
    return FileResponse(path=output_path, media_type=media_type, filename=audio_file.name)


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=Config.BACKEND_HOST,
        port=Config.BACKEND_PORT,
        log_level="info",
        reload=Config.BACKEND_RELOAD,
        reload_dirs=[
            str(Config.BASE_DIR / "services"),
            str(Config.BASE_DIR / "rag"),
            str(Config.BASE_DIR / "memory"),
            str(Config.BASE_DIR / "agents"),
            str(Config.BASE_DIR / "prompts"),
            str(Config.BASE_DIR),
        ],
        reload_excludes=["data/*", "*.log", "rviz_captured_images/*"],
    )
