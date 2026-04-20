# main.py
import asyncio
import logging
import datetime
import json
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from services.llm_service import LLMService
from config import Config
from memory import MemoryHub
from memory.models import UserGroupProfile

# 初始化日志
logger = logging.getLogger("Backend")

app = FastAPI()

# LLMService 依赖 RAG 向量库；库未构建时 MuseumRetriever 会 sys.exit(1)。
# 这里包一层，让 /chat 之外的端点（特别是 /memory/*）在未建库时也可用。
model_inference: Optional[LLMService] = None
try:
    model_inference = LLMService()
except SystemExit as exc:
    logger.warning("⚠️ [BOOT] LLMService 未能初始化 (RAG DB 缺失?) code=%s", exc.code)
except Exception as exc:
    logger.warning("⚠️ [BOOT] LLMService 初始化异常: %s", exc)


# ========== 记忆系统 (独立测试端点) ==========
memory_hub = MemoryHub()


def _llm_caller_for_memory(messages: List[Dict[str, str]]) -> str:
    """把 LLMService 私有方法包一层，喂给 InsightExtractor。"""
    if model_inference is None:
        raise RuntimeError("LLMService 未初始化，无法执行内省")
    provider = model_inference.provider
    if provider == "deepseek":
        return model_inference._call_deepseek(messages)
    if provider == "ollama":
        system_prompt = next(
            (m["content"] for m in messages if m.get("role") == "system"),
            "",
        )
        return model_inference._call_ollama(messages, system_prompt)
    raise RuntimeError(f"unsupported LLM provider for extractor: {provider}")


if model_inference is not None:
    try:
        memory_hub.attach_extractor(_llm_caller_for_memory)
    except Exception as exc:
        logger.warning("⚠️ [MEMORY] 绑定 extractor 失败: %s", exc)
else:
    logger.info("🪞 [MEMORY] extractor 未绑定；/memory/reflect 将返回 503")

# 添加CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前端静态托管：浏览器打开 http://localhost:{port}/ 直接看主页，
# 避免 file:// 导致的跨 origin 与 Unsafe URL 问题。
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
    return {"message": "后端服务正在运行", "status": "ok"}


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
):
    logger.info("=== 收到前端聊天请求 ===")
    image_data = None 
    current_question = question if question else "请识别这张图中的展品信息"
    parsed_history = None

    if image:
        try:
            image_data = await image.read()
            logger.info("✅ 图片读取成功: %d 字节", len(image_data))
        except Exception as e:
            logger.error("❌ 读取图片失败: %s", e)

    if history:
        try:
            loaded_history = json.loads(history)
            if isinstance(loaded_history, list):
                parsed_history = [
                    {
                        "role": item.get("role", "user"),
                        "content": item.get("content", "")
                    }
                    for item in loaded_history
                    if isinstance(item, dict) and item.get("content")
                ]
                logger.info("🧾 收到历史消息: %d 条", len(parsed_history))
        except Exception as e:
            logger.warning("⚠️ 解析 history 失败，忽略本次上下文: %s", e)

    if model_inference is None:
        logger.error("❌ [CHAT] LLMService 未初始化，无法处理 /chat 请求")
        return {
            "status": "error",
            "message": "后端 LLMService 尚未初始化，可能是 RAG 向量库未构建。请先运行 `python rag/ingest.py` 再启动服务。",
        }

    try:
        # 调用模型进行同步推理 (集成 RAG)
        result = model_inference.generate_response_sync(
            image_data,
            current_question,
            history=parsed_history,
        )
        logger.info("✅ 推理完成，回答长度: %d 字符", len(result["answer"]))
        
        return {
            "status": "success",
            "data": {
                "answer": result["answer"],
                "context": result.get("context", ""),
                "topic_type": result.get("topic_type", ""),
                "topic_subject": result.get("topic_subject", ""),
                "retrieval_query": result.get("retrieval_query", ""),
                "provider": result.get("provider", ""),
                "model_name": result.get("model_name", ""),
                "confidence": "高"
            },
            "message": "处理成功",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception("❌ 模型处理异常: %s", e)
        return {"status": "error", "message": f"模型处理异常: {str(e)}"}

# ========== 记忆系统测试端点 ==========
# 这组路由独立于 /chat 主链路，前端测试页 memory_test.html 直接调用。
# 后续如需把记忆真正接进 /chat，参考 architecture_design.md §12.8。


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
    ok = memory_hub.short_term.clear_chat_history(session_id)
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
        raise HTTPException(
            status_code=503,
            detail="insight extractor 未绑定 (检查 LLM_PROVIDER / API Key)",
        )
    try:
        committed = await memory_hub.reflect_on_conversation(
            session_id=payload.session_id,
            user_id=payload.user_id,
            topic_subject=payload.topic_subject,
        )
    except Exception as exc:
        logger.exception("❌ [MEMORY] 内省失败: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
    return {
        "status": "ok",
        "committed": [entry.dict() for entry in committed],
        "count": len(committed),
    }


@app.get("/memory/insights/{user_id}")
def memory_list_insights(user_id: str, limit: int = 50):
    entries = memory_hub.insights.list_user_insights(user_id, limit=limit)
    return {
        "status": "ok",
        "user_id": user_id,
        "count": len(entries),
        "data": [entry.dict() for entry in entries],
    }


@app.delete("/memory/insight/{insight_id}")
def memory_delete_insight(insight_id: str):
    ok = memory_hub.insights.delete_insight(insight_id)
    return {"status": "ok" if ok else "error", "insight_id": insight_id}


@app.get("/memory/groups")
def memory_list_groups():
    return {
        "status": "ok",
        "data": [p.dict() for p in memory_hub.user_groups.list_all_profiles()],
    }


@app.get("/memory/group/{group_id}")
def memory_get_group(group_id: str):
    profile = memory_hub.user_groups.get_group_config(group_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"group not found: {group_id}")
    return {"status": "ok", "data": profile.dict()}


@app.post("/memory/group/match")
def memory_match_group(payload: GroupMatchIn):
    group_id = memory_hub.user_groups.match_group(payload.user_features)
    profile = memory_hub.user_groups.get_group_config(group_id)
    return {
        "status": "ok",
        "group_id": group_id,
        "data": profile.dict() if profile else None,
    }


@app.post("/memory/group")
def memory_save_group(payload: GroupProfileIn):
    profile = UserGroupProfile(**payload.dict())
    memory_hub.user_groups.save_group_profile(profile)
    return {"status": "ok", "data": profile.dict()}


if __name__ == "__main__":
    # 默认端口从 config 读，避免与本机 ComfyUI 等常驻服务冲突
    uvicorn.run(
        "main:app",
        host=Config.BACKEND_HOST,
        port=Config.BACKEND_PORT,
        log_level="info",
        reload=True,
        # 只监听源码目录，避免 data/ 里的 JSON / chroma 写入触发热重载循环
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
