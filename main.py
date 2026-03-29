# main.py
import logging
import datetime
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from services.llm_service import LLMService
from config import Config

# 初始化日志
logger = logging.getLogger("Backend")

app = FastAPI()
model_inference = LLMService()

# 添加CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "后端服务正在运行", "status": "ok"}

@app.get("/health")
def health_check():
    return {"backend": "healthy", "rag": "healthy", "model": "healthy"}

@app.get("/status")
def status_check():
    return {"backend": "healthy", "rag": "healthy", "model": "healthy"}

@app.post("/chat")
async def chat_endpoint(
    image: UploadFile = File(None),
    question: str = Form(None)
):
    logger.info("=== 收到前端聊天请求 ===")
    image_data = None 
    current_question = question if question else "请识别这张图中的展品信息"

    if image:
        try:
            image_data = await image.read()
            logger.info("✅ 图片读取成功: %d 字节", len(image_data))
        except Exception as e:
            logger.error("❌ 读取图片失败: %s", e)

    try:
        # 调用模型进行同步推理 (集成 RAG)
        result = model_inference.generate_response_sync(image_data, current_question)
        logger.info("✅ 推理完成，回答长度: %d 字符", len(result["answer"]))
        
        return {
            "status": "success",
            "data": {
                "answer": result["answer"],
                "context": result.get("context", ""),
                "confidence": "高"
            },
            "message": "处理成功",
            "timestamp": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.exception("❌ 模型处理异常: %s", e)
        return {"status": "error", "message": f"模型处理异常: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run(
        app,               
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
