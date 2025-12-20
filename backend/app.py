# backend/app.py
from fastapi import FastAPI, UploadFile, File, Form
#uplaodfile,file处理用户输入内容，form是表单内容
from fastapi.middleware.cors import CORSMiddleware
#端口连接
import uvicorn
#服务器
import datetime
import uuid

""" import httpx #异步http
import asyncio #异步
from typing import Optional
async def call_model_api( #一个调用其他模型的函数
    question: str, 
    image_data: Optional[bytes] = None,#这个参数可以是X类型，也可以是None
    timeout: float = 30.0  # 30秒超时
) -> dict:
    
    调用A同学的模型API
    Args:
        question: 问题文本
        image_data: 图片二进制数据
        timeout: 超时时间（秒）
    Returns:
        包含回答的字典

    # A同学的API地址
    API_URL = "http://localhost:5000/predict"  # 示例地址
    try:
        # 准备请求数据，这是file格式
        files = None
        data = {"question": question}
        if image_data:
            # 如果有图片，使用multipart/form-data
            files = {"image": ("image.jpg", image_data, "image/jpeg")}
            #image_jpg是建议的文件名，jpeg是告诉服务器文件类型
        else:
            # 如果没有图片，用JSON
            headers = {"Content-Type": "application/json"}
            #要明确的告诉服务器类型，没有文件就json，JSON是Web API的标准数据交换格式
        # 异步发送请求
        async with httpx.AsyncClient(timeout=timeout) as client:
            if files:
                response = await client.post(API_URL, data=data, files=files)
            else:
                response = await client.post(API_URL, json=data, headers=headers)
        # 检查响应
        response.raise_for_status()  # 如果不是2xx，抛出异常，抛出异常就不会让下一个语句崩溃
        #   2xx：成功（200 OK, 201 Created, 204 No Content）
            3xx：重定向（301 Moved, 304 Not Modified）
            4xx：客户端错误（400 Bad Request, 401 Unauthorized, 404 Not Found）
            5xx：服务器错误
        result = response.json()
        # 根据A同学API的实际返回格式调整
        return {
            "success": True,
            "answer": result.get("answer", result.get("text", "无回答")),
            "raw_response": result,  # 保存原始响应，方便调试
            "status_code": response.status_code
        }
    except httpx.TimeoutException:
        return {
            "success": False,
            "answer": "模型API请求超时",
            "error": f"超过{timeout}秒未响应"
        }
    except httpx.RequestError as e:
        return {
            "success": False,
            "answer": "无法连接到模型服务",
            "error": str(e)
        }
    except Exception as e:
        return {
            "success": False,
            "answer": "模型API处理失败",
            "error": str(e)
        } """
    
app = FastAPI()

# 添加CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],#允许所有来源访问
    allow_credentials=True,#允许前端发送认证信息
    allow_methods=["*"],#允许所有http的方法
    allow_headers=["*"],#允许所有请求头
)

# 根路径 - 测试用
@app.get("/")
def root():
    return {"message": "后端服务正在运行", "status": "ok"}

# 健康检查 - 前端会调用
@app.get("/health")
def health_check():
    return {"backend": "healthy", "rag": "healthy", "model": "healthy"}

# 状态检查 - 前端代码第87行调用的 /status
@app.get("/status")
def status_check():
    return {"backend": "healthy", "rag": "healthy", "model": "healthy"}

#前端调用的聊天接口
@app.post("/chat")
async def chat_endpoint( #异步处理
    image: UploadFile = File(None),  # 前端可能不传图片
    question: str = Form(None)       # 前端可能不传问题
):
    print("=== 收到前端请求 ===")
    print(f"图片: {image.filename if image else '无图片'}")
    print(f"问题: {question if question else '无问题'}")
     
    # 若没有收到问题
    if not question:
        question = "这是什么作品？"
    # 读取图片（如果有）
    image_data = None
    if image:
        try:
            image_data = await image.read() #异步读取图片二进制数据
            print(f"图片大小: {len(image_data) if image_data else 0} 字节")
        except Exception as e:
            print(f"读取图片失败: {e}")

    """ model_result = await call_ai_model_safe(question, image_data)
    
    if model_result["success"]:
        status = "success"
        answer = model_result["answer"]
        confidence = "高"
        message = "处理成功"
    else:
        status = "error"
        answer = model_result["answer"]  # 可能是"模型处理失败"
        confidence = "低"
        message = f"模型调用失败: {model_result['error']}"  """
    # TODO: 这里应该调用A同学的模型
    # 暂时返回测试数据
    
    # 模拟不同情况
    if image_data:
        answer = f"✅ 已收到图片和问题\n图片名称: {image.filename}\n您的问题是: {question}"
    else:
        answer = f"✅ 已收到问题\n您的问题是: {question}"
    return {
        "status": "success",
        "data": {
            "answer": answer,
            "context": "这是测试数据，实际会从知识库获取",
            "confidence": "高"
        },
        "message": "处理成功",
        "timestamp": datetime.datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        app,               
        host="0.0.0.0",
        port=8000,
        #reload=True,
        log_level="info"
    )
