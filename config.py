import os
import logging
from pathlib import Path

class Config:
    # --- 核心路径配置 ---
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    PROMPTS_DIR = BASE_DIR / "prompts"
    LOG_FILE = BASE_DIR / "service.log"
    
    # ... (原有配置保持不变)
    PERSONA_DIR = DATA_DIR / "persona_setting_profile"
    PERSONA_JSON_PATH = PERSONA_DIR / "persona.json"
    
    # 向量数据库路径
    CHROMA_PATH = DATA_DIR / "chroma_db_local_model"
    
    # --- 模型配置 ---
    LLM_MODEL_NAME = "qwen2.5:1.5b"
    EMBEDDING_MODEL_PATH = MODELS_DIR / "bge-small-zh-v1.5"
    
    # --- 讯飞语音服务配置 ---
    XF_APPID = os.getenv("XF_APPID", "812ac76a")
    XF_API_KEY = os.getenv("XF_API_KEY", "46834d00f6389d11d8cb73206c756e72")
    XF_API_SECRET = os.getenv("XF_API_SECRET", "ODM1NTYyNDU3MGY0NmVlZjc1MjA2MjVi")
    
    # --- RAG 参数 ---
    RETRIEVAL_TOP_K = 3
    
    # --- 机器人交互配置 ---
    VISION_SAVE_DIR = BASE_DIR / "rviz_captured_images"
    LATEST_IMAGE_PATH = VISION_SAVE_DIR / "latest.jpg"
    
    ASR_SAVE_DIR = DATA_DIR / "voice_to_text"
    AUDIO_OUT_DIR = DATA_DIR / "audio_out"
    
    # 服务文件路径
    SERVICES_DIR = BASE_DIR / "services"
    ASR_SERVICE_PATH = SERVICES_DIR / "asr_service.py"
    VISION_SERVICE_PATH = SERVICES_DIR / "vision_service.py"
    LLM_SERVICE_PATH = SERVICES_DIR / "llm_service.py"
    TTS_SERVICE_PATH = SERVICES_DIR / "tts_service.py"

    @classmethod
    def setup_logging(cls):
        """全局日志初始化：同时输出到控制台和 service.log"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(cls.LOG_FILE, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        # 禁用三方库不必要的调试输出
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("ollama").setLevel(logging.WARNING)
        
        logging.info("🚀 [SYSTEM] 统一日志系统已启动，日志记录于: %s", cls.LOG_FILE)

    @classmethod
    def ensure_dirs(cls):
        """确保必要的目录存在"""
        for attr in ["DATA_DIR", "PROMPTS_DIR", "AUDIO_OUT_DIR", "VISION_SAVE_DIR", "ASR_SAVE_DIR"]:
            path = getattr(cls, attr)
            path.mkdir(parents=True, exist_ok=True)

# 初始化
Config.ensure_dirs()
Config.setup_logging()
