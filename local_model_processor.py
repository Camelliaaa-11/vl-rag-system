#!/usr/bin/env python3
import rclpy
import os
import json
import threading
import logging
from rclpy.node import Node
from std_msgs.msg import String
from datetime import datetime
from services.llm_service import LLMService
from services.tts_service import TTSService
from config import Config

# 初始化日志模块
logger = logging.getLogger("RobotBrain")

class StreamingPopProcessor(Node):
    def __init__(self):
        super().__init__('streaming_pop_processor')
        
        # 1. 初始化核心服务
        logger.info("🧠 [INIT] 正在启动机器人核心控制程序...")
        self.model = LLMService()
        self.tts = TTSService()
        
        # 2. 路径配置
        self.latest_image_path = str(Config.LATEST_IMAGE_PATH)
        self.audio_dir = str(Config.AUDIO_OUT_DIR.absolute())
        Config.AUDIO_OUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # 3. 记忆管理
        self.chat_history = []
        self.max_history = 10

        # 4. ROS 通信
        self.sub = self.create_subscription(String, '/asr/user_input', self.on_input, 10)
        self.tts_pub = self.create_publisher(String, '/xunfei/tts_play', 10)
        
        self.sentence_seps = ["。", "！", "？", "\n", "；", "!", "?", "..."]
        
        logger.info("✅ [SYSTEM] 语音输出路径: %s", self.audio_dir)
        logger.info("✅ [SYSTEM] 导览讲解系统已就绪，正在等待交互...")

    def on_input(self, msg):
        user_text = msg.data.strip()
        if not user_text: return
        
        # 记录 ASR 输入
        logger.info("🎧 [ASR] 识别到语音输入: %s", user_text)

        # 检查最新捕捉图像
        image_data = None
        if os.path.exists(self.latest_image_path):
            with open(self.latest_image_path, "rb") as f:
                image_data = f.read()
            # logger.info("📸 [Vision] 已加载最新现场画面")

        current_sentence = ""
        full_reply = ""
        
        # 控制台实时回显 (仅用于开发者本地查看，不污染文件日志)
        print("\n🤖 [BRAIN]: ", end="", flush=True)

        # 流式调用模型接口
        generator = self.model.generate_response_stream(image_data, user_text, history=self.chat_history)

        for chunk in generator:
            print(chunk, end="", flush=True)
            current_sentence += chunk
            full_reply += chunk

            # 断句分段语音合成，提升响应速度
            if any(sep in chunk for sep in self.sentence_seps):
                text_to_speak = current_sentence.strip()
                if len(text_to_speak) > 1:
                    threading.Thread(target=self.run_tts_and_play, args=(text_to_speak,)).start()
                current_sentence = ""

        # 尾部处理
        if current_sentence.strip():
            self.run_tts_and_play(current_sentence.strip())

        # 推理结束后，将完整回复记录进入系统日志文件
        logger.info("🤖 [BRAIN] 生成完整回复: %s", full_reply)

        # 更新对话上下文关联
        self.chat_history.append({"role": "user", "content": user_text})
        self.chat_history.append({"role": "assistant", "content": full_reply})
        # 保持对话上下文窗口大小
        if len(self.chat_history) > self.max_history:
            self.chat_history = self.chat_history[-self.max_history:]

    def run_tts_and_play(self, text):
        """执行 TTS 音频合成与机器人端的播放指令下发"""
        try:
            # 1. 生成唯一时间戳文件名
            ts = datetime.now().strftime("%H%M%S_%f")
            audio_path = os.path.join(self.audio_dir, f"pop_{ts}.mp3")
            
            # 2. 调用服务合成音频
            if self.tts.generate_speech(text, audio_path):
                # 3. 构造播放 JSON 指令并发布
                play_cmd = {
                    "cmd": "append",
                    "file": audio_path
                }
                msg = String()
                msg.data = json.dumps(play_cmd)
                self.tts_pub.publish(msg)
                # logger.info("🔊 [TTS] 已发布朗读指令: %s", audio_path)
            else:
                logger.error("❌ [TTS] 音频文件合成失败: %s", text[:20])

        except Exception as e:
            logger.exception("❌ [CRITICAL] 语音播报逻辑异常: %s", e)

def main(args=None):
    rclpy.init(args=args)
    node = StreamingPopProcessor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        logger.warning("⚠️ [SYSTEM] 接收到退出指令，程序即将关闭")
    except Exception as e:
        logger.exception("❌ [CRITICAL] 程序运行崩溃: %s", e)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()