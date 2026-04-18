# -*- coding:utf-8 -*-
import websocket
import hashlib
import base64
import hmac
import json
import os
import ssl
import logging
import pygame
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from config import Config

logger = logging.getLogger("TTS")

class TTSService:
    """语音合成输出服务"""
    def __init__(self, APPID=Config.XF_APPID, APIKey=Config.XF_API_KEY, APISecret=Config.XF_API_SECRET):
        logger.info("🔊 [INIT] 正在初始化 TTS 服务...")
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = "ws-api.xfyun.cn"
        self.config = {
            "voice": "x4_yezi",  # 默认语音
            "speed": 1.0,        # 默认语速
            "volume": 1.0,       # 默认音量
            "pitch": 1.0         # 默认音调
        }
        self.status = "ready"
        
        # 初始化pygame用于音频播放
        try:
            pygame.mixer.init()
            logger.info("✅ [TTS] 音频播放器初始化成功")
        except Exception as e:
            logger.warning("⚠️ [TTS] 音频播放器初始化失败: %s", e)
    
    def _create_url(self):
        """创建WebSocket连接URL"""
        url = 'wss://tts-api.xfyun.cn/v2/tts'
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = f"host: {self.host}\ndate: {date}\nGET /v2/tts HTTP/1.1"
        signature_sha = hmac.new(self.APISecret.encode('utf-8'), signature_origin.encode('utf-8'),
                                 digestmod=hashlib.sha256).digest()
        signature_sha = base64.b64encode(signature_sha).decode(encoding='utf-8')
        auth_origin = f'api_key="{self.APIKey}", algorithm="hmac-sha256", headers="host date request-line", signature="{signature_sha}"'
        authorization = base64.b64encode(auth_origin.encode('utf-8')).decode(encoding='utf-8')
        v = {"authorization": authorization, "date": date, "host": self.host}
        return url + '?' + urlencode(v)

    def generate_speech(self, text, output_path):
        """同步生成音频文件"""
        if os.path.exists(output_path):
            os.remove(output_path)

        def on_message(ws, message):
            try:
                msg_body = json.loads(message)
                if msg_body["code"] != 0:
                    logger.error("TTS Error: %s", msg_body['message'])
                    return
                
                audio = base64.b64decode(msg_body["data"]["audio"])
                status = msg_body["data"]["status"]

                with open(output_path, 'ab') as f:
                    f.write(audio)

                if status == 2:
                    ws.close()
            except Exception as e:
                logger.exception("TTS parsing exception: %s", e)

        def on_error(ws, error):
            logger.error("### TTS WebSocket Error: %s", error)

        def on_open(ws):
            d = {
                "common": {"app_id": self.APPID},
                "business": {
                    "aue": "lame",
                    "sfl": 1,
                    "auf": "audio/L16;rate=16000",
                    "vcn": self.config["voice"],
                    "tte": "utf8",
                    "speed": int(self.config["speed"] * 100),
                    "volume": int(self.config["volume"] * 100),
                    "pitch": int(self.config["pitch"] * 100)
                },
                "data": {
                    "status": 2,
                    "text": str(base64.b64encode(text.encode('utf-8')), "utf-8")
                }
            }
            ws.send(json.dumps(d))

        ws_url = self._create_url()
        ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error)
        ws.on_open = on_open
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        return os.path.exists(output_path)
    
    def speak(self, text, voice_type=None):
        """
        合成并播放语音
        """
        try:
            self.status = "speaking"
            logger.info("🗣️ [TTS] 开始合成语音: %s", text)
            
            # 使用指定的语音类型
            if voice_type:
                original_voice = self.config["voice"]
                self.config["voice"] = voice_type
            
            # 生成音频文件
            output_path = os.path.join(Config.TTS_OUTPUT_DIR, f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
            success = self.generate_speech(text, output_path)
            
            # 恢复原始语音类型
            if voice_type:
                self.config["voice"] = original_voice
            
            if success:
                # 播放音频
                try:
                    pygame.mixer.music.load(output_path)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        pygame.time.Clock().tick(10)
                    logger.info("✅ [TTS] 语音播放完成")
                except Exception as e:
                    logger.error("❌ [TTS] 播放音频失败: %s", e)
            else:
                logger.error("❌ [TTS] 合成语音失败")
            
            self.status = "ready"
            return success
        except Exception as e:
            self.status = "error"
            logger.error("❌ [TTS] 语音合成失败: %s", e)
            return False
    
    def set_config(self, config):
        """
        设置TTS配置
        """
        try:
            self.config.update(config)
            logger.info("🔧 [TTS] 配置已更新: %s", self.config)
            return True
        except Exception as e:
            logger.error("❌ [TTS] 配置更新失败: %s", e)
            return False
    
    def get_status(self):
        """
        获取当前TTS服务状态
        """
        return {
            "status": self.status,
            "config": self.config
        }
    
    def synthesize(self, text, voice_type=None):
        """
        仅合成语音，不播放
        """
        try:
            # 使用指定的语音类型
            if voice_type:
                original_voice = self.config["voice"]
                self.config["voice"] = voice_type
            
            # 生成音频文件
            output_path = os.path.join(Config.TTS_OUTPUT_DIR, f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
            success = self.generate_speech(text, output_path)
            
            # 恢复原始语音类型
            if voice_type:
                self.config["voice"] = original_voice
            
            if success:
                logger.info("✅ [TTS] 语音合成完成: %s", output_path)
                return output_path
            else:
                logger.error("❌ [TTS] 合成语音失败")
                return None
        except Exception as e:
            logger.error("❌ [TTS] 语音合成失败: %s", e)
            return None
    
    def play_audio(self, audio_data):
        """
        播放音频数据
        """
        try:
            # 保存音频数据到临时文件
            temp_file = os.path.join(Config.TTS_OUTPUT_DIR, f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3")
            with open(temp_file, 'wb') as f:
                f.write(audio_data)
            
            # 播放音频
            pygame.mixer.music.load(temp_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # 删除临时文件
            os.remove(temp_file)
            logger.info("✅ [TTS] 音频播放完成")
            return True
        except Exception as e:
            logger.error("❌ [TTS] 播放音频失败: %s", e)
            return False

