# -*- coding:utf-8 -*-
import websocket
import hashlib
import base64
import hmac
import json
import os
import ssl
import logging
from datetime import datetime
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time
from config import Config

logger = logging.getLogger("TTS")

class TTSService:
    def __init__(self, APPID=Config.XF_APPID, APIKey=Config.XF_API_KEY, APISecret=Config.XF_API_SECRET):
        self.APPID = APPID
        self.APIKey = APIKey
        self.APISecret = APISecret
        self.host = "ws-api.xfyun.cn"

    def _create_url(self):
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
                    "vcn": "x4_yezi",
                    "tte": "utf8"
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
