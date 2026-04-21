import base64
import hashlib
import hmac
import json
import logging
import re
import ssl
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from time import mktime
from urllib.parse import urlencode
from wsgiref.handlers import format_date_time

import pygame
import websocket

from config import Config

logger = logging.getLogger("TTS")


class BaseTTSProvider(ABC):
    def __init__(self):
        self.config = {}

    @abstractmethod
    def generate_speech(self, text, output_path):
        raise NotImplementedError

    def set_config(self, config):
        self.config.update(config)
        return True

    def get_status(self):
        return {"provider": self.__class__.__name__, "config": self.config}


class XunfeiTTSProvider(BaseTTSProvider):
    def __init__(self, appid=Config.XF_APPID, api_key=Config.XF_API_KEY, api_secret=Config.XF_API_SECRET):
        super().__init__()
        self.appid = appid
        self.api_key = api_key
        self.api_secret = api_secret
        self.host = "ws-api.xfyun.cn"
        self.config = {
            "voice": Config.TTS_DEFAULT_VOICE or "x4_yezi",
            "speed": 1.0,
            "volume": 1.0,
            "pitch": 1.0,
        }

    def _create_url(self):
        url = "wss://tts-api.xfyun.cn/v2/tts"
        now = datetime.now()
        date = format_date_time(mktime(now.timetuple()))
        signature_origin = f"host: {self.host}\ndate: {date}\nGET /v2/tts HTTP/1.1"
        signature_sha = hmac.new(
            self.api_secret.encode("utf-8"),
            signature_origin.encode("utf-8"),
            digestmod=hashlib.sha256,
        ).digest()
        signature_sha = base64.b64encode(signature_sha).decode("utf-8")
        auth_origin = (
            f'api_key="{self.api_key}", algorithm="hmac-sha256", '
            f'headers="host date request-line", signature="{signature_sha}"'
        )
        authorization = base64.b64encode(auth_origin.encode("utf-8")).decode("utf-8")
        params = {"authorization": authorization, "date": date, "host": self.host}
        return url + "?" + urlencode(params)

    def generate_speech(self, text, output_path):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            output_file.unlink()

        def on_message(ws, message):
            try:
                msg_body = json.loads(message)
                if msg_body["code"] != 0:
                    logger.error("讯飞 TTS 返回错误: %s", msg_body["message"])
                    return

                audio = base64.b64decode(msg_body["data"]["audio"])
                status = msg_body["data"]["status"]

                with output_file.open("ab") as f:
                    f.write(audio)

                if status == 2:
                    ws.close()
            except Exception as exc:
                logger.exception("解析讯飞 TTS 响应失败: %s", exc)

        def on_error(ws, error):
            logger.error("讯飞 TTS WebSocket 错误: %s", error)

        def on_open(ws):
            payload = {
                "common": {"app_id": self.appid},
                "business": {
                    "aue": "lame",
                    "sfl": 1,
                    "auf": "audio/L16;rate=16000",
                    "vcn": self.config["voice"],
                    "tte": "utf8",
                    "speed": int(self.config["speed"] * 100),
                    "volume": int(self.config["volume"] * 100),
                    "pitch": int(self.config["pitch"] * 100),
                },
                "data": {
                    "status": 2,
                    "text": base64.b64encode(text.encode("utf-8")).decode("utf-8"),
                },
            }
            ws.send(json.dumps(payload))

        ws_url = self._create_url()
        ws = websocket.WebSocketApp(ws_url, on_message=on_message, on_error=on_error)
        ws.on_open = on_open
        ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
        return output_file.exists()


class QwenTTSProvider(BaseTTSProvider):
    LANGUAGE_MAP = {
        "zh": "Chinese",
        "zh-cn": "Chinese",
        "cn": "Chinese",
        "chinese": "Chinese",
        "en": "English",
        "en-us": "English",
        "english": "English",
        "ja": "Japanese",
        "japanese": "Japanese",
        "ko": "Korean",
        "korean": "Korean",
        "fr": "French",
        "french": "French",
        "de": "German",
        "german": "German",
        "it": "Italian",
        "italian": "Italian",
        "pt": "Portuguese",
        "portuguese": "Portuguese",
        "ru": "Russian",
        "russian": "Russian",
        "es": "Spanish",
        "spanish": "Spanish",
        "auto": "Auto",
    }

    def __init__(self):
        super().__init__()
        self._model = None
        self._soundfile = None
        self.config = {
            "model_path": Config.QWEN_TTS_MODEL_PATH,
            "device": Config.TTS_DEVICE,
            "voice": Config.QWEN_TTS_VOICE,
            "language": Config.QWEN_TTS_LANGUAGE,
            "style_prompt": Config.QWEN_TTS_STYLE_PROMPT,
            "dtype": Config.QWEN_TTS_DTYPE,
            "sample_rate": Config.QWEN_TTS_SAMPLE_RATE,
        }

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            import torch
            import soundfile as sf
            from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
        except ImportError as exc:
            raise RuntimeError(
                "Qwen TTS 依赖未安装。请安装 `qwen-tts` 和 `soundfile`，"
                "并确认 Python 版本满足官方要求。"
            ) from exc

        requested_device = self.config["device"]
        actual_device = requested_device
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("当前 torch 不支持 CUDA，Qwen TTS 自动回退到 CPU")
            actual_device = "cpu"

        dtype_name = str(self.config.get("dtype") or "").lower()
        if actual_device.startswith("cuda"):
            if dtype_name in ("float16", "fp16", "half"):
                model_dtype = torch.float16
            elif dtype_name in ("bfloat16", "bf16"):
                model_dtype = torch.bfloat16
            else:
                model_dtype = torch.float16
        else:
            model_dtype = torch.float32

        logger.info("加载 Qwen TTS 模型: %s", self.config["model_path"])
        self._soundfile = sf
        self._model = Qwen3TTSModel.from_pretrained(
            self.config["model_path"],
            device_map=actual_device,
            dtype=model_dtype,
            low_cpu_mem_usage=True,
        )
        self.config["resolved_device"] = actual_device
        self.config["resolved_dtype"] = str(model_dtype).replace("torch.", "")
        return self._model

    def _normalize_language(self, language):
        if not language:
            return "Auto"
        return self.LANGUAGE_MAP.get(str(language).strip().lower(), language)

    def generate_speech(self, text, output_path):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        model = self._load_model()
        language = self._normalize_language(self.config.get("language") or "zh")
        voice = self.config.get("voice") or "Vivian"
        style_prompt = (self.config.get("style_prompt") or "").strip() or None

        model_type = getattr(model.model, "tts_model_type", None)
        if model_type == "custom_voice":
            wavs, sample_rate = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=voice,
                instruct=style_prompt,
            )
        elif model_type == "voice_design":
            instruct = style_prompt or f"Use a clear and natural {language} speaking voice."
            wavs, sample_rate = model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
            )
        elif model_type == "base":
            raise RuntimeError(
                "当前加载的是 Qwen Base 模型。该模型需要参考音频，"
                "不适合作为你项目里的默认直接朗读模型。"
            )
        else:
            raise RuntimeError(f"未知的 Qwen TTS 模型类型: {model_type}")

        if not wavs:
            raise RuntimeError("Qwen TTS 未返回有效音频数据")

        audio = wavs[0]
        self._soundfile.write(str(output_file), audio, sample_rate)
        return output_file.exists()


class TTSService:
    def __init__(self, provider=None):
        self.provider_name = (provider or Config.TTS_PROVIDER).lower()
        self.provider = self._create_provider(self.provider_name)
        self.status = "ready"

        try:
            pygame.mixer.init()
            logger.info("音频播放器初始化成功")
        except Exception as exc:
            logger.warning("音频播放器初始化失败: %s", exc)

    def _create_provider(self, provider_name):
        if provider_name == "qwen":
            return QwenTTSProvider()
        if provider_name == "xf":
            return XunfeiTTSProvider()
        raise ValueError(f"不支持的 TTS provider: {provider_name}")

    def generate_speech(self, text, output_path):
        self.status = "speaking"
        try:
            success = self.provider.generate_speech(text, output_path)
            self.status = "ready" if success else "error"
            return success
        except Exception as exc:
            self.status = "error"
            logger.exception("TTS 合成失败(provider=%s): %s", self.provider_name, exc)
            return False

    def speak(self, text, voice_type=None):
        if voice_type:
            self.provider.set_config({"voice": voice_type})

        output_suffix = Config.QWEN_TTS_OUTPUT_FORMAT if self.provider_name == "qwen" else "mp3"
        output_path = Config.TTS_OUTPUT_DIR / f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_suffix}"
        success = self.generate_speech(text, str(output_path))
        if not success:
            return False

        try:
            pygame.mixer.music.load(str(output_path))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            return True
        except Exception as exc:
            logger.error("播放音频失败: %s", exc)
            return False

    def set_config(self, config):
        return self.provider.set_config(config)

    def get_status(self):
        return {
            "status": self.status,
            "provider": self.provider_name,
            "provider_status": self.provider.get_status(),
        }

    def synthesize(self, text, voice_type=None):
        if voice_type:
            self.provider.set_config({"voice": voice_type})

        output_suffix = Config.QWEN_TTS_OUTPUT_FORMAT if self.provider_name == "qwen" else "mp3"
        output_path = Config.TTS_OUTPUT_DIR / f"tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{output_suffix}"
        success = self.generate_speech(text, str(output_path))
        return str(output_path) if success else None

    def get_output_suffix(self):
        return Config.QWEN_TTS_OUTPUT_FORMAT if self.provider_name == "qwen" else "mp3"

    def split_text_for_stream(self, text):
        normalized = (text or "").strip()
        if not normalized:
            return []

        parts = re.split(r"(?<=[。！？；!?;.\n])", normalized)
        segments = [part.strip() for part in parts if part and part.strip()]

        if not segments:
            return [normalized]
        return segments

    def synthesize_stream_segments(self, text, voice_type=None):
        if voice_type:
            self.provider.set_config({"voice": voice_type})

        segments = self.split_text_for_stream(text)
        output_suffix = self.get_output_suffix()

        for index, segment in enumerate(segments):
            output_path = Config.TTS_OUTPUT_DIR / (
                f"tts_stream_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{index}.{output_suffix}"
            )
            success = self.generate_speech(segment, str(output_path))
            if not success:
                raise RuntimeError(f"failed to synthesize segment {index}")
            yield {
                "index": index,
                "text": segment,
                "path": str(output_path),
                "filename": output_path.name,
            }

    def play_audio(self, audio_data):
        temp_file = Config.TTS_OUTPUT_DIR / f"temp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp3"
        try:
            with temp_file.open("wb") as f:
                f.write(audio_data)
            pygame.mixer.music.load(str(temp_file))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            return True
        except Exception as exc:
            logger.error("播放音频失败: %s", exc)
            return False
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
                except OSError:
                    logger.warning("临时音频文件删除失败: %s", temp_file)
