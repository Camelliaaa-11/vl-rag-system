# VL-RAG-System: 视觉语言增强的机器人导览系统

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![ROS2](https://img.shields.io/badge/ROS2-Foxy/Humble-orange.svg)

## 📌 项目概述

VL-RAG-System 是一个集成了 **多模态大语言模型 (Qwen-VL)**、**RAG (检索增强生成)** 和 **ROS 2 机器人框架** 的智能导览系统。该系统旨在赋予实体机器人（如“技心”）在展厅环境中的视觉感知、专业知识检索和具有情感美学的交互能力。

### 核心特性
- **视觉识别与对齐**: 利用 Qwen-VL 模型实时分析展品图像。
- **专业知识检索**: 基于 ChromaDB + BGE 嵌入模型的 RAG 系统，提供 80+ 件展品的深度背景知识。
- **人格化叙事**: 通过高度定制的 `Ji Xin` (技心) 人设协议，实现沉静、具有技术美感且自然的对话风格。
- **全链路集成**: 覆盖从 ASR (语音识别) 到 LLM 推理再到 TTS (语音合成) 的完整机器人交互闭环。

---

## 🏗️ 系统架构

```text
vl-rag-system/
├── services/                # 🧱 核心服务层 (集成逻辑与 ROS 节点)
│   ├── llm_service.py       # 🧠 大模型推理与 RAG 整合 (视觉大脑)
│   ├── tts_service.py       # 🔊 语音合成服务
│   ├── asr_service.py       # 🎙️ 语音识别服务 (原 voice_to_text.py)
│   └── vision_service.py    # 📸 图像捕获服务 (原 rviz_image_capture_node.py)
├── local_model_processor.py # 🤖 机器人核心控制节点 (Orchestrator)
├── main.py                  # 🌐 Web 后端入口 (FastAPI)
├── config.py                # ⚙️ 全局配置中心
├── voice_to_text.py         # 🎙️ ASR 语音识别节点
├── rviz_image_capture_node.py # 📸 视觉快照采集节点
├── rag/                     # 📚 RAG 检索逻辑与知识库管理
├── prompts/                 # 📝 提示词模板 (人设协议与任务引导)
├── frontend/                # 💻 Web 交互界面
└── data/                    # 💾 数据库、音频输出与临时缓存
```

---

## 🚀 快速开始

### 1. 环境准备
确保您的环境中已安装 Python 3.8+ 和 ROS 2，并安装必要的依赖：

```bash
pip install -r requirements.txt
```

### 2. 模型部署
系统依赖 [Ollama](https://ollama.com/) 运行 Qwen-VL 模型。请先启动 Ollama 服务并拉取模型：

```bash
ollama run qwen2.5:1.5b # 或您配置的其他版本
```

### 3. 系统启动
使用内置脚本一键启动 ROS 各核心节点：

```bash
./start_all.sh
```

同时启动 FastAPI 后端提供 Web 接口支持：

```bash
python main.py
```

---

## 🛠️ 配置说明

所有的核心配置（如 API 密钥、文件路径、检索参数）均在根目录下的 `config.py` 中进行统一管理。建议通过环境变量配置敏感信息：

- `XF_APPID`: 讯飞语音服务 APPID
- `XF_API_KEY`: 讯飞语音服务 APIKey
- `XF_API_SECRET`: 讯飞语音服务 APISecret

---

## 📖 交互流程

1. **感知**: 机器人通过 `rviz_image_capture_node.py` 获取当前视野快照。
2. **输入**: 用户通过语音提问，`voice_to_text.py` 将语音转化为文字。
3. **思考**: `local_model_processor.py` 触发 RAG 检索并调用 Qwen-VL 产生具备人设特征的回复。
4. **输出**: 语音合成模块将文字转化为音频并由机器人播出，同时 Web 前端实时展示对话内容。

---

## 🛡️ 开源协议
本项目采用 MIT 协议。
