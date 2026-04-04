# VL-RAG-System: 视觉语言增强的机器人导览系统

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![ROS2](https://img.shields.io/badge/ROS2-Foxy/Humble-orange.svg)

## 📌 项目概述

VL-RAG-System 是一个集成了 **多模态大语言模型 (Qwen-VL)**、**RAG (检索增强生成)** 和 **ROS 2 机器人框架** 的智能导览系统。该系统旨在赋予实体机器人（如“技心”）在展厅环境中的视觉感知、专业知识检索和具有情感美学的交互能力。

### 核心特性
- **视觉识别与对齐**: 利用 Qwen-VL 模型实时分析展馆现场图像。
- **专业知识检索**: 基于 ChromaDB + BGE 嵌入模型的 RAG 系统，提供 80+ 件展品的深度背景知识。
- **人格化叙事**: 通过高度定制的 `Ji Xin` (技心) 人设协议，实现沉静、具有技术美感且自然的对话风格。
- **全链路集成**: 在 **Ubuntu** 环境下实现从 ASR 到 LLM 推理再到 TTS 的完整机器人交互闭环。

---

## 🏗️ 系统架构

```text
vl-rag-system/
├── agents/                  # 🤖 代理层 (业务逻辑与策略引擎)
│   ├── base_agent.py        # 🆔 代理基类
│   ├── intro_agent.py       # 🏺 展品讲解代理
│   ├── chat_agent.py        # 💬 深度聊天代理
│   └── smalltalk_agent.py   # 🌸 闲聊情感代理
├── services/                # 🧱 核心服务层 (集成逻辑与 ROS 节点)
│   ├── agent_manager.py     # 🧭 场景分发与路由管理
│   ├── llm_service.py       # 🧠 大模型推理引导 (视觉大脑)
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

## 🛠️ 环境要求 (Prerequisites)

*   **操作系统**: **Ubuntu 22.04 LTS** (推荐) 或更高版本。
*   **机器人框架**: ROS 2 Humble (必须在 Ubuntu 下运行)。
*   **资源映射**: 
    - **Mode A (PC 调试)**: 支持 Windows (WSL2) 或 Linux。
    - **Mode B (实机交互)**: 必须运行在原生 Linux 或双系统的 Ubuntu 环境中。

---

## 🚀 启动指引 (Getting Started)

本项目支持两种独立的运行模式，分别适配不同的开发与部署需求：

### 模式 A：PC 本地调试 (PC Debugging Mode)
**适用场景**：在开发机（Windows/Linux/WSL）上快速验证 RAG 检索效果、模型回答质量或 UI 交互逻辑。
1.  **环境准备**: 确保 Python 环境已安装 `requirements.txt`。
2.  **启动后端**:
    ```bash
    python main.py
    ```
3.  **开始调试**: 通过前端页面或调用 `http://localhost:8000/chat` 与系统交互。**此模式不依赖 ROS 2 环境。**

### 模式 B：机器人实机交互 (Robot Interaction Mode)
**适用场景**：部署在机器人板卡（如 Orin/Jetson）上，利用 ASR/Vision 硬件进行实时的实景导览交互。
1.  **环境准备**: 确保已进入 ROS 2 工作空间环境。
2.  **一键启动**:
    ```bash
    chmod +x start_all.sh
    ./start_all.sh
    ```
    该脚本将依次拉起：ASR 语音识别节点 🎙️ ⮕ Vision 图像采集节点 📸 ⮕ Robot Brain 核心处理器 🧠。
3.  **运行监控**: 日志将实时汇总记录在根目录下的 `service.log` 中。

---

## ⚙️ 配置说明 (Configuration)

所有的核心配置（如 API 密钥、文件路径、模型参数）均在根目录下的 **`config.py`** 中统一管理。
- **路径对齐**: 采用 `pathlib` 动态计算路径，完美适配 Windows 与 Linux 环境。
- **统一日志**: 调用 `Config.setup_logging()` 可实现全系统日志向 `service.log` 的实时写入。

---

## 📖 交互流程对比 (Interaction Flow)

| 流程阶段 | 模式 A (PC 本地调试) | 模式 B (机器人模式) |
| :--- | :--- | :--- |
| **感知 (Input)** | 用户手动上传图片 / 输入文字 | `vision_service.py` 自动抓拍快照 |
| **触发 (Trigger)** | FastAPI 端点 `/chat` 接收 HTTP 请求 | `asr_service.py` 监听语音并发布到 ASR 话题 |
| **思考 (Brain)** | `LLMService` 在 FastAPI 线程内同步推理 | `local_model_processor.py` 响应话题并流式推理 |
| **输出 (Output)** | HTTP 回包返回完整文本 | `tts_service.py` 生成 MP3 并下发机器人播放指令 |

---

## 🛡️ 开源协议
本项目采用 MIT 协议。
