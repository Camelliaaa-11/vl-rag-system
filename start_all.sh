#!/bin/bash

echo "启动 VL-RAG-System: 技心机器人全链路交互系统..."
echo "========================================"

# 1. 启动语音识别服务 (耳朵)
echo "[1/3] 🎙️ 正在启动 ASR 语音识别服务..."
gnome-terminal -- bash -c "echo 'ASR 节点运行中...'; python3 services/asr_service.py; exec bash"

sleep 2

# 2. 启动图像捕捉服务 (眼睛)
echo "[2/3] 📸 正在启动 Vision 图像采集服务..."
gnome-terminal -- bash -c "echo 'Vision 节点运行中...'; python3 services/vision_service.py; exec bash"

sleep 2

# 3. 启动机器人大脑处理器 (指挥官)
echo "[3/3] 🧠 正在启动 Robot Brain 大脑处理器..."
gnome-terminal -- bash -c "echo 'Robot Brain 节点运行中...'; python3 local_model_processor.py; exec bash"

sleep 2

echo ""
echo "✅ 全系统器官与大脑已完成一键同步启动！"
echo "🔍 核心日志实时查看: tail -f service.log"
echo ""
echo "💡 温馨提示:"
echo "   1. 确保已在后台启动 Ollama (qwen2.5:1.5b)"
echo "   2. 若需调试 Web 后端，请另外运行: python3 main.py"
echo ""
echo "按 Ctrl+C 停止任意窗口对应的节点"