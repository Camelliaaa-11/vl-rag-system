#!/bin/bash

echo "启动ROS 2图文问答系统..."
echo "========================================"

# 启动语音识别节点
echo "[1/3] 启动语音识别节点..."
gnome-terminal -- bash -c "echo '语音识别节点启动中...'; python3 voice_to_text.py; exec bash"

sleep 2

# 启动图像捕获节点
echo "[2/3] 启动图像捕获节点..."
gnome-terminal -- bash -c "echo '图像捕获节点启动中...'; python3 image_capture.py; exec bash"

sleep 2

# 启动大模型处理节点
echo "[3/3] 启动大模型处理节点..."
gnome-terminal -- bash -c "echo '大模型处理节点启动中...'; python3 local_model_processor.py; exec bash"

sleep 2

echo ""
echo "✅ 所有节点已启动！"
echo ""
echo "📋 系统状态:"
echo "   1. 语音识别 -> 发布到 /asr/user_input"
echo "   2. 图像捕获 -> 保存到 rviz_captured_images/latest.jpg"
echo "   3. 大模型处理 -> 读取图像 + 文字 -> 回复"
echo ""
echo "💡 使用方法:"
echo "   1. 说话 -> 语音识别为文字"
echo "   2. 大模型读取最新图像 + 文字"
echo "   3. 生成图文回复"
echo ""
echo "按 Ctrl+C 停止任意节点"