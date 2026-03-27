#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime


class RvizImageCaptureNode(Node):
    def __init__(self):
        super().__init__('rviz_image_capture_node')

        self.bridge = CvBridge()

        # 订阅虚拟摄像头话题
        self.subscription = self.create_subscription(
            Image,
            'virtual_camera/image_raw',
            self.image_callback,
            10)

        # 保存目录
        self.save_dir = "rviz_captured_images"
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.get_logger().info('🎯 RViz2图像捕捉节点已启动')
        self.get_logger().info('实时捕获图像，保存为latest.jpg供大模型使用')
        self.get_logger().info('按 Ctrl+C 停止')

        # 保存计数器
        self.save_count = 0

    def image_callback(self, msg):
        """保存接收到的图像"""
        try:
            # 将ROS Image消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

            # 转换RGB到BGR用于保存
            cv_image_bgr = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            # ✅ 核心修改：保存为固定的latest.jpg
            latest_filename = f"{self.save_dir}/latest.jpg"
            cv2.imwrite(latest_filename, cv_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            self.save_count += 1

            # 每10帧打印一次日志
            if self.save_count % 10 == 0:
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.get_logger().info(f'📸 已保存 {self.save_count} 帧图像 (最新: {timestamp})')
                self.get_logger().info(f'   文件: {latest_filename}')

            # 可选：同时保存时间戳版本（用于调试）
            debug_save = False
            if debug_save and self.save_count % 50 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                debug_filename = f"{self.save_dir}/debug_{timestamp}.jpg"
                cv2.imwrite(debug_filename, cv_image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                self.get_logger().info(f'💾 调试保存: {debug_filename}')

        except Exception as e:
            self.get_logger().error(f'❌ 图像保存失败: {str(e)}')


def main():
    rclpy.init()
    node = RvizImageCaptureNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 RViz2图像捕捉节点已关闭')
        node.get_logger().info(f'总计保存 {node.save_count} 帧图像')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()