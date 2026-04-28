import cv2
import sys
import argparse
from datetime import datetime
import torch
import numpy as np
import os

# 导入人脸交换相关模块
from models.mobile_model import FaceSwap, l2_norm
from models.align_face import dealign, align_img, align_imgs
from models.prepare_data import LandmarkModel
from models.util import tesnor2cv, cv2tensor

def get_id(id_net, id_img, device):
    """获取人脸ID特征"""
    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2tensor(id_img)
    # 移动到GPU
    id_img = id_img.to(device)
    mean = torch.tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1)).to(device)
    std = torch.tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1)).to(device)
    id_img = (id_img - mean) / std
    id_emb, id_feature = id_net(id_img)
    id_emb = l2_norm(id_emb)
    return id_emb, id_feature

class RealtimeFaceSwap:
    def __init__(self, source_img_path, use_gpu=True): #实现算法

class RealtimeFaceSwapCamera:
    def __init__(self, source_img_path, camera_id=0, width=1920, height=1080, fps=30, 
                 use_gstreamer=True, use_gpu=True):
        """
        初始化实时人脸交换摄像头系统
        
        参数:
            source_img_path: 源人脸图片路径
            camera_id: 摄像头ID
            width: 图像宽度
            height: 图像高度
            fps: 帧率
            use_gstreamer: 是否使用GStreamer（Jetson平台推荐）
            use_gpu: 是否使用GPU加速
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.use_gstreamer = use_gstreamer
        self.cap = None
        self.running = False
        
        # 初始化人脸交换系统
        self.faceswap = RealtimeFaceSwap(source_img_path, use_gpu=use_gpu)
    
    def get_gstreamer_pipeline(self):
        """构建GStreamer pipeline（适用于Jetson平台）"""
        pipeline = (
            f"nvarguscamerasrc sensor-id={self.camera_id} ! "
            f"video/x-raw(memory:NVMM), "
            f"width={self.width}, height={self.height}, "
            f"format=NV12, framerate={self.fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! appsink"
        )
        return pipeline
    
    def get_v4l2_pipeline(self):
        """构建V4L2 pipeline（适用于USB摄像头）"""
        pipeline = (
            f"v4l2src device=/dev/video{self.camera_id} ! "
            f"video/x-raw, width={self.width}, height={self.height}, "
            f"framerate={self.fps}/1, format=YUY2 ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! appsink"
        )
        return pipeline
    
    def initialize_camera(self):
        """初始化摄像头"""
        print(f"正在初始化摄像头 (ID: {self.camera_id})...")
        
        if self.use_gstreamer:
            # 尝试使用GStreamer pipeline
            try:
                gst_pipeline = self.get_gstreamer_pipeline()
                print(f"尝试使用GStreamer CSI pipeline...")
                self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    print("✅ 成功使用GStreamer CSI摄像头")
                    return True
            except Exception as e:
                print(f"GStreamer CSI摄像头失败: {e}")
            
            # 尝试V4L2 pipeline
            try:
                v4l2_pipeline = self.get_v4l2_pipeline()
                print(f"尝试使用GStreamer V4L2 pipeline...")
                self.cap = cv2.VideoCapture(v4l2_pipeline, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    print("✅ 成功使用GStreamer V4L2摄像头")
                    return True
            except Exception as e:
                print(f"GStreamer V4L2摄像头失败: {e}")
        
        # 回退到标准OpenCV方法
        print("回退到标准OpenCV方法...")
        try:
            self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_V4L2)
        except:
            self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ 错误：无法打开摄像头 {self.camera_id}")
            return False
        
        # 设置摄像头参数
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        print(f"✅ 摄像头已打开（标准方法）")
        print(f"   分辨率: {actual_width}x{actual_height}")
        print(f"   帧率: {actual_fps} FPS\n")
        
        return True
    
    def run(self):
        """运行实时人脸交换"""
        if not self.initialize_camera():
            return
        
        self.running = True
        frame_count = 0
        process_count = 0
        start_time = datetime.now()
        
        print("=" * 60)
        print("实时人脸交换系统已启动")
        print("=" * 60)
        print("操作说明:")
        print("  按 'q' 键 - 退出程序")
        print("  按 's' 键 - 保存当前帧")
        print("  按 'f' 键 - 切换全屏模式")
        print("  按 'r' 键 - 显示/隐藏信息")
        print("=" * 60 + "\n")
        
        window_name = "Real-time Face Swap Camera"
        fullscreen = False
        show_info = True
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("警告：无法读取摄像头画面")
                    break
                
                # 进行人脸交换
                processed_frame = self.faceswap.frame_swap(frame)
                process_count += 1
                frame_count += 1
                
                # 显示信息
                if show_info:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    process_fps = process_count / elapsed if elapsed > 0 else 0
                    
                    # 添加文字信息
                    cv2.putText(processed_frame, f"Capture FPS: {current_fps:.1f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Process FPS: {process_fps:.1f}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.putText(processed_frame, f"Frame: {frame_count}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Resolution: {processed_frame.shape[1]}x{processed_frame.shape[0]}", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Device: {self.faceswap.device}", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(processed_frame, "Press 'q' to quit, 's' to save, 'f' for fullscreen, 'r' to toggle info",
                               (10, processed_frame.shape[0] - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # 显示画面
                if fullscreen:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                
                cv2.imshow(window_name, processed_frame)
                
                # 处理按键
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n用户按下 'q' 键，退出...")
                    break
                elif key == ord('s'):
                    filename = f"faceswap_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(filename, processed_frame)
                    print(f"已保存画面到: {filename}")
                elif key == ord('f'):
                    fullscreen = not fullscreen
                    print(f"全屏模式: {'开启' if fullscreen else '关闭'}")
                elif key == ord('r'):
                    show_info = not show_info
                    print(f"信息显示: {'开启' if show_info else '关闭'}")
        
        except KeyboardInterrupt:
            print("\n接收到中断信号，退出...")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n摄像头已关闭，资源已释放")

def main():
    parser = argparse.ArgumentParser(description='实时人脸交换摄像头系统（GPU加速）')
    parser.add_argument('--source',default="/home1/Code/MobileFaceswap/demo_file/6.jpg", type=str, 
                       help='源人脸图片路径')
    parser.add_argument('--camera', type=int, default=0, 
                       help='摄像头ID (默认: 0)')
    parser.add_argument('--width', type=int, default=960, 
                       help='图像宽度 (默认: 1920)')
    parser.add_argument('--height', type=int, default=640, 
                       help='图像高度 (默认: 1080)')
    parser.add_argument('--fps', type=int, default=25, 
                       help='帧率 (默认: 30)')
    parser.add_argument('--no-gstreamer', action='store_true', 
                       help='禁用GStreamer，使用标准OpenCV')
    parser.add_argument('--cpu', action='store_true', 
                       help='强制使用CPU（默认使用GPU如果可用）')
    
    args = parser.parse_args()
    
    # 检查源图片是否存在
    if not os.path.exists(args.source):
        print(f"错误：源人脸图片不存在: {args.source}")
        sys.exit(1)
    
    # 创建实时人脸交换系统
    camera_system = RealtimeFaceSwapCamera(
        source_img_path=args.source,
        camera_id=args.camera,
        width=args.width,
        height=args.height,
        fps=args.fps,
        use_gstreamer=not args.no_gstreamer,
        use_gpu=not args.cpu
    )
    
    camera_system.run()

if __name__ == '__main__':
    main()

