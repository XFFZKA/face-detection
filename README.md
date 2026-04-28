# 基于嵌入式平台的实时人脸匿名化系统

## 📌 项目简介
本项目实现了一套面向嵌入式边缘计算平台的实时人脸匿名化系统。系统采用端到端处理流程，对输入视频流中的人脸进行检测、对齐、身份特征提取与人脸生成，从而在保证视觉效果的同时实现身份隐私保护。

该系统兼顾实时性与生成质量，适用于隐私保护、视频处理及嵌入式视觉应用等场景。

---

## 🚀 主要功能
- 实时人脸匿名化处理
- 模块化系统设计（检测 / 对齐 / 特征 / 生成）
- 支持嵌入式平台部署
- GPU加速推理
- 多摄像头支持（CSI / USB）
- 长时间稳定运行与异常容错机制

---

## 🧩 系统流程
1. 人脸检测与关键点定位  
2. 仿射变换实现人脸对齐  
3. 身份特征提取  
4. 人脸生成  
5. 掩膜融合与逆变换  
6. 输出结果  

---

## ⚙️ 安装方法

### 📦1. 克隆项目
```bash
git clone https://github.com/XFFZKA/face-protection.git

cd face-protectionxxxxxxxxxx git clone https://github.com/XFFZKA/face-protection.gitcd face-protectionbash
```

### 📦2. 安装依赖
```bash
pip install -r requirements.txt
```

### 📦3. 启动程序

```python
python realtime_faceswap_camera.py
```



