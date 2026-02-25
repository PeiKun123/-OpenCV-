基于 Python OpenCV 的图像采集与文字区域检测工具，专为嵌入式 Linux (如 Raspberry Pi) 设计。
功能特点
图像采集: 调用 libcamera-still 进行预览与抓拍。
图像增强: 集成高斯/中值滤波、CLAHE 对比度增强、自适应阈值二值化及形态学去噪。
区域检测: 自动识别文字/物体轮廓，计算位置与面积。
质量评估: 输出 SSIM 指数以量化增强效果。
数据导出: 自动生成标注图片及 CSV 分析报告。
环境依赖
系统: Linux (推荐 Raspberry Pi OS)
硬件: 兼容 libcamera 的摄像头模块
软件: Python 3.8+, libcamera-apps
Python 库:pip install opencv-python scikit-image matplotlib numpy
 **操作流程**:
- 程序启动后进入相机预览。
- 按 `Ctrl+C` 拍照并退出预览。
- 程序自动处理图像，并在 `~/Desktop/pictures` (或 `~/pictures`) 保存结果。

 **输出文件**:
- `capture_*.jpg`: 原始图像
- `processed_*.jpg`: 带检测框的处理结果图
- `text_analysis_*.csv`: 包含坐标和面积数据的表格
- `*_report.png`: 三步处理流程对比图

