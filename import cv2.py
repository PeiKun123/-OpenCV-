import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os
import subprocess

def preview_and_capture():
    """使用libcamera拍摄图像"""
    save_dir = os.path.expanduser("~/Desktop/pictures")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(save_dir, f"capture_{timestamp}.jpg")
    
    try:
        cmd = f"libcamera-still -t 0 --timelapse 100 -o {output_path} --immediate"
        print("准备拍摄，按Ctrl+C拍照并退出预览")
        subprocess.run(cmd, shell=True)
        
        if os.path.exists(output_path):
            print(f"图像已保存至: {output_path}")
            return output_path
        else:
            print("拍摄失败")
            return None
    except KeyboardInterrupt:
        if os.path.exists(output_path):
            print(f"图像已保存至: {output_path}")
            return output_path
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None

def process_image(image):
    """改进的图像预处理函数"""
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 使用更大的核进行高斯模糊降噪
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 添加中值滤波进一步去噪
    denoised = cv2.medianBlur(denoised, 3)
    
    # CLAHE改善对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 使用自适应阈值，调整参数
    processed = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,  # 增大邻域大小
        4    # 调整常数减数
    )
    
    # 使用形态学操作清理噪声
    kernel = np.ones((2,2), np.uint8)
    processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
    processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
    
    # 再次应用中值滤波去除剩余噪点
    processed = cv2.medianBlur(processed, 3)
    
    return enhanced, processed

def analyze_text(processed_img):
    """改进的文字分析函数"""
    # 使用原始图像进行轮廓检测
    contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        # 调整区域筛选条件
        if area > 50 and w > 5 and h > 5:
            text_regions.append({
                'position': (x, y),
                'size': (w, h),
                'area': area
            })
    return text_regions

def save_results(text_regions, save_dir):
    """保存分析结果"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(save_dir, f'text_analysis_{timestamp}.csv')
    
    with open(filename, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['检测时间', '区域编号', 'X位置', 'Y位置', '宽度', '高度', '面积'])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for idx, region in enumerate(text_regions, 1):
            writer.writerow([
                timestamp,
                idx,
                region['position'][0],
                region['position'][1],
                region['size'][0],
                region['size'][1],
                region['area']
            ])
    print(f"分析结果已保存至: {filename}")

def main():
    print("按Ctrl+C进行拍照")
    captured_image_path = preview_and_capture()
    
    if captured_image_path is None:
        print("未拍摄图片，程序退出")
        return

    original = cv2.imread(captured_image_path)
    if original is None:
        print("无法读取已拍摄的图像")
        return

    # 图像处理
    enhanced, processed = process_image(original)
    
    # 计算SSIM
    ssim_index = ssim(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), enhanced)
    print(f'SSIM Index: {ssim_index}')
    
    text_regions = analyze_text(processed)

    # 显示处理结果
        # 在显示处理结果部分修改代码
    plt.figure(figsize=(15, 5))
    
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(enhanced, cmap='gray')
    plt.title(f'增强后的图像 (SSIM: {ssim_index:.3f})')  # 在标题中显示SSIM值
    plt.axis('off')
    
    plt.subplot(133)
    # 创建白色背景
    result_img = np.ones_like(processed) * 255
    # 将处理后的图像叠加到白色背景上
    result_img = cv2.bitwise_and(result_img, processed)
    result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
    
    for region in text_regions:
        x, y = region['position']
        w, h = region['size']
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 128, 0), 1)
    
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title('文字识别结果')
    plt.axis('off')
    
    plt.tight_layout()
    
    # 保存处理后的图像
    save_dir = os.path.expanduser("~/Desktop/pictures")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_image_path = os.path.join(save_dir, f'processed_{timestamp}.jpg')
    cv2.imwrite(processed_image_path, result_img)
    print(f"处理后的图像已保存至: {processed_image_path}")
    
    plt.show()
    
    # 保存分析结果到CSV
    save_results(text_regions, save_dir)

if __name__ == "__main__":
    main()