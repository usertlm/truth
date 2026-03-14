"""
传统图像篡改检测算法
结合 ELA, 噪声分析, 中值滤波检测, 颜色直方图等方法
"""

import cv2
import numpy as np
from pathlib import Path


class TraditionalDetector:
    """传统图像篡改检测器"""
    
    def __init__(self):
        self.results = {}
    
    def detect_ela(self, image_path, quality=90):
        """
        Error Level Analysis (ELA)
        检测JPEG压缩痕迹
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # JPEG 压缩模拟
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', gray, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
        
        # 计算差异
        ela = cv2.absdiff(gray, decoded)
        ela = cv2.normalize(ela, None, 0, 255, cv2.NORM_MINMAX)
        
        # 阈值处理
        _, mask = cv2.threshold(ela, 15, 255, cv2.THRESH_BINARY)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask, ela
    
    def detect_noise_anomaly(self, image_path):
        """
        噪声异常检测
        篡改区域往往有异常的噪声分布
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        
        # 局部标准差
        kernel = 5
        mean = cv2.blur(gray, (kernel, kernel))
        variance = cv2.blur((gray - mean) ** 2, (kernel, kernel))
        std_dev = np.sqrt(variance)
        
        std_norm = cv2.normalize(std_dev, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        _, mask = cv2.threshold(std_norm, 40, 255, cv2.THRESH_BINARY)
        
        return mask, std_norm
    
    def detect_median_filter(self, image_path):
        """
        中值滤波检测
        常用于隐藏篡改痕迹
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 应用中值滤波
        filtered = cv2.medianBlur(gray, 3)
        
        diff = cv2.absdiff(gray, filtered)
        _, mask = cv2.threshold(diff, 3, 255, cv2.THRESH_BINARY)
        
        return mask, diff
    
    def detect_clone_detection(self, image_path):
        """
        复制-粘贴检测
        使用 SIFT 特征检测重复区域
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None, 0
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # SIFT 特征检测
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        
        result = img.copy()
        cv2.drawKeypoints(
            img, keypoints, result, 
            (0, 255, 0), 
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )
        
        return result, len(keypoints)
    
    def detect_color_anomaly(self, image_path, block_size=32):
        """
        颜色直方图异常检测
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None, None
        
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for y in range(0, h - block_size, block_size):
            for x in range(0, w - block_size, block_size):
                block = img[y:y+block_size, x:x+block_size]
                
                for i in range(3):
                    hist = cv2.calcHist([block], [i], None, [32], [0, 256])
                    peak = hist.max() / (hist.sum() + 1e-6)
                    
                    if peak > 8:
                        mask[y:y+block_size, x:x+block_size] = 255
                        break
        
        return mask, None
    
    def detect_laplacian_variance(self, image_path):
        """
        Laplacian 方差检测
        模糊区域方差较小
        """
        img = cv2.imread(str(image_path))
        if img is None:
            return None, None
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance_map = cv2.blur(laplacian ** 2, (5, 5))
        
        variance_norm = cv2.normalize(variance_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return variance_map, variance_norm
    
    def full_analysis(self, image_path):
        """
        综合分析
        """
        results = {}
        
        print(f"🔍 Analyzing: {Path(image_path).name}")
        
        # ELA
        mask, ela = self.detect_ela(image_path)
        if mask is not None:
            results['ela'] = mask
            print(f"  ELA: {cv2.countNonZero(mask)} suspicious pixels")
        
        # Noise
        mask, noise = self.detect_noise_anomaly(image_path)
        if mask is not None:
            results['noise'] = mask
            print(f"  Noise: {cv2.countNonZero(mask)} anomaly regions")
        
        # Median filter
        mask, median = self.detect_median_filter(image_path)
        if mask is not None:
            results['median'] = mask
            print(f"  Median: {cv2.countNonZero(mask)} filtered regions")
        
        # Color
        mask, _ = self.detect_color_anomaly(image_path)
        if mask is not None:
            results['color'] = mask
            print(f"  Color: {cv2.countNonZero(mask)} anomaly blocks")
        
        # Laplacian
        variance, variance_norm = self.detect_laplacian_variance(image_path)
        if variance is not None:
            results['laplacian'] = variance_norm
        
        # SIFT
        sift_img, num_kp = self.detect_clone_detection(image_path)
        results['sift'] = (sift_img, num_kp)
        print(f"  SIFT: {num_kp} keypoints")
        
        # Combined
        combined = np.zeros_like(list(results.values())[0])
        for name, data in results.items():
            if isinstance(data, np.ndarray) and len(data.shape) == 2:
                combined = cv2.add(combined, data)
        combined = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        results['combined'] = combined
        
        return results


def create_traditional_detector():
    """创建传统检测器"""
    return TraditionalDetector()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        detector = TraditionalDetector()
        results = detector.full_analysis(sys.argv[1])
        print("\n✅ Analysis complete!")
