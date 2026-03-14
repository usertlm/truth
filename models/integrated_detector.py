"""
Truth - 综合图像篡改检测系统
结合传统算法 + Transformer + GAN
"""

import cv2
import numpy as np
import torch
from pathlib import Path

from models.traditional_detector import TraditionalDetector
from models.transformer_detector import create_vit_detector
from models.gan_detector import create_gan


class TruthDetector:
    """
    综合图像篡改检测器
    融合多种算法：
    1. 传统 CV (ELA, 噪声, 中值滤波)
    2. Transformer (深度学习)
    3. GAN (生成对抗)
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.traditional = TraditionalDetector()
        
        # 加载深度学习模型
        try:
            self.transformer = create_vit_detector(pretrained=True)
            self.transformer.to(device)
            self.transformer.eval()
        except Exception as e:
            print(f"Warning: Could not load Transformer model: {e}")
            self.transformer = None
        
        try:
            self.gan = create_gan(device=device)
            # 尝试加载预训练权重
            try:
                self.gan.load('weights/gan_tampering.pth')
            except:
                print("Warning: GAN weights not found, using untrained model")
        except Exception as e:
            print(f"Warning: Could not load GAN model: {e}")
            self.gan = None
    
    def detect(self, image_path, save_path=None):
        """
        综合检测
        """
        results = {}
        
        # 1. 传统算法检测
        print("🔍 Running traditional detection...")
        traditional_results = self.traditional.full_analysis(image_path)
        results['traditional'] = traditional_results
        
        # 2. Transformer 检测
        if self.transformer:
            print("🧠 Running Transformer detection...")
            vit_result = self._detect_transformer(image_path)
            results['transformer'] = vit_result
        
        # 3. GAN 生成热力图
        if self.gan:
            print("🎨 Generating GAN heatmap...")
            gan_heatmap = self._detect_gan(image_path)
            results['gan'] = gan_heatmap
        
        # 4. 综合结果
        print("⚖️ Computing ensemble result...")
        ensemble = self._ensemble_results(results)
        results['ensemble'] = ensemble
        
        # 保存结果
        if save_path:
            self._save_results(results, save_path)
        
        return results
    
    def _detect_transformer(self, image_path):
        """Transformer 检测"""
        from PIL import Image
        import torchvision.transforms as transforms
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        
        img = Image.open(image_path).convert('RGB')
        x = transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.transformer(x)
        
        mask = output['segmentation'].squeeze().cpu().numpy()
        
        # 调整大小
        orig_img = cv2.imread(str(image_path))
        mask = cv2.resize(mask, (orig_img.shape[1], orig_img.shape[0]))
        
        return mask
    
    def _detect_gan(self, image_path):
        """GAN 检测"""
        heatmap = self.gan.generate_heatmap(image_path)
        
        # 调整大小
        orig_img = cv2.imread(str(image_path))
        heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
        
        return heatmap
    
    def _ensemble_results(self, results):
        """综合多个检测结果"""
        masks = []
        
        # 收集所有掩码
        if 'traditional' in results and 'combined' in results['traditional']:
            combined = results['traditional']['combined']
            if combined is not None:
                masks.append(combined)
        
        if 'transformer' in results:
            vit_mask = results['transformer']
            if vit_mask is not None:
                vit_mask = (vit_mask * 255).astype(np.uint8)
                masks.append(vit_mask)
        
        if 'gan' in results:
            gan_mask = results['gan']
            if gan_mask is not None:
                gan_mask = (gan_mask * 255).astype(np.uint8)
                masks.append(gan_mask)
        
        if not masks:
            return None
        
        # 平均融合
        ensemble = np.mean(masks, axis=0).astype(np.uint8)
        
        # 阈值处理
        _, ensemble = cv2.threshold(ensemble, 128, 255, cv2.THRESH_BINARY)
        
        return ensemble
    
    def _save_results(self, results, save_path):
        """保存检测结果"""
        cv2.imwrite(str(save_path), results['ensemble'])
    
    def visualize(self, image_path, output_dir='output'):
        """
        可视化检测结果
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        name = Path(image_path).stem
        
        results = self.detect(image_path)
        
        # 读取原图
        img = cv2.imread(str(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原图
        axes[0, 0].imshow(img_rgb)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        # ELA
        if 'traditional' in results and 'ela' in results['traditional']:
            axes[0, 1].imshow(results['traditional']['ela'], cmap='hot')
            axes[0, 1].set_title('ELA')
            axes[0, 1].axis('off')
        
        # Noise
        if 'traditional' in results and 'noise' in results['traditional']:
            axes[0, 2].imshow(results['traditional']['noise'], cmap='hot')
            axes[0, 2].set_title('Noise Anomaly')
            axes[0, 2].axis('off')
        
        # Transformer
        if 'transformer' in results:
            axes[1, 0].imshow(results['transformer'], cmap='hot')
            axes[1, 0].set_title('Transformer')
            axes[1, 0].axis('off')
        
        # GAN
        if 'gan' in results:
            axes[1, 1].imshow(results['gan'], cmap='hot')
            axes[1, 1].set_title('GAN')
            axes[1, 1].axis('off')
        
        # Ensemble
        if 'ensemble' in results and results['ensemble'] is not None:
            ensemble_color = cv2.applyColorMap(results['ensemble'], cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(img_rgb, 0.7, cv2.cvtColor(ensemble_color, cv2.COLOR_BGR2RGB), 0.3, 0)
            axes[1, 2].imshow(overlay)
            axes[1, 2].set_title('Ensemble Result')
            axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{name}_detection.png")
        print(f"✅ Visualization saved to {output_dir}/{name}_detection.png")


def create_detector(device='cuda'):
    """创建综合检测器"""
    return TruthDetector(device=device)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        detector = create_detector()
        detector.visualize(sys.argv[1])
