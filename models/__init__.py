"""
Truth - 图像篡改检测模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtEncoder(nn.Module):
    """ConvNeXt 编码器"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        # 使用 torchvision 的 ConvNeXt
        try:
            from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
            self.backbone = convnext_tiny(weights=weights)
            self.feature_dims = [96, 192, 384, 768]
        except ImportError:
            raise ImportError("Please install torchvision: pip install torchvision")
    
    def forward(self, x):
        """返回多尺度特征"""
        features = []
        x = self.backbone.features(x)
        # 解析特征
        return x


class LightHamDecoder(nn.Module):
    """LightHam 解码器 - 轻量级混合注意力"""
    
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        self.output = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        attn = self.attention(x)
        x = x * attn
        return self.output(x)


class EANetDecoder(nn.Module):
    """EANet 解码器 - 边缘感知注意力网络"""
    
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1)
        )
        self.attention = nn.Conv2d(in_channels, 1, 1)
        self.output = nn.Conv2d(in_channels, out_channels, 1)
    
    def forward(self, x):
        edge = self.edge_conv(x)
        attn = torch.sigmoid(self.attention(x))
        x = x * attn + edge
        return self.output(x)


class TruthDetector(nn.Module):
    """Truth 篡改检测主模型"""
    
    def __init__(self, pretrained=True):
        super().__init__()
        self.encoder = ConvNeXtEncoder(pretrained=pretrained)
        self.light_ham = LightHamDecoder(768, 1)
        self.eanet = EANetDecoder(768, 1)
        
    def forward(self, x):
        features = self.encoder(x)
        
        # 并行解码
        light_ham_out = self.light_ham(features)
        eanet_out = self.eanet(features)
        
        # 特征融合
        fused = (light_ham_out + eanet_out) / 2
        
        return {
            'mask': torch.sigmoid(fused),
            'light_ham': torch.sigmoid(light_ham_out),
            'eanet': torch.sigmoid(eanet_out)
        }


def create_model(pretrained=True):
    """创建模型实例"""
    return TruthDetector(pretrained=pretrained)


if __name__ == '__main__':
    # 测试模型
    model = create_model(pretrained=False)
    x = torch.randn(1, 3, 256, 256)
    output = model(x)
    print(f"Output shape: {output['mask'].shape}")
    print("Model created successfully!")
