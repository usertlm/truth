"""
Transformer 增强模块
用于图像篡改检测的 Vision Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """将图像切分为 patches 并嵌入"""
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    
    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer 编码器块"""
    
    def __init__(self, embed_dim, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformerEncoder(nn.Module):
    """Vision Transformer 编码器"""
    
    def __init__(
        self, 
        img_size=224, 
        patch_size=16, 
        in_chans=3, 
        embed_dim=768,
        depth=12,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        self.blocks = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # 初始化位置编码
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x):
        B = x.shape[0]
        
        x = self.patch_embed(x)
        
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        x = self.blocks(x)
        x = self.norm(x)
        
        return x


class TamperingTransformer(nn.Module):
    """
    篡改检测 Transformer 模型
    结合 CNN 特征和 Transformer 注意力
    """
    
    def __init__(
        self,
        cnn_backbone='convnext_tiny',
        embed_dim=768,
        depth=6,
        num_heads=8,
        num_classes=1
    ):
        super().__init__()
        
        # CNN 特征提取器
        if cnn_backbone == 'convnext_tiny':
            from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
            weights = ConvNeXt_Tiny_Weights.DEFAULT
            self.cnn = convnext_tiny(weights=weights)
            cnn_embed_dim = 768
        else:
            raise ValueError(f"Unsupported backbone: {cnn_backbone}")
        
        # 投影层
        self.proj = nn.Linear(cnn_embed_dim, embed_dim)
        
        # Transformer
        self.transformer = nn.Sequential(*[
            TransformerBlock(embed_dim, num_heads)
            for _ in range(depth)
        ])
        
        # 分类头
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # 分割头 (用于生成篡改掩码)
        self.seg_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        # CNN 特征
        cnn_features = self.cnn.features(x)
        
        # 全局池化
        pooled = F.adaptive_avg_pool2d(cnn_features, 1).flatten(1)
        
        # 投影到 embed dim
        x = self.proj(pooled).unsqueeze(1)  # [B, 1, embed_dim]
        
        # Transformer
        x = self.transformer(x)
        
        # 分类输出
        cls_output = x[:, 0]
        cls_logit = self.head(cls_output)
        
        # 分割输出
        seg_output = self.seg_head(x)
        
        return {
            'classification': torch.sigmoid(cls_logit),
            'segmentation': torch.sigmoid(seg_output)
        }


def create_vit_detector(pretrained=True):
    """创建 Transformer 检测器"""
    return TamperingTransformer(cnn_backbone='convnext_tiny')


if __name__ == '__main__':
    model = create_vit_detector(pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Classification: {output['classification'].shape}")
    print(f"Segmentation: {output['segmentation'].shape}")
