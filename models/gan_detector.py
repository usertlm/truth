"""
GAN 模块
用于生成篡改区域可视化对比图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """生成器 - 生成篡改区域热力图"""
    
    def __init__(self, in_channels=3, out_channels=1, base_dim=64):
        super().__init__()
        
        # 编码器
        self.enc1 = self._conv_block(in_channels, base_dim, stride=2)
        self.enc2 = self._conv_block(base_dim, base_dim * 2, stride=2)
        self.enc3 = self._conv_block(base_dim * 2, base_dim * 4, stride=2)
        self.enc4 = self._conv_block(base_dim * 4, base_dim * 8, stride=2)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_dim * 8, base_dim * 8, 3, padding=1),
            nn.BatchNorm2d(base_dim * 8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim * 8, base_dim * 8, 3, padding=1),
            nn.BatchNorm2d(base_dim * 8),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.dec1 = self._upconv_block(base_dim * 8, base_dim * 4)
        self.dec2 = self._upconv_block(base_dim * 4, base_dim * 2)
        self.dec3 = self._upconv_block(base_dim * 2, base_dim)
        self.dec4 = self._upconv_block(base_dim, base_dim // 2)
        
        # 输出
        self.output = nn.Conv2d(base_dim // 2, out_channels, 1)
    
    def _conv_block(self, in_channels, out_channels, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def _upconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        # 编码
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        # 瓶颈
        b = self.bottleneck(e4)
        
        # 解码
        d1 = self.dec1(b)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        d4 = self.dec4(d3)
        
        return torch.sigmoid(self.output(d4))


class Discriminator(nn.Module):
    """判别器 - 区分真实篡改区域和生成区域"""
    
    def __init__(self, in_channels=4, base_dim=64):
        super().__init__()
        
        self.model = nn.Sequential(
            self._conv_block(in_channels, base_dim, stride=2),
            self._conv_block(base_dim, base_dim * 2, stride=2),
            self._conv_block(base_dim * 2, base_dim * 4, stride=2),
            self._conv_block(base_dim * 4, base_dim * 8, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_dim * 8, 1),
            nn.Sigmoid()
        )
    
    def _conv_block(self, in_channels, out_channels, stride=2):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        return self.model(x)


class TamperingGAN:
    """
    篡改检测 GAN
    用于生成高质量篡改区域可视化
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.generator = Generator().to(device)
        self.discriminator = Discriminator().to(device)
        
        # 优化器
        self.g_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.bce_loss = nn.BCELoss()
    
    def train_step(self, real_images, real_masks, fake_masks):
        """
        训练一步
        """
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, 1).to(self.device)
        fake_labels = torch.zeros(batch_size, 1).to(self.device)
        
        # 训练判别器
        self.d_optimizer.zero_grad()
        
        # 真实样本
        real_input = torch.cat([real_images, real_masks], dim=1)
        real_output = self.discriminator(real_input)
        d_loss_real = self.bce_loss(real_output, real_labels)
        
        # 生成样本
        fake_input = torch.cat([real_images, fake_masks.detach()], dim=1)
        fake_output = self.discriminator(fake_input)
        d_loss_fake = self.bce_loss(fake_output, fake_labels)
        
        d_loss = (d_loss_real + d_loss_fake) / 2
        d_loss.backward()
        self.d_optimizer.step()
        
        # 训练生成器
        self.g_optimizer.zero_grad()
        
        fake_masks_generated = self.generator(real_images)
        fake_input = torch.cat([real_images, fake_masks_generated], dim=1)
        fake_output = self.discriminator(fake_input)
        
        g_loss = self.bce_loss(fake_output, real_labels)
        g_loss.backward()
        self.g_optimizer.step()
        
        return {
            'd_loss': d_loss.item(),
            'g_loss': g_loss.item()
        }
    
    def generate_heatmap(self, image):
        """
        生成篡改区域热力图
        """
        self.generator.eval()
        with torch.no_grad():
            if isinstance(image, str):
                from PIL import Image
                import torchvision.transforms as transforms
                img = Image.open(image).convert('RGB')
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),
                    transforms.ToTensor()
                ])
                image = transform(img).unsqueeze(0).to(self.device)
            
            heatmap = self.generator(image)
            return heatmap.squeeze().cpu().numpy()
    
    def save(self, path):
        """保存模型"""
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'g_optimizer': self.g_optimizer.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer'])


def create_gan(device='cuda'):
    """创建 GAN 模型"""
    return TamperingGAN(device=device)


if __name__ == '__main__':
    gan = create_gan()
    print("GAN created successfully!")
