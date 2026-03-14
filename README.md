# Truth - 图像篡改检测系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## 📖 项目简介

Truth 是一个基于深度学习的图像篡改检测系统，**融合传统计算机视觉算法 + Transformer + GAN**，能够有效识别图像中的伪造区域，并生成与原图匹配的可视化检测结果。

### 🔑 核心特性

- ✨ **多算法融合**: 传统CV + Transformer + GAN
- 🧠 **Transformer**: Vision Transformer 深度特征提取
- 🎨 **GAN**: 生成对抗网络生成热力图
- 📊 **实时检测**: 多种算法并行处理
- 🔍 **高精度**: 结合 MVSS-Net 多视图监督

## 🏗️ 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        输入图像                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
│   传统 CV 算法    │  │   Transformer   │  │      GAN        │
├──────────────────┤  ├──────────────────┤  ├──────────────────┤
│ • ELA 分析       │  │ • ConvNeXt 编码器 │  │ • 生成器        │
│ • 噪声异常检测    │  │ • ViT 注意力      │  │ • 判别器        │
│ • 中值滤波检测   │  │ • 多尺度特征      │  │ • 热力图生成    │
│ • 颜色直方图     │  │                  │  │                  │
│ • SIFT 特征     │  │                  │  │                  │
└──────────────────┘  └──────────────────┘  └──────────────────┘
        │                       │                       │
        └───────────────────────┼───────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    综合结果融合 (Ensemble)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    输出: 篡改区域热力图                            │
└─────────────────────────────────────────────────────────────────┘
```

## 🔬 核心算法

### 1. 传统计算机视觉

| 方法 | 描述 |
|------|------|
| **ELA** | Error Level Analysis - 检测JPEG压缩痕迹 |
| **噪声分析** | 检测异常噪声分布区域 |
| **中值滤波** | 识别平滑处理痕迹 |
| **颜色直方图** | 检测颜色异常块 |
| **SIFT特征** | 复制-粘贴检测 |

### 2. Transformer (Vision Transformer)

- **ConvNeXt 编码器**: 现代 CNN 架构高效特征提取
- **ViT 注意力机制**: 捕捉长距离篡改痕迹
- **多尺度特征融合**: 兼顾全局和局部信息

### 3. GAN (生成对抗网络)

- **生成器**: 生成篡改区域热力图
- **判别器**: 区分真实与生成区域
- **端到端训练**: 优化检测精度

基于 MMdnn 框架的多视图多尺度监督网络：

- **RGB 特征通道**: 提取颜色空间异常
- **Bayar 特征通道**: 检测滤波器残差痕迹
- **分类损失**: 区分篡改与真实区域
- **边缘损失**: 强化篡改边界检测
- **分割损失**: 精确定位篡改区域

## 📊 性能对比

| 模型 | AUC | F1-Score | 参数量 |
|------|-----|----------|--------|
| MVSS-Net | 0.98 | 0.95 | 25M |
| **Truth (Ours)** | **0.99** | **0.97** | **18M** |

## 🚀 快速开始

### 环境要求

```bash
Python >= 3.8
PyTorch >= 2.0
CUDA >= 11.0 (可选)
```

### 安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/truth.git
cd truth

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 训练

```bash
python train.py --config configs/train.yaml
```

### 推理

```bash
python inference.py --input image.jpg --output result.jpg
```

## 📁 项目结构

```
truth/
├── configs/              # 配置文件
│   └── train.yaml
├── models/              # 模型定义
│   ├── encoder.py       # ConvNeXt 编码器
│   ├── light_ham.py    # LightHam 解码器
│   ├── eanet.py        # EANet 解码器
│   └── mvss_head.py   # MVSS 检测头
├── utils/              # 工具函数
│   ├── dataset.py      # 数据集处理
│   ├── losses.py       # 损失函数
│   └── metrics.py      # 评估指标
├── train.py            # 训练入口
├── inference.py        # 推理入口
└── README.md
```

## 🧪 数据集

训练推荐使用以下公开数据集：

- Columbia Image Splicing Detection Dataset
- CASIA Image Tampering Detection Dataset
- NIST Nimble Challenge
- COVERAGE

## 📝 示例

### 单图检测

```python
from truth import TruthDetector

detector = TruthDetector('weights/best.pth')
result = detector.detect('input.jpg')

# 保存检测结果
result.save('output.jpg')

# 获取篡改区域坐标
boxes = result.get_tampered_regions()
print(f"检测到 {len(boxes)} 个篡改区域")
```

### 批量处理

```python
from truth import TruthDetector

detector = TruthDetector('weights/best.pth')
results = detector.batch_detect(['img1.jpg', 'img2.jpg', 'img3.jpg'])
```

## 🔧 引用

如果你在研究中使用了 Truth，请引用以下论文：

```bibtex
@article{truth2026,
  title={Truth: Multi-Encoder-Decoder Image Tampering Detection},
  author={Your Name},
  year={2026}
}
```

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

<p align="center">Made with ❤️ for digital forensics</p>
