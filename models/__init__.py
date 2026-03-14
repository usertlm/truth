"""
Truth - 图像篡改检测系统
"""

from models.traditional_detector import TraditionalDetector, create_traditional_detector
from models.transformer_detector import TamperingTransformer, create_vit_detector
from models.gan_detector import TamperingGAN, create_gan
from models.integrated_detector import TruthDetector, create_detector

__all__ = [
    'TraditionalDetector',
    'create_traditional_detector',
    'TamperingTransformer', 
    'create_vit_detector',
    'TamperingGAN',
    'create_gan',
    'TruthDetector',
    'create_detector'
]

__version__ = '1.0.0'
