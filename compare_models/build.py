from .UNet3 import UNet3
from .ResUnet import ResUNet
# from .ResACEUnet import ResACEUNet2  # 暂时跳过
from .SwinUNETR import SwinUNETR
import torch
import torch.nn as nn


class ModelWithSigmoid(nn.Module):
    """为对比模型添加 Sigmoid 输出，并强制限制在 [0,1] 范围内"""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        output = torch.sigmoid(self.model(x))
        # 强制限制在 [0, 1] 范围内，防止CUDA断言错误
        return torch.clamp(output, min=0.0, max=1.0)


def build_model(config):
    model_type = config.model.name
    # ResACEUNET 暂时跳过
    if model_type == 'RESUNET':
        model = ResUNet(
            n_channels=config.model.in_chans,
            n_classes=config.model.num_classes
        )
    elif model_type == 'UNET3':
        model = UNet3(
            n_channels=config.model.in_chans,
            n_classes=config.model.num_classes
        )
    elif model_type == 'SWINUNETR':
        # 修复 MONAI v1.5 兼容性问题 - img_size 仍需要但只用于检查
        model = SwinUNETR(
            img_size=config.data.img_size,  # 需要但不再强制检查
            in_channels=config.model.in_chans,
            out_channels=config.model.num_classes,
            feature_size=48,
            drop_rate=config.model.drop,
            attn_drop_rate=config.model.attn_drop,
            dropout_path_rate=config.model.drop_path
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    # 添加 Sigmoid 包装，确保输出在 [0,1] 范围内
    return ModelWithSigmoid(model)
