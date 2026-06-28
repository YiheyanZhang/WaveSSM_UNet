import sys
sys.path.append('./models')

import itertools
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from .wavelet_analyze import MultiScaleWaveletFeatures

# 尝试导入mamba-ssm
try:
    from mamba_ssm import Mamba
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Use: pip install mamba-ssm")


# ==================== 手动实现3D可变形卷积 ====================
class DeformConv3d(nn.Module):
    """
    完整版3D可变形卷积
    
    核心思想:
    - 学习偏移量,自适应采样位置
    - 使用grid_sample进行双线性插值
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, bias=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.num_kernels = kernel_size ** 3
        
        # 权重: [out_ch, in_ch/groups, K, K, K]
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels // groups, kernel_size, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # 偏移量卷积
        self.conv_offset = nn.Conv3d(
            in_channels, self.num_kernels * 3,
            kernel_size, stride=stride, padding=padding, bias=True
        )
        
        # 初始化
        nn.init.zeros_(self.conv_offset.weight)
        nn.init.zeros_(self.conv_offset.bias)
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
        
    def forward(self, x):
        B, C_in, D, H, W = x.shape
        K = self.kernel_size
        
        # 预测偏移
        offset = self.conv_offset(x)
        
        # 计算输出尺寸
        D_out = (D + 2 * self.padding - K) // self.stride + 1
        H_out = (H + 2 * self.padding - K) // self.stride + 1
        W_out = (W + 2 * self.padding - K) // self.stride + 1
        
        # 完整实现: 对每个采样位置进行偏移采样
        output = self._deform_sample(x, offset, D_out, H_out, W_out)
        
        return output
    
    def _deform_sample(self, x, offset, D_out, H_out, W_out):
        """完整版可变形采样 - 对每个 kernel 位置独立进行 trilinear 插值采样"""
        B, C_in, D, H, W = x.shape
        K = self.kernel_size
        device = x.device

        offset = offset.view(B, K**3, 3, D_out, H_out, W_out)

        output = torch.zeros(B, self.out_channels, D_out, H_out, W_out, device=device)

        for idx, (di, hi, wi) in enumerate(itertools.product(range(K), repeat=3)):
            w = self.weight[:, :, di, hi, wi]

            d_base = torch.arange(D_out, device=device) * self.stride - self.padding + di
            h_base = torch.arange(H_out, device=device) * self.stride - self.padding + hi
            w_base = torch.arange(W_out, device=device) * self.stride - self.padding + wi

            d_coord = d_base[None, :, None, None] + offset[:, idx, 0]
            h_coord = h_base[None, None, :, None] + offset[:, idx, 1]
            w_coord = w_base[None, None, None, :] + offset[:, idx, 2]

            d_norm = 2 * d_coord / (D - 1) - 1
            h_norm = 2 * h_coord / (H - 1) - 1
            w_norm = 2 * w_coord / (W - 1) - 1

            grid = torch.stack([w_norm, h_norm, d_norm], dim=-1)

            sampled = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

            if self.groups == 1:
                output += torch.einsum('bcdhw,oc->bodhw', sampled, w)
            else:
                C_in_g = C_in // self.groups
                C_out_g = self.out_channels // self.groups
                sampled_g = sampled.view(B, self.groups, C_in_g, D_out, H_out, W_out)
                w_g = w.view(self.groups, C_out_g, C_in_g)
                output += torch.einsum('bgcdhw,goc->bgodhw', sampled_g, w_g).reshape(
                    B, self.out_channels, D_out, H_out, W_out
                )

        if self.bias is not None:
            output += self.bias.view(1, -1, 1, 1, 1)

        return output


# 保留原始FAM作为备选
class ResidualDeformConv(nn.Module):
    """
    残差可变形卷积 (V15)
    
    标准卷积 + 可变形卷积残差分支
    
    out = DoubleConv(x) + alpha * DeformConv(x)
    
    优势:
    - 保留标准卷积的特征提取能力
    - 增强对倾斜/弯曲断层的捕捉
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.main = DoubleConv(in_ch, out_ch)
        self.deform = DeformConv3d(in_ch, out_ch, 3, padding=1)
        self.alpha = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, x):
        main_out = self.main(x)
        def_out = self.deform(x)
        return main_out + self.alpha * def_out


class FaultOrientedMamba(nn.Module):
    """
    物理引导的断层感知Mamba (V17)
    
    核心思想:
    - 利用小波特征的物理属性(方差、梯度、Semblance)判断断层方向
    - 断层区域: 多方向扫描增强
    - 非断层区域: 保持简单
    
    扫描方向:
    - D方向: 垂直走向
    - H方向: crossline方向
    - W方向: inline方向
    """
    def __init__(self, in_channels, d_state=16, d_conv=3, expand=1, max_seq_len=128):
        super().__init__()
        self.in_channels = in_channels

        if not MAMBA_AVAILABLE:
            raise ImportError("mamba-ssm required. Install: pip install mamba-ssm")

        # 三个方向的Mamba
        self.mamba_d = Mamba(d_model=in_channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_h = Mamba(d_model=in_channels, d_state=d_state, d_conv=d_conv, expand=expand)
        self.mamba_w = Mamba(d_model=in_channels, d_state=d_state, d_conv=d_conv, expand=expand)

        # 输入投影
        self.in_proj = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1, bias=False),
            nn.GroupNorm(in_channels // 4, in_channels),
            nn.ReLU(inplace=True)
        )

        # 输出投影
        self.out_proj = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1, bias=False),
            nn.GroupNorm(in_channels // 4, in_channels),
        )

        # 物理属性提取 + 方向权重预测
        self.physics_extractor = nn.Sequential(
            nn.Conv3d(in_channels, 32, 1),
            nn.ReLU(),
            nn.Conv3d(32, 3, 1)
        )

        # 残差权重
        self.alpha = nn.Parameter(torch.tensor(0.5))

        # 位置编码
        self.register_buffer('pe', self._create_3d_pe(max_seq_len, in_channels))

    def _create_3d_pe(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _get_pe(self, seq_len):
        return self.pe[:seq_len]

    def scan_along_dim(self, x, dim):
        """沿指定方向扫描"""
        B, C, D, H, W = x.shape
        
        if dim == 2:  # D方向
            x_seq = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, C)
            pe = self._get_pe(D)
        elif dim == 3:  # H方向
            x_seq = x.permute(0, 2, 4, 3, 1).reshape(B * D * W, H, C)
            pe = self._get_pe(H)
        else:  # W方向 (dim == 4)
            x_seq = x.permute(0, 2, 3, 4, 1).reshape(B * D * H, W, C)
            pe = self._get_pe(W)
        
        # 添加位置编码
        x_seq = x_seq + pe.unsqueeze(0)
        
        # 双向扫描
        mamba = self.mamba_d if dim == 2 else (self.mamba_h if dim == 3 else self.mamba_w)
        out_fwd = mamba(x_seq)
        out_rev = mamba(x_seq.flip(1)).flip(1)
        out = (out_fwd + out_rev) / 2
        
        # 重排形状
        if dim == 2:
            out = out.reshape(B, H, W, D, C).permute(0, 4, 3, 1, 2).contiguous()
        elif dim == 3:
            out = out.reshape(B, D, W, H, C).permute(0, 4, 1, 3, 2).contiguous()
        else:
            out = out.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        
        return out

    def forward(self, x, wavelet_feat=None):
        """
        Args:
            x: [B, C, D, H, W] CNN特征 (已包含wavelet信息)
        """
        identity = x
        x = self.in_proj(x)
        
        # 三个方向扫描
        out_d = self.scan_along_dim(x, dim=2)
        out_h = self.scan_along_dim(x, dim=3)
        out_w = self.scan_along_dim(x, dim=4)
        
        # 直接用 x 预测方向权重（编码器已融合wavelet）
        direction_weights = self.physics_extractor(x)
        direction_weights = F.softmax(direction_weights, dim=1)
        
        w = direction_weights
        mamba_out = w[:, 0:1] * out_d + w[:, 1:2] * out_h + w[:, 2:3] * out_w
        
        out = self.out_proj(mamba_out)
        alpha = torch.sigmoid(self.alpha)
        out = identity + alpha * out

        return out


class WaveletFusion(nn.Module):
    """在编码器阶段融合 wavelet 特征"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(16 + channels, channels, 1)
        
    def forward(self, x_enc, wavelet_feat):
        wavelet_down = F.interpolate(wavelet_feat, size=x_enc.shape[2:], 
                                   mode='trilinear', align_corners=False)
        return self.conv(torch.cat([x_enc, wavelet_down], dim=1))


# ==================== 基础模块 ====================
class DoubleConv(nn.Module):
    """
    双层卷积块：(Conv3d -> BN -> ReLU) × 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class LosslessDown(nn.Module):
    """
    无损下采样：Space-to-Channel像素重排

    原理：将2×2×2空间块重排到8个通道，零信息丢失
    [B, C, D, H, W] → [B, C*8, D//2, H//2, W//2] → [B, C_out, D//2, H//2, W//2]

    对比MaxPool：
    - MaxPool: 丢失7/8的像素
    - LosslessDown: 保留全部像素，只是重新排列
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 重排后通道数 = in_channels * 8
        # 用卷积调整到目标通道数并混合特征
        self.feature_mix = nn.Sequential(
            nn.Conv3d(in_channels * 8, out_channels, kernel_size=1),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3,
                      padding=1, padding_mode='replicate'),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        B, C, D, H, W = x.shape

        # Space-to-Channel重排
        # [B, C, D, H, W] → [B, C, D//2, 2, H//2, 2, W//2, 2]
        x = x.view(B, C, D // 2, 2, H // 2, 2, W // 2, 2)

        # 把空间的2×2×2移到通道维度
        # [B, C, D//2, 2, H//2, 2, W//2, 2] → [B, C, 2, 2, 2, D//2, H//2, W//2]
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()

        # [B, C, 2, 2, 2, D//2, H//2, W//2] → [B, C*8, D//2, H//2, W//2]
        x = x.view(B, C * 8, D // 2, H // 2, W // 2)

        # 通道调整 + 特征混合
        x = self.feature_mix(x)

        return x


class PCDM3D(nn.Module):
    """
    PCDM3D - Pixel-level Context-aware Downsampling Module (3D版本)
    
    双路径下采样:
    - 路径1 (卷积): 3x3x3卷积 + stride=2, 提取特征
    - 路径2 (无损): 手动Space-to-Channel重排, 保留原始信息
    
    融合: 拼接后GroupNorm + ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # 路径1: 卷积下采样 (stride=2) - 学习特征
        self.conv = nn.Conv3d(in_channels, out_channels - in_channels, (3, 3, 3),
                              stride=2, padding=1, bias=False)
        
        # 路径2: 无损下采样 (手动Space-to-Channel) + 通道调整
        self.pool_conv = nn.Conv3d(in_channels * 8, out_channels - in_channels, 1)
        
        # 归一化和激活
        self.gn = nn.GroupNorm(out_channels // 4, out_channels)

    def _space_to_channel(self, x):
        """Space-to-Channel重排: [B, C, D, H, W] -> [B, C*8, D/2, H/2, W/2]"""
        B, C, D, H, W = x.shape
        # [B, C, D, H, W] -> [B, C, D//2, 2, H//2, 2, W//2, 2]
        x = x.view(B, C, D // 2, 2, H // 2, 2, W // 2, 2)
        # -> [B, C, 2, 2, 2, D//2, H//2, W//2]
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        # -> [B, C*8, D//2, H//2, W//2]
        x = x.view(B, C * 8, D // 2, H // 2, W // 2)
        return x

    def forward(self, x):
        # 路径1: 卷积下采样
        conv_out = self.conv(x)
        
        # 路径2: 无损下采样 + 通道调整
        pool_out = self._space_to_channel(x)
        pool_out = self.pool_conv(pool_out)
        
        # 拼接融合
        output = torch.cat([conv_out, pool_out], dim=1)
        output = self.gn(output)
        
        return F.relu(output, inplace=True)


class LosslessUp(nn.Module):
    """
    无损上采样：Channel-to-Space像素重排

    原理：将8个通道重排到2×2×2空间块，零信息丢失
    [B, C, D, H, W] → [B, C_out*8, D, H, W] → [B, C_out, D*2, H*2, W*2]

    对比Trilinear插值：
    - Trilinear: 插值猜测，无法恢复丢失的信息
    - LosslessUp: 可学习的重排，配合skip connection效果更好
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 先调整通道数为 out_channels * 8
        self.channel_expand = nn.Sequential(
            nn.Conv3d(in_channels, out_channels * 8, kernel_size=1),
            nn.GroupNorm(out_channels * 8 // 4, out_channels * 8),
            nn.ReLU()
        )

        # 重排后的特征混合
        self.feature_mix = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3,
                      padding=1, padding_mode='replicate'),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        B, C, D, H, W = x.shape

        # 通道扩展
        x = self.channel_expand(x)  # [B, out_channels*8, D, H, W]

        C_out = self.out_channels

        # Channel-to-Space重排
        # [B, C_out*8, D, H, W] → [B, C_out, 2, 2, 2, D, H, W]
        x = x.view(B, C_out, 2, 2, 2, D, H, W)

        # 把通道的2×2×2移到空间维度
        # [B, C_out, 2, 2, 2, D, H, W] → [B, C_out, D, 2, H, 2, W, 2]
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()

        # [B, C_out, D, 2, H, 2, W, 2] → [B, C_out, D*2, H*2, W*2]
        x = x.view(B, C_out, D * 2, H * 2, W * 2)

        # 特征混合
        x = self.feature_mix(x)

        return x


class MambaSkip(nn.Module):
    """
    轻量级 Mamba 跳跃连接模块
    用 Mamba 处理跳跃连接特征，增强长距离建模能力
    
    优势:
    - Mamba 提供全局依赖建模
    - 多方向扫描(D/H/W)，捕获不同方向的断层
    - 通道压缩减少计算量
    - 添加位置编码，减少边界效应
    """
    def __init__(self, channels, d_state=8, d_conv=3, expand=1, max_seq_len=128):
        super().__init__()
        
        # 通道压缩（减少计算量）
        self.channel_proj = nn.Conv3d(channels, channels // 2, 1, bias=False)
        
        # 三个方向的 Mamba（轻量级，expand=1）
        self.mamba_d = Mamba(
            d_model=channels // 2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.mamba_h = Mamba(
            d_model=channels // 2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.mamba_w = Mamba(
            d_model=channels // 2,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # 归一化和激活
        self.norm = nn.GroupNorm((channels // 2) // 4 if channels // 2 >= 4 else 1, channels // 2)
        self.act = nn.SiLU()
        
        # 输出投影
        self.out_proj = nn.Conv3d(channels // 2, channels, 1, bias=False)
        
        # 位置编码（三个方向各自独立）
        self._init_position_encoding(max_seq_len, channels // 2)
        
    def _init_position_encoding(self, max_len, d_model):
        """初始化 Sinusoidal 位置编码"""
        # D方向位置编码
        pe_d = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_d[:, 0::2] = torch.sin(position * div_term)
        pe_d[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_d', pe_d)
        
        # H方向位置编码
        pe_h = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_h[:, 0::2] = torch.sin(position * div_term)
        pe_h[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_h', pe_h)
        
        # W方向位置编码
        pe_w = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_w[:, 0::2] = torch.sin(position * div_term)
        pe_w[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe_w', pe_w)
        
    def _get_pe(self, pe_buffer, seq_len):
        """获取指定长度的位置编码"""
        return pe_buffer[:seq_len]
        
    def forward(self, x):
        # x: [B, C, D, H, W]
        B, C, D, H, W = x.shape
        
        # 通道压缩
        x = self.channel_proj(x)  # [B, C//2, D, H, W]
        C_proj = C // 2
        
        # ===== 三个方向扫描 =====
        # D方向 (深度) - 双向扫描
        x_d = x.permute(0, 3, 4, 2, 1).reshape(B * H * W, D, C_proj)
        pe_d = self._get_pe(self.pe_d, D)
        x_d = x_d + pe_d.unsqueeze(0)  # 添加位置编码
        out_d_fwd = self.mamba_d(x_d)
        out_d_rev = self.mamba_d(x_d.flip(1)).flip(1)
        out_d = (out_d_fwd + out_d_rev) / 2
        out_d = out_d.reshape(B, H, W, D, C_proj).permute(0, 4, 3, 1, 2)
        
        # H方向 (crossline) - 双向扫描
        x_h = x.permute(0, 2, 4, 3, 1).reshape(B * D * W, H, C_proj)
        pe_h = self._get_pe(self.pe_h, H)
        x_h = x_h + pe_h.unsqueeze(0)  # 添加位置编码
        out_h_fwd = self.mamba_h(x_h)
        out_h_rev = self.mamba_h(x_h.flip(1)).flip(1)
        out_h = (out_h_fwd + out_h_rev) / 2
        out_h = out_h.reshape(B, D, W, H, C_proj).permute(0, 4, 1, 3, 2)
        
        # W方向 (inline) - 双向扫描
        x_w = x.permute(0, 2, 3, 4, 1).reshape(B * D * H, W, C_proj)
        pe_w = self._get_pe(self.pe_w, W)
        x_w = x_w + pe_w.unsqueeze(0)  # 添加位置编码
        out_w_fwd = self.mamba_w(x_w)
        out_w_rev = self.mamba_w(x_w.flip(1)).flip(1)
        out_w = (out_w_fwd + out_w_rev) / 2
        out_w = out_w.reshape(B, D, H, W, C_proj).permute(0, 4, 1, 2, 3)
        
        # 三个方向平均融合
        x = (out_d + out_h + out_w) / 3
        
        x = self.act(self.norm(x))
        return self.out_proj(x)


class ProgressiveDenseFusion(nn.Module):
    """
    渐进式密集跳跃连接 (V16)
    
    每次只融合2个特征，避免显存爆炸:
    1. x3 + x4_up → fuse1 → 64ch
    2. fuse1_out + x2 → fuse2 → 32ch  
    3. fuse2_out + x1 → fuse3 → 16ch
    """
    def __init__(self):
        # Level 3: 64(x3) + 64(x4_up) = 128 → 64
        super().__init__()
        self.fuse1 = nn.Sequential(
            nn.Conv3d(128, 64, 1),
            nn.GroupNorm(16, 64),
            nn.ReLU(inplace=True)
        )
        
        # Level 2: 64(out) + 32(x2) = 96 → 32
        self.fuse2 = nn.Sequential(
            nn.Conv3d(96, 32, 1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True)
        )
        
        # Level 1: 32(out) + 16(x1) = 48 → 16
        self.fuse3 = nn.Sequential(
            nn.Conv3d(48, 16, 1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x3, x2, x1, d3_input):
        # Level 3
        d3 = self.fuse1(torch.cat([x3, d3_input], dim=1))
        
        # 上采样到Level 2
        d3_up = F.interpolate(d3, scale_factor=2, mode='trilinear', align_corners=False)
        d2 = self.fuse2(torch.cat([d3_up, x2], dim=1))
        
        # 上采样到Level 1
        d2_up = F.interpolate(d2, scale_factor=2, mode='trilinear', align_corners=False)
        d1 = self.fuse3(torch.cat([d2_up, x1], dim=1))
        
        return d1


# 删除旧的DenseSkipFusion类


class FaultAwareCBAM(nn.Module):
    """
    断层感知CBAM模块 (V13)
    
    改进点:
    1. 通道注意力: 使用MaxPool+AvgPool+Variance，更关注断层的方差特征
    2. 空间注意力: 保留边界区域，抑制均匀区域
    
    原理:
    - 断层处特征方差大 (不连续)
    - 边界处空间差异大
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.channels = channels
        
        # 通道注意力: Max + Avg + Variance
        self.channel_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        
        # 空间注意力: 边界感知
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3, padding_mode='replicate'),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            增强后的特征
        """
        # ===== 通道注意力 =====
        # MaxPool + AvgPool
        max_pool = F.adaptive_max_pool3d(x, 1).view(x.size(0), -1)
        avg_pool = F.adaptive_avg_pool3d(x, 1).view(x.size(0), -1)
        
        # 方差 (断层处方差大)
        var_pool = x.var(dim=(2, 3, 4), keepdim=True).view(x.size(0), -1)
        
        # 三个统计量分别通过MLP
        max_out = self.channel_mlp(max_pool)
        avg_out = self.channel_mlp(avg_pool)
        var_out = self.channel_mlp(var_pool)
        
        # 融合: 给方差更高的权重(断层敏感)
        channel_attn = torch.sigmoid(max_out + avg_out + 0.5 * var_out)
        channel_attn = channel_attn.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        x = x * channel_attn
        
        # ===== 空间注意力 =====
        # 沿通道维度做MaxPool + AvgPool -> [B, 2, D, H, W]
        max_spatial = x.max(dim=1, keepdim=True)[0]
        avg_spatial = x.mean(dim=1, keepdim=True)
        spatial_input = torch.cat([max_spatial, avg_spatial], dim=1)
        
        spatial_attn = self.spatial_conv(spatial_input)
        
        # 边界增强: 空间注意力高的地方(边界)保留，抑制平滑区域
        x = x * spatial_attn
        
        return x


class SeismicFaultCBAM(nn.Module):
    """
    地震断层感知注意力 (V14最终版)
    
    核心创新:
    1. 方差统计量 - 断处方差大，增强信号
    2. 门控降噪 - 抑制噪声，减少假阳性
    3. 残差稳定 - identity + gate * enhance
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        mid = max(channels // reduction, 8)
        
        # 通道注意力: FC (含方差)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels)
        )
        
        # 空间注意力: 原始CBAM
        self.spatial_conv = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3, padding_mode='replicate'),
            nn.Sigmoid()
        )
        
        # 门控降噪 (核心创新)
        self.gate_conv = nn.Sequential(
            nn.Conv3d(channels, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        identity = x
        B, C, D, H, W = x.shape
        
        # ===== 1. 通道注意力 (含方差) =====
        avg_pool = F.adaptive_avg_pool3d(x, 1).view(B, C)
        var_pool = x.var(dim=(2, 3, 4), keepdim=True).view(B, C)
        
        # 方差加权融合 (创新: 给方差更高权重)
        avg_out = self.channel_fc(avg_pool)
        var_out = self.channel_fc(var_pool)
        channel_attn = torch.sigmoid(avg_out + 0.5 * var_out).view(B, C, 1, 1, 1)
        
        x = x * channel_attn
        
        # ===== 2. 空间注意力 =====
        max_spatial = x.max(dim=1, keepdim=True)[0]
        avg_spatial = x.mean(dim=1, keepdim=True)
        spatial_input = torch.cat([max_spatial, avg_spatial], dim=1)
        spatial_attn = self.spatial_conv(spatial_input)
        
        x = x * spatial_attn
        
        # ===== 3. 门控降噪 (核心创新) =====
        gate = self.gate_conv(identity)
        
        # 残差: identity + gate * enhance
        # gate低(噪声区): identity主导，不增强
        # gate高(断层区): 增强特征
        out = identity + gate * (x - identity)
        
        return out


class DAC(nn.Module):
    """
    方向感知卷积模块 v2 (Direction-Aware Convolution)

    创新点：
    1. 响应驱动的方向选择：用边缘检测响应强度自动选择主方向，而非盲目预测
    2. 多倾角边缘检测核：6个真正的边缘检测核覆盖0°-150°倾角
    3. 自适应残差融合：可学习的融合权重
    4. 跨通道方向一致性：同一位置不同通道倾向选择相同方向

    针对断层检测优化：
    - 断层是面状结构，具有特定走向和倾角
    - 边缘检测核能突出断层边界
    - 响应驱动选择能自动适应不同倾角的断层

    【修复】使用register_buffer存储固定的边缘检测核，避免参数浪费
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.num_directions = 6  # 6个方向：0°, 30°, 60°, 90°, 120°, 150°

        # ====== 多方向边缘检测核（使用register_buffer，不参与训练） ======
        self._init_edge_kernels()

        # ====== 响应驱动的方向选择 ======
        # 轻量级：从响应强度直接计算方向权重
        self.response_norm = nn.GroupNorm(in_channels // 4, in_channels)

        # ====== 方向一致性模块：鼓励空间邻域选择相似方向 ======
        self.direction_refine = nn.Conv3d(
            self.num_directions, self.num_directions,
            kernel_size=3, padding=1, groups=1, bias=False
        )

        # ====== 特征融合 ======
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(in_channels // 4, in_channels),
            nn.ReLU(inplace=True)
        )

        # 可学习的残差权重（初始化为0.5，增强方向感知特征的贡献）
        self.residual_weight = nn.Parameter(torch.tensor([0.5]))

    def _init_edge_kernels(self):
        """
        初始化6个方向的3D边缘检测核

        【修复】使用register_buffer存储固定核：
        - 核形状改为 [in_channels, 1, 3, 3, 3] 用于深度可分离卷积
        - 使用register_buffer确保不参与训练
        - 使用F.conv3d + groups=in_channels 进行高效卷积

        在DH平面（深度-高度平面，即inline剖面）上设计不同倾角的边缘检测核
        倾角：0°(水平), 30°, 60°, 90°(垂直), 120°, 150°

        每个角度使用不同的Sobel-like核，确保6个方向真正不同
        """
        # 6个方向的角度
        angles = [0, 30, 60, 90, 120, 150]

        # 预定义6个不同方向的2D边缘检测核（在DH平面上）
        edge_kernels_2d = {
            0: torch.tensor([    # 水平边缘（检测垂直变化）
                [-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]
            ], dtype=torch.float32),
            30: torch.tensor([   # 30度方向
                [-2, -1,  0],
                [-1,  0,  1],
                [ 0,  1,  2]
            ], dtype=torch.float32) * 0.8 + torch.tensor([
                [-1, -2, -1],
                [ 0,  0,  0],
                [ 1,  2,  1]
            ], dtype=torch.float32) * 0.2,  # 30度 = 0.8*45° + 0.2*0°
            60: torch.tensor([   # 60度方向
                [-2, -1,  0],
                [-1,  0,  1],
                [ 0,  1,  2]
            ], dtype=torch.float32) * 0.8 + torch.tensor([
                [-1,  0,  1],
                [-2,  0,  2],
                [-1,  0,  1]
            ], dtype=torch.float32) * 0.2,  # 60度 = 0.8*45° + 0.2*90°
            90: torch.tensor([   # 垂直边缘（检测水平变化）
                [-1,  0,  1],
                [-2,  0,  2],
                [-1,  0,  1]
            ], dtype=torch.float32),
            120: torch.tensor([  # 120度方向
                [ 0, -1, -2],
                [ 1,  0, -1],
                [ 2,  1,  0]
            ], dtype=torch.float32) * 0.8 + torch.tensor([
                [-1,  0,  1],
                [-2,  0,  2],
                [-1,  0,  1]
            ], dtype=torch.float32) * 0.2,  # 120度 = 0.8*135° + 0.2*90°
            150: torch.tensor([  # 150度方向
                [ 0, -1, -2],
                [ 1,  0, -1],
                [ 2,  1,  0]
            ], dtype=torch.float32) * 0.8 + torch.tensor([
                [ 1,  2,  1],
                [ 0,  0,  0],
                [-1, -2, -1]
            ], dtype=torch.float32) * 0.2,  # 150度 = 0.8*135° + 0.2*180°
        }

        # 为每个方向创建深度可分离卷积核 [in_channels, 1, 3, 3, 3]
        for idx, angle in enumerate(angles):
            # 创建深度可分离核：每个通道独立，共享同一个2D边缘检测模式
            kernel = torch.zeros(self.in_channels, 1, 3, 3, 3)
            kernel_2d = edge_kernels_2d[angle]

            # 在中心切片(W=1)设置边缘检测模式
            for c in range(self.in_channels):
                kernel[c, 0, :, :, 1] = kernel_2d

            # 归一化
            kernel = kernel / (kernel.abs().sum() + 1e-8) * 3

            # 使用register_buffer注册为固定参数（不参与训练）
            self.register_buffer(f'edge_kernel_{idx}', kernel)

    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W] 输入特征
        Returns:
            out: [B, C, D, H, W] 方向增强后的特征
        """
        B, C, D, H, W = x.shape
        identity = x

        # ====== Step 1: 计算各方向的边缘响应（使用固定核） ======
        edge_responses = []
        # 【边界伪影修复】使用replicate padding代替零填充
        x_padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')
        for idx in range(self.num_directions):
            # 获取固定的边缘检测核
            kernel = getattr(self, f'edge_kernel_{idx}')
            # 使用F.conv3d + groups实现深度可分离卷积
            resp = F.conv3d(x_padded, kernel, padding=0, groups=C)  # [B, C, D, H, W]
            resp = self.response_norm(resp)
            edge_responses.append(resp)

        # 堆叠响应 [B, 6, C, D, H, W]
        stacked_responses = torch.stack(edge_responses, dim=1)

        # ====== Step 2: 响应驱动的方向选择 ======
        # 计算每个方向的响应强度（取绝对值的均值，边缘处响应强）
        response_strength = stacked_responses.abs().mean(dim=2)  # [B, 6, D, H, W]

        # 方向一致性精炼：空间邻域倾向选择相似方向
        response_strength = self.direction_refine(response_strength)

        # Softmax得到方向权重（响应强的方向权重大）
        direction_weights = F.softmax(response_strength, dim=1)  # [B, 6, D, H, W]

        # ====== Step 3: 加权融合 ======
        weights = direction_weights.unsqueeze(2)  # [B, 6, 1, D, H, W]
        fused = (stacked_responses * weights).sum(dim=1)  # [B, C, D, H, W]

        # 特征融合
        out = self.fusion(fused)

        # 自适应残差连接（权重可学习）
        alpha = torch.sigmoid(self.residual_weight)  # 限制在0-1之间
        out = identity + alpha * out

        return out

class ChannelAttention(nn.Module):
    """
    断层感知的通道注意力模块（精简版）

    保留核心：方差感知 + 双池化
    针对断层：方差大的通道对边缘敏感，自动增强
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # 共享MLP
        mid_channels = max(in_channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, in_channels, 1, bias=False)
        )

        # 方差感知：断层边缘处方差大
        self.var_fc = nn.Conv3d(in_channels, in_channels, 1, bias=False)

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))

        # 方差统计
        mean = x.mean(dim=(2, 3, 4), keepdim=True)
        var = ((x - mean) ** 2).mean(dim=(2, 3, 4), keepdim=True)
        var_out = self.var_fc(var)

        return x * torch.sigmoid(avg_out + max_out + var_out)


class FEAB(nn.Module):
    """
    特征增强注意力模块

    组件：
    - 通道注意力：增强重要通道
    - DAC方向感知卷积：增强断层方向性特征
    """
    def __init__(self, in_channels):
        super().__init__()
        self.channel_attn = ChannelAttention(in_channels=in_channels)
        self.dac = DAC(in_channels=in_channels)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.dac(x)
        return x



class BADH(nn.Module):
    """
    边界感知解耦头 (Boundary-Aware Decoupled Head) - 改进版

    改进点：
    1. 边界引导使用加法注意力而非乘法，避免破坏logits分布
    2. 区域头和边界头使用独立的特征变换
    3. 添加残差连接保证稳定性
    4. 边界特征反馈到区域预测
    """
    def __init__(self, in_channels=32, n_classes=2):
        super().__init__()
        self.n_classes = n_classes

        # 独立的特征变换分支
        # 【边界伪影修复】使用replicate填充
        self.region_transform = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(in_channels // 4, in_channels),
            nn.ReLU(inplace=True)
        )

        self.boundary_transform = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(in_channels // 4, in_channels),
            nn.ReLU(inplace=True)
        )

        # 区域分割头
        self.region_head = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(in_channels // 8, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, n_classes, kernel_size=1)
        )

        # 边界检测头
        self.boundary_head = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // 2, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(in_channels // 8, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 2, 1, kernel_size=1)
        )

        # 边界特征到区域的注意力（加法方式，不破坏logits分布）
        self.boundary_to_region = nn.Sequential(
            nn.Conv3d(1, in_channels // 4, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // 4, n_classes, kernel_size=1)
        )

        # 可学习的融合权重（固定值，避免多GPU问题）
        self.register_buffer('fusion_weight', torch.tensor([0.1]))

    def forward(self, x):
        # 独立特征变换
        region_feat = self.region_transform(x)
        boundary_feat = self.boundary_transform(x)

        # 区域预测（基础）
        region_logits_base = self.region_head(region_feat)

        # 边界预测
        boundary_logits = self.boundary_head(boundary_feat)

        # 边界引导的区域精炼（加法方式）
        boundary_attention = self.boundary_to_region(torch.sigmoid(boundary_logits))

        # 使用固定权重控制边界贡献（加法，不是乘法！）
        fusion_w = self.fusion_weight.item()
        region_logits = region_logits_base + fusion_w * boundary_attention

        return region_logits, boundary_logits

    @staticmethod
    def compute_boundary_target(labels, dilation=2):
        """
        从分割标签计算边界目标 - 改进版

        改进：使用更大的膨胀核生成更宽的边界带，便于学习

        Args:
            labels: [B, D, H, W] - 分割标签 (0/1)
            dilation: 膨胀大小，控制边界宽度
        Returns:
            boundary: [B, 1, D, H, W] - 边界标签 (0/1)
        """
        labels = labels.unsqueeze(1).float()

        kernel_size = 2 * dilation + 1
        padding = dilation

        # 膨胀
        dilated = F.max_pool3d(labels, kernel_size=kernel_size, stride=1, padding=padding)

        # 腐蚀
        eroded = -F.max_pool3d(-labels, kernel_size=kernel_size, stride=1, padding=padding)

        # 边界 = 膨胀 - 腐蚀（更宽的边界带）
        boundary = dilated - eroded

        return torch.clamp(boundary, 0, 1)


class FDEM(nn.Module):
    """
    细节增强模块 (Fine Detail Enhancement Module)

    专门针对小断层检测优化：
    1. 局部对比度增强：突出小断层的微弱特征
    2. 高频特征提取：小断层通常表现为高频信号（使用固定Laplacian核）
    3. 尺度感知注意力：抑制大目标，关注小目标

    原理：
    - 小断层在下采样过程中容易丢失
    - 通过在高分辨率特征上增强小目标来弥补
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # 局部对比度增强：使用小卷积核捕捉细节
        # 【边界伪影修复】使用replicate填充
        self.local_contrast = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=3, padding=1, padding_mode='replicate', groups=in_channels),
            nn.Conv3d(in_channels, in_channels, kernel_size=1),
            nn.GroupNorm(in_channels // 4, in_channels),
            nn.ReLU()
        )

        # 高频特征提取：使用固定的Laplacian核（不可学习，保证高频提取）
        self._create_laplacian_kernel(in_channels)

        # 高频特征后处理（可学习）
        self.high_freq_post = nn.Sequential(
            nn.GroupNorm(in_channels // 4, in_channels),
            nn.ReLU()
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels * 2, in_channels, kernel_size=1),
            nn.GroupNorm(in_channels // 4, in_channels),
            nn.ReLU()
        )

        # 输出门控：控制细节增强的强度
        self.output_gate = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def _create_laplacian_kernel(self, in_channels):
        """创建固定的3D Laplacian核"""
        kernel = torch.zeros(in_channels, 1, 3, 3, 3)

        # D方向
        kernel[:, 0, 0, 1, 1] = -1.0
        kernel[:, 0, 2, 1, 1] = -1.0

        # H方向
        kernel[:, 0, 1, 0, 1] = -1.0
        kernel[:, 0, 1, 2, 1] = -1.0

        # W方向
        kernel[:, 0, 1, 1, 0] = -1.0
        kernel[:, 0, 1, 1, 2] = -1.0

        # 中心
        kernel[:, 0, 1, 1, 1] = 6.0

        self.register_buffer('laplacian_kernel', kernel)

    def forward(self, x):
        """
        输入: x - 高分辨率特征 [B, C, D, H, W]
        输出: 增强后的特征 [B, C, D, H, W]
        """
        B, C, D, H, W = x.shape

        # 1. 局部对比度增强
        local_feat = self.local_contrast(x)

        # 2. 高频特征（使用固定Laplacian核）
        # 使用replicate padding
        x_padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode='replicate')
        high_freq_feat = F.conv3d(x_padded, self.laplacian_kernel, padding=0, groups=C)
        high_freq_feat = self.high_freq_post(high_freq_feat)

        # 3. 融合增强特征
        enhanced = torch.cat([local_feat, high_freq_feat], dim=1)
        enhanced = self.fusion(enhanced)

        # 4. 门控输出（自适应增强强度）
        gate = self.output_gate(enhanced)

        # 残差连接 + 门控增强
        return x + gate * enhanced


class WaveSSM(nn.Module):
    def __init__(self, n_channels, n_classes, use_wavelet=True):
        """
        MSM-UNet: 多尺度断层分割网络 (断层感知Mamba版本)

        Args:
            n_channels: 输入通道数（原始地震数据通道数，通常为1）
            n_classes: 输出类别数
            use_wavelet: 是否使用小波特征提取（16通道输入）

        架构：
            - 小波特征提取：5尺度×3属性+原始 = 16通道
            - 编码器：DoubleConv (16→32→64→128)
            - 断层感知Mamba：x3跳跃连接 + x4瓶颈层
            - 解码器：FEMB特征融合 + FEAB注意力增强
            - 输出头：ASSN自适应尺度选择 + BADH边界感知解耦头

        改进点（相比V1）：
            - 去掉所有Dropout（当前问题是域差异，不是过拟合）
            - Mamba从后处理移到跳跃连接+瓶颈层（更有效的位置）
            - 使用断层感知门控（边界处重置状态，层位处保持连续）
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_wavelet = use_wavelet

        # ==================== 小波特征提取 ====================
        if use_wavelet:
            # 5尺度×3属性+原始 = 16通道
            self.wavelet = MultiScaleWaveletFeatures(
                scales=[2, 3, 4, 6, 8],
                omega0=6.0,
                learnable=False,
                min_kernel_size=7,
                window_size=5,
                use_channel_attention=True
            )
            encoder_in_channels = 16
        else:
            self.wavelet = None
            encoder_in_channels = n_channels

        # ==================== 编码器（无Dropout）====================
        self.inc = DoubleConv(encoder_in_channels, 16)

        # 【PCDM下采样】
        self.down1 = PCDM3D(16, 32)
        self.down2 = PCDM3D(32, 64)
        self.down3 = PCDM3D(64, 128)
        
        # Encoder卷积 - 与checkpoint匹配
        self.enc1 = DoubleConv(32, 32)
        self.enc2 = DoubleConv(64, 64)
        self.enc3 = ResidualDeformConv(128, 128)

        # Wavelet融合模块
        self.wavelet_fuse1 = WaveletFusion(channels=32)
        self.wavelet_fuse2 = WaveletFusion(channels=64)

        # ==================== 物理引导三方向Mamba (V17) ====================
        self.mamba_bottleneck = FaultOrientedMamba(
            in_channels=128,
            d_state=16,
            d_conv=3,
            expand=1
        )

        # ==================== 可变形卷积残差 (V15) ====================
        self.deform_bottleneck = DeformConv3d(128, 128, 3, padding=1)

        # 【无损上采样】
        self.upblock3 = LosslessUp(in_channels=128, out_channels=64)
        self.upblock2 = LosslessUp(in_channels=64, out_channels=32)
        self.upblock1 = LosslessUp(in_channels=32, out_channels=16)

        # 【Mamba 跳跃连接】- 替换渐进式密集连接
        self.mamba_skip3 = MambaSkip(channels=64, d_state=8)
        self.mamba_skip2 = MambaSkip(channels=32, d_state=8)
        self.mamba_skip1 = MambaSkip(channels=16, d_state=8)

        # 断层感知CBAM (加在每个解码层级) - V14原创版本
        self.cbam3 = SeismicFaultCBAM(channels=64, reduction=8)
        self.cbam2 = SeismicFaultCBAM(channels=32, reduction=8)
        self.cbam1 = SeismicFaultCBAM(channels=16, reduction=8)

        # 【FDEM细节增强模块】- 解码器每层添加
        self.fdem64 = FDEM(in_channels=64)
        self.fdem32 = FDEM(in_channels=32)
        self.fdem16 = FDEM(in_channels=16)

        # 输出头：保留BADH边界感知解耦头
        self.badh = BADH(in_channels=16, n_classes=1)

    def forward(self, x):
        """
        V14 SeismicFaultCBAM版前向传播

        架构：
        - 小波特征提取：5尺度×3属性+原始 = 16通道
        - 编码器：3层DoubleConv + LosslessDown
        - 瓶颈层：简化版D方向Mamba
        - 解码器：简单跳跃连接 + SeismicFaultCBAM
        - 输出头：BADH边界感知解耦头
        """
        # ==================== 小波特征提取 ====================
        if self.use_wavelet:
            wavelet_feat = self.wavelet(x)  # [B, 1, D, H, W] -> [B, 16, D, H, W]
            x = wavelet_feat
        else:
            wavelet_feat = None

        # ==================== Encoder ====================
        x1 = self.inc(x)  # [B, 16, D, H, W]

        x2 = self.down1(x1)  # [B, 32, D/2, H/2, W/2]
        x2 = self.enc1(x2)
        x2 = self.wavelet_fuse1(x2, wavelet_feat)

        x3 = self.down2(x2)  # [B, 64, D/4, H/4, W/4]
        x3 = self.enc2(x3)
        x3 = self.wavelet_fuse2(x3, wavelet_feat)

        x4 = self.down3(x3)  # [B, 128, D/8, H/8, W/8]
        x4 = self.enc3(x4)

        # ==================== 瓶颈层Mamba + 可变形卷积 ====================
        x4 = self.mamba_bottleneck(x4)
        x4 = x4 + 0.3 * self.deform_bottleneck(x4)

        # ==================== 解码器 + FDEM + CBAM ====================
        
        # ----- 第1层: 64³ -----
        d3 = self.upblock3(x4)       # 128→64, 16³→64³
        d3 = self.fdem64(d3)         # FDEM细节增强
        d3 = self.cbam3(d3)         # CBAM增强
        d3 = d3 + self.mamba_skip3(x3)  # Mamba跳跃连接

        # ----- 第2层: 32³ -----
        d2 = self.upblock2(d3)       # 64→32, 64³→32³
        d2 = self.fdem32(d2)         # FDEM细节增强
        d2 = self.cbam2(d2)         # CBAM增强
        d2 = d2 + self.mamba_skip2(x2)  # Mamba跳跃连接

        # ----- 第3层: 16³ -----
        d1 = self.upblock1(d2)        # 32→16, 32³→16³
        d1 = self.fdem16(d1)         # FDEM细节增强
        d1 = d1 + self.mamba_skip1(x1)  # Mamba跳跃连接
        d1 = self.cbam1(d1)          # 最终CBAM增强

        # ==================== 输出头 ====================
        logits, boundary_logits = self.badh(d1)
        main_out = torch.sigmoid(logits)

        return main_out, boundary_logits


if __name__ == '__main__':
    # 查看网络参数量
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = MSM_UNet(1, 1).to(device)  # 单通道输出
    summary(net, input_size=(1, 1, 128, 128, 128))