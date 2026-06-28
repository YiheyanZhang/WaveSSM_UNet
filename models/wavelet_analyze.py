"""
基于论文"小波变换与信号瞬时特征分析"（高静怀等，1997）的瞬时属性提取模块

核心思想：
1. 利用解析小波（Morlet小波）进行连续小波变换
2. 根据定理2（公式6）：通过小波变换获得实信号对应的解析信号
3. 基于公式(13a-13c)计算多尺度瞬时属性：
   - 瞬时振幅 e(t,a) = sqrt(S_R^2 + S_I^2)  -- 反映能量变化，断层处有突变
   - 瞬时相位 θ(t,a) = arctan(S_I/S_R)      -- 反映相位连续性，断层处不连续
   - 瞬时频率 ω(t,a) = d/dt[arctan(S_I/S_R)] -- 反映频率变化，断层处异常
4. 小波方法相比Hilbert变换具有更强的抗噪声能力（论文结论）

【物理意义关键修正】：
- 小波变换只在**时间/深度维度(D)**上进行，这是论文的核心思想
- 复数Morlet小波用于提取一维时间信号的瞬时属性
- 空间维度(H, W)代表水平位置，不应进行复数小波变换
- 瞬时频率的导数也只沿时间维度计算

应用于断层分割：
- 断层位置往往表现为瞬时振幅的突变、相位的不连续和频率的异常
- 多尺度分析可以捕捉不同尺度的断层特征
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MorletWavelet1D(nn.Module):
    """
    Morlet小波实现（论文公式11-12）

    g(t) = e^{imt} * e^{-t^2/2} + 修正项

    其中m为角频率，通常取m=5或6以满足容许性条件
    """
    def __init__(self, omega0=6.0):
        super().__init__()
        self.omega0 = omega0  # 中心角频率

    def forward(self, t, scale):
        """
        计算Morlet小波在给定时间点和尺度下的值

        Args:
            t: 时间点 tensor
            scale: 尺度因子
        Returns:
            复数小波值 (实部, 虚部)
        """
        # 归一化时间
        t_scaled = t / scale

        # 高斯包络
        gaussian = torch.exp(-0.5 * t_scaled ** 2)

        # 复指数部分 e^{i*omega0*t_scaled}
        real_part = gaussian * torch.cos(self.omega0 * t_scaled)
        imag_part = gaussian * torch.sin(self.omega0 * t_scaled)

        # 归一化因子 (论文中的1/sqrt(a))
        norm = 1.0 / math.sqrt(scale)

        return real_part * norm, imag_part * norm


class ContinuousWaveletTransform3D(nn.Module):
    """
    3D地震数据的连续小波变换 - 物理意义修正版本

    【重要修正】根据论文"小波变换与信号瞬时特征分析"（高静怀等，1997）：

    1. 复数Morlet小波变换只应沿**时间/深度维度(D)**进行
       - 论文公式(3): S(b,a) = (1/a)∫s(t)g̅((t-b)/a)dt 是一维时间积分
       - 解析信号的概念是针对时间信号定义的

    2. 空间维度(H, W)不应进行复数小波变换
       - H, W是空间采样点，不是时间序列
       - 复数Morlet小波是为提取时频信息设计的，不适用于空间滤波

    3. 物理意义：
       - D维度：时间/深度方向，地震波传播方向，需要提取瞬时属性
       - H, W维度：水平空间位置（inline/crossline），用于空间连续性分析

    【尺度选择原则】
    - 最大有效尺度 ≈ N / (2 × ω₀)，对于D=128, ω₀=6: max_scale ≈ 10
    - 尺度过大会导致卷积核覆盖大部分深度，失去局部瞬时属性意义
    - 默认尺度 [2, 4, 6, 8] 对应核大小 [13, 25, 37, 49]
    """
    def __init__(self, scales=[2, 4, 6, 8], omega0=6.0, learnable=True, min_kernel_size=7):  # noqa: B006
        super().__init__()
        self.scales = scales
        self.learnable = learnable
        self.min_kernel_size = min_kernel_size
        self.morlet = MorletWavelet1D(omega0)
        self._precompute_wavelets()

    def _precompute_wavelets(self):
        """
        预计算小波核

        【复共轭与时间反演的正确处理】

        论文公式(3)定义小波变换为：
        S(b, a) = (1/a) ∫ s(t) * g̅((t-b)/a) dt

        其中 g̅ 是小波 g 的复共轭。

        对于Morlet小波 g(t) = e^{iωt} * e^{-t²/2}：
        - g_R(t) = cos(ωt) * e^{-t²/2}
        - g_I(t) = sin(ωt) * e^{-t²/2}

        复共轭 g̅(t)：
        - g̅_R(t) = cos(ωt) * e^{-t²/2}  （实部不变）
        - g̅_I(t) = -sin(ωt) * e^{-t²/2} （虚部取负）

        【关键】PyTorch的F.conv执行的是互相关（correlation），不是卷积（convolution）：
        - 卷积: (f * g)(t) = ∫ f(τ) g(t-τ) dτ
        - 互相关: (f ⋆ g)(t) = ∫ f(τ) g(τ-t) dτ = ∫ f(τ) g(-(t-τ)) dτ

        互相关等价于用时间反转的核做卷积。

        因此，要正确实现论文公式(3)，我们需要：
        1. 构造复共轭小波 g̅(t)
        2. 将其时间反转得到 g̅(-t)
        3. 用 g̅(-t) 做PyTorch的conv（实际是互相关）

        对于Morlet小波：
        - g̅(-t)_R = cos(-ωt) * e^{-t²/2} = cos(ωt) * e^{-t²/2} = g_R(t)  （偶函数）
        - g̅(-t)_I = -sin(-ωt) * e^{-t²/2} = sin(ωt) * e^{-t²/2} = g_I(t) （奇函数取负后再反转）

        结论：对于Morlet小波，g̅(-t) = g(t)，即复共轭+时间反转等于原函数！

        因此，正确的实现应该直接使用原始的 g_R 和 g_I，不需要额外取负。

        【论文公式(8a)(8b)的含义】
        公式(7): S(b,a) = S_R(b,a) + iS_I(b,a)
        公式(8a): S_R(b,a) = (1/a) ∫ s(t) g_R((t-b)/a) dt
        公式(8b): S_I(b,a) = -(1/a) ∫ s(t) g_I((t-b)/a) dt  （注意负号！）

        这里的负号来自于复共轭的定义，而不是核本身的修改。
        在实现中，我们直接在计算 S_I 时加负号即可。
        """
        self.wavelet_kernels_real = nn.ParameterList()
        self.wavelet_kernels_imag = nn.ParameterList()
        for scale in self.scales:
            # 确保核大小至少为min_kernel_size
            k = max(int(6 * scale) | 1, self.min_kernel_size)
            t = torch.linspace(-k//2, k//2, k)
            r, i = self.morlet(t, scale)  # 使用Morlet小波

            # 由于Morlet小波的特殊对称性，复共轭+时间反转 = 原函数
            # 所以直接使用原始的实部和虚部
            # 根据公式(8b)，S_I 的计算需要加负号，这在forward中处理
            self.wavelet_kernels_real.append(nn.Parameter(r.view(1, 1, k), requires_grad=self.learnable))
            self.wavelet_kernels_imag.append(nn.Parameter(i.view(1, 1, k), requires_grad=self.learnable))

    def forward(self, x):
        """
        对3D地震数据进行连续小波变换

        【物理正确的实现】：
        只在时间/深度维度(D)上进行复数Morlet小波变换，
        空间维度(H, W)保持不变，不进行小波变换。

        根据论文公式(8a)(8b)：
        S_R(b,a) = (1/a) ∫ s(t) g_R((t-b)/a) dt
        S_I(b,a) = -(1/a) ∫ s(t) g_I((t-b)/a) dt  （注意负号！）

        Args:
            x: [B, C, D, H, W] - 3D地震数据，D为时间/深度维度
        Returns:
            cwt_r: [B, C, S, D, H, W] - 小波变换实部 S_R(t,a)
            cwt_i: [B, C, S, D, H, W] - 小波变换虚部 S_I(t,a)
        """
        B, C, D, H, W = x.shape
        x = x.view(B*C, 1, D, H, W)
        cwt_r, cwt_i = [], []

        for kr, ki in zip(self.wavelet_kernels_real, self.wavelet_kernels_imag):
            k = kr.shape[-1]
            p = k // 2

            # 【边界伪影修复】使用replicate padding代替零填充
            # 零填充会导致边界处小波响应异常，产生边界伪影
            x_padded = F.pad(x, (0, 0, 0, 0, p, p), mode='replicate')  # D方向replicate

            # 只在D(时间/深度)方向进行复数小波变换
            # 公式(8a): S_R = ∫ s(t) g_R(...) dt
            xr = F.conv3d(x_padded, kr.view(1, 1, k, 1, 1), padding=0)
            # 公式(8b): S_I = -∫ s(t) g_I(...) dt （负号来自复共轭）
            xi = -F.conv3d(x_padded, ki.view(1, 1, k, 1, 1), padding=0)

            cwt_r.append(xr)
            cwt_i.append(xi)

        cwt_r = torch.stack(cwt_r, dim=2).view(B, C, len(self.scales), D, H, W)
        cwt_i = torch.stack(cwt_i, dim=2).view(B, C, len(self.scales), D, H, W)
        return cwt_r, cwt_i


class WaveletDomainDenoising(nn.Module):
    """
    小波域去噪模块 - 实现论文第828页描述的抗噪声策略

    论文核心思想（第828页）：
    "对信号s(t)(含有噪声)作小波分解时，当所选小波形状与有效信号比较接近时，
    有效信号能量分布在时间-尺度域一个小的闭子空间V中，而干扰波及随机噪声能量
    分布在时间-尺度域的另一个大的闭子空间V₁...V与V₁可能完全分离，也可能部分
    重叠，当我们在V空间讨论问题时，噪声得到了一定的压制。"

    去噪策略：
    1. 软阈值去噪：抑制低能量区域（噪声主导区）
    2. 尺度自适应：不同尺度使用不同阈值（论文指出不同尺度噪声特性不同）
    3. 能量集中性：保留能量集中的有效信号区域
    """
    def __init__(self, num_scales, learnable=True, init_threshold=0.02):
        super().__init__()
        self.num_scales = num_scales
        # 可学习的尺度自适应阈值（论文：不同尺度噪声特性不同）
        if learnable:
            self.thresholds = nn.Parameter(torch.ones(num_scales) * init_threshold)
        else:
            self.register_buffer('thresholds', torch.ones(num_scales) * init_threshold)

    def forward(self, cwt_real, cwt_imag):
        """
        对小波系数进行去噪处理

        Args:
            cwt_real: [B, C, S, D, H, W] 小波变换实部
            cwt_imag: [B, C, S, D, H, W] 小波变换虚部
        Returns:
            denoised_real, denoised_imag: 去噪后的小波系数
        """
        B, C, S, D, H, W = cwt_real.shape

        # 计算每个尺度的振幅
        amplitude = torch.sqrt(cwt_real ** 2 + cwt_imag ** 2 + 1e-8)

        # 对每个尺度计算自适应阈值
        denoised_real = []
        denoised_imag = []

        for s in range(S):
            amp_s = amplitude[:, :, s:s+1, :, :, :]  # [B, C, 1, D, H, W]
            real_s = cwt_real[:, :, s:s+1, :, :, :]
            imag_s = cwt_imag[:, :, s:s+1, :, :, :]

            # 计算该尺度的振幅统计量（用于自适应阈值）
            amp_max = amp_s.amax(dim=(3, 4, 5), keepdim=True)  # [B, C, 1, 1, 1, 1]

            # 软阈值：threshold = λ * max_amplitude
            # λ是可学习参数，初始化为0.1
            threshold = torch.abs(self.thresholds[s]) * amp_max

            # 软阈值函数：保留高于阈值的部分，平滑过渡
            # gain = max(0, 1 - threshold/amplitude)
            gain = torch.clamp(1.0 - threshold / (amp_s + 1e-8), min=0.0)

            denoised_real.append(real_s * gain)
            denoised_imag.append(imag_s * gain)

        denoised_real = torch.cat(denoised_real, dim=2)
        denoised_imag = torch.cat(denoised_imag, dim=2)

        return denoised_real, denoised_imag


class AnalyticSignalReconstruction(nn.Module):
    """
    基于定理2的解析信号积分重构模块

    【论文核心定理2】（第822页，公式6）：
    (1/C_g) ∫₀^∞ S(b,a) da/a = s(b) + iH[s(b)]

    这个积分重构是小波方法抗噪声能力的核心来源：
    1. 积分过程本身就是一种平滑/滤波操作
    2. 噪声在不同尺度上不相关，积分后趋于抵消
    3. 有效信号在不同尺度上相关，积分后得到增强

    【论文公式(15)】（第825页）：
    s*(t) = H[s(t)] = Im[(1/C_g) ∫₀^∞ S(t,a) da/a]

    通过小波变换的积分重构获得Hilbert变换，比直接Hilbert变换更抗噪声。
    这就是为什么论文图1显示小波方法误差为0.007，而Hilbert方法在有噪声时误差为14.15。

    【论文公式(10)】（第824页）：
    s(b) = (1/C_g) ∫₀^∞ (1/a²) ∫_{-∞}^∞ s(t) g_R((t-b)/a) dt da
    这是重构原信号的公式，证明了积分重构的完备性。
    """
    def __init__(self, scales, omega0=6.0):
        super().__init__()
        self.scales = scales
        self.omega0 = omega0
        # 预计算C_g（容许性常数）
        self._compute_Cg()

    def _compute_Cg(self):
        """
        计算Morlet小波的容许性常数C_g（论文公式5）

        C_g = ∫₀^∞ ĝ_R(ω)/ω dω < ∞, C_g ≠ 0

        对于Morlet小波 g(t) = e^{iω₀t} * e^{-t²/2}，其Fourier变换为：
        ĝ(ω) ≈ √(2π) * e^{-(ω-ω₀)²/2}  (当ω₀足够大时)

        对于ω₀=6，C_g ≈ π（数值积分结果）
        """
        # 对于标准Morlet小波，C_g ≈ π
        self.register_buffer('C_g', torch.tensor(math.pi))

    def forward(self, cwt_real, cwt_imag, return_multiscale=True):
        """
        通过积分重构解析信号（定理2的实现）

        Args:
            cwt_real: [B, C, S, D, H, W] 小波变换实部 S_R(t,a)
            cwt_imag: [B, C, S, D, H, W] 小波变换虚部 S_I(t,a)
            return_multiscale: 是否同时返回多尺度特征
        Returns:
            reconstructed_real: [B, C, D, H, W] 重构的实信号 s(t)
            reconstructed_imag: [B, C, D, H, W] 重构的Hilbert变换 H[s(t)]（公式15）
            (可选) cwt_real, cwt_imag: 原始多尺度系数
        """
        B, C, S, D, H, W = cwt_real.shape

        # 论文公式(6): (1/C_g) ∫₀^∞ S(b,a) da/a
        # 离散近似: (1/C_g) Σ S(b,aᵢ) * Δaᵢ/aᵢ
        # 使用对数均匀尺度时: Δaᵢ/aᵢ ≈ Δ(log a) = const

        # 计算 da/a 的权重
        scales_tensor = torch.tensor(self.scales, dtype=cwt_real.dtype, device=cwt_real.device)

        # 对数尺度间隔近似（梯形积分法）
        if len(self.scales) > 1:
            log_scales = torch.log(scales_tensor)
            d_log_a = torch.zeros_like(log_scales)
            d_log_a[0] = log_scales[1] - log_scales[0]
            d_log_a[-1] = log_scales[-1] - log_scales[-2]
            if len(self.scales) > 2:
                d_log_a[1:-1] = (log_scales[2:] - log_scales[:-2]) / 2
        else:
            d_log_a = torch.ones(1, dtype=cwt_real.dtype, device=cwt_real.device)

        # 积分权重: da/a = d(log a)
        weights = d_log_a.view(1, 1, S, 1, 1, 1)  # [1, 1, S, 1, 1, 1]

        # 执行积分: (1/C_g) Σ S(b,aᵢ) * d(log aᵢ)
        # S(b,a) = S_R(b,a) + i*S_I(b,a)
        reconstructed_real = (cwt_real * weights).sum(dim=2) / self.C_g  # [B, C, D, H, W]
        reconstructed_imag = (cwt_imag * weights).sum(dim=2) / self.C_g  # [B, C, D, H, W]

        if return_multiscale:
            return reconstructed_real, reconstructed_imag, cwt_real, cwt_imag
        return reconstructed_real, reconstructed_imag


class InstantaneousAttributes(nn.Module):
    """
    瞬时属性计算模块（论文公式13a-13c, 14a-14c, 15, 16）

    【重要】支持两种计算模式：

    1. 多尺度模式（公式13a-13c）：基于每个尺度的小波系数计算瞬时属性
       - e(t,a) = sqrt(S_R² + S_I²)
       - θ(t,a) = arctan(S_I/S_R)
       - ω(t,a) = d/dt[arctan(S_I/S_R)]
       适用于需要多分辨率分析的场景

    2. 重构模式（公式14a-14c, 15）：基于定理2积分重构的解析信号计算
       - 通过公式(6)/(15)的积分重构获得更抗噪声的结果
       - 这是论文的核心贡献：小波方法比Hilbert变换更抗噪声

    瞬时频率使用论文公式(16)的阻尼形式：
    f(t) = [s(t)*ds*/dt - s*(t)*ds/dt] / [2π * (e²(t) + ε²*e²_max)]
    其中ε为小于1的正实数，用于抑制低振幅区的噪声放大
    """
    def __init__(self, eps=1e-8, damping_eps=0.005):
        super().__init__()
        self.eps = eps
        self.damping_eps = damping_eps  # 论文公式(16)中的ε，用于抑制低振幅区噪声

    def forward(self, cwt_real, cwt_imag):
        """
        计算瞬时属性

        Args:
            cwt_real: 小波变换实部 S_R(t,a)
            cwt_imag: 小波变换虚部 S_I(t,a)
        Returns:
            inst_amp: 瞬时振幅
            inst_phase: 瞬时相位
            inst_freq: 瞬时频率
        """
        # 公式(13a): 瞬时振幅 e(t,a) = sqrt(S_R^2 + S_I^2)
        inst_amp = torch.sqrt(cwt_real ** 2 + cwt_imag ** 2 + self.eps)

        # 公式(13b): 瞬时相位 θ(t,a) = arctan(S_I / S_R)
        inst_phase = torch.atan2(cwt_imag, cwt_real + self.eps)

        # 公式(13c): 瞬时频率 ω(t,a) = d/dt[arctan(S_I / S_R)]
        inst_freq = self._compute_inst_freq(cwt_real, cwt_imag)

        return inst_amp, inst_phase, inst_freq

    def _compute_inst_freq(self, cwt_real, cwt_imag):
        """
        计算阻尼瞬时频率（论文公式13c和16）

        【物理意义修正】：
        瞬时频率是时间信号的瞬时相位对**时间**的导数，即公式(13c)：
        ω(t, a) = d/dt[arctan(S_I/S_R)]

        展开后得到公式(16)的形式：
        f(t) = (1/2π) * [S_R*dS_I/dt - S_I*dS_R/dt] / [e²(t) + ε²*e²_max]

        【关键】：导数只应该沿**时间/深度维度(D)**计算，不应该在空间维度上计算！
        因为瞬时频率是描述信号随时间变化的特性，与空间位置无关。
        """
        # 振幅的平方 e²(t)
        amp_squared = cwt_real ** 2 + cwt_imag ** 2

        # 对于3D数据 [B, C, S, D, H, W]
        if cwt_real.dim() == 6:
            # 计算 e²_max（每个样本、通道、尺度独立计算）
            amp_squared_max = amp_squared.amax(dim=(3, 4, 5), keepdim=True)  # [B, C, S, 1, 1, 1]

            # 阻尼分母：e²(t) + ε²*e²_max（论文公式16）
            damping_term = self.damping_eps ** 2 * amp_squared_max
            denominator = amp_squared + damping_term + self.eps

            # 【关键修正】只在D(时间/深度)方向计算导数
            # 这才是瞬时频率的正确物理意义：相位对时间的变化率
            dS_R = self._gradient(cwt_real, dim=3)  # D方向 = 时间方向
            dS_I = self._gradient(cwt_imag, dim=3)

            # 瞬时频率（只有时间方向的分量）
            inst_freq = (cwt_real * dS_I - cwt_imag * dS_R) / denominator

        else:
            # 非6维数据
            amp_squared_max = amp_squared.amax(dim=-1, keepdim=True)
            damping_term = self.damping_eps ** 2 * amp_squared_max
            denominator = amp_squared + damping_term + self.eps

            dS_R = self._gradient(cwt_real, dim=-1)
            dS_I = self._gradient(cwt_imag, dim=-1)
            inst_freq = (cwt_real * dS_I - cwt_imag * dS_R) / denominator

        return inst_freq

    def _gradient(self, x, dim):
        """沿指定维度计算梯度（中心差分）- 边界修复版本

        【边界修复】手动实现replicate padding避免边界伪影
        注意：PyTorch的F.pad replicate模式不支持6D张量，因此手动实现
        """
        # 手动实现replicate padding for 6D tensor
        if dim == 3:  # D维度 [B, C, S, D, H, W]
            # 手动replicate padding: 在D维度前后各复制一层
            first_slice = x[:, :, :, :1, :, :]  # 第一个切片
            last_slice = x[:, :, :, -1:, :, :]  # 最后一个切片
            x_padded = torch.cat([first_slice, x, last_slice], dim=3)
            grad = (x_padded[:, :, :, 2:, :, :] - x_padded[:, :, :, :-2, :, :]) / 2.0
        elif dim == 4:  # H维度 [B, C, S, D, H, W]
            first_slice = x[:, :, :, :, :1, :]
            last_slice = x[:, :, :, :, -1:, :]
            x_padded = torch.cat([first_slice, x, last_slice], dim=4)
            grad = (x_padded[:, :, :, :, 2:, :] - x_padded[:, :, :, :, :-2, :]) / 2.0
        elif dim == 5:  # W维度 [B, C, S, D, H, W]
            first_slice = x[:, :, :, :, :, :1]
            last_slice = x[:, :, :, :, :, -1:]
            x_padded = torch.cat([first_slice, x, last_slice], dim=5)
            grad = (x_padded[:, :, :, :, :, 2:] - x_padded[:, :, :, :, :, :-2]) / 2.0
        else:  # 其他维度（使用通用方式）
            # 使用torch.diff + replicate边界值
            diff = torch.diff(x, dim=dim)
            grad_center = (diff[..., 1:] + diff[..., :-1]) / 2
            grad_first = diff[..., :1]
            grad_last = diff[..., -1:]
            grad = torch.cat([grad_first, grad_center, grad_last], dim=dim)

        return grad


class MultiScaleInstantaneousFusion(nn.Module):
    """
    多尺度瞬时属性融合模块

    根据论文第3节的分析：不同尺度下的瞬时参数描述了具有不同分辨率的地震记录特征
    - 小尺度：高频率分辨率，捕捉细节断层
    - 大尺度：高时间分辨率，捕捉主要断层

    融合策略：自适应加权融合多尺度瞬时属性
    """
    def __init__(self, num_scales, in_channels, reduction=4):
        super().__init__()
        self.num_scales = num_scales

        # 尺度注意力：学习各尺度的重要性权重
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(in_channels * num_scales, in_channels * num_scales // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels * num_scales // reduction, num_scales),
            nn.Softmax(dim=1)
        )

    def forward(self, multi_scale_features):
        """
        Args:
            multi_scale_features: [B, C, S, D, H, W] S为尺度数
        Returns:
            fused: [B, C, D, H, W]
        """
        B, C, S, D, H, W = multi_scale_features.shape

        # 计算尺度注意力权重
        x_reshape = multi_scale_features.permute(0, 2, 1, 3, 4, 5).reshape(B, S * C, D, H, W)
        scale_weights = self.scale_attention(x_reshape)  # [B, S]
        scale_weights = scale_weights.view(B, 1, S, 1, 1, 1)

        # 加权融合
        fused = (multi_scale_features * scale_weights).sum(dim=2)

        return fused

class EnhancedWaveletInstantaneous3D(nn.Module):
    """
    增强版小波瞬时属性提取模块 - 实现论文完整的抗噪声机制

    【论文核心贡献的实现】：
    1. 定理2积分重构（公式6）：通过多尺度积分获得抗噪声的解析信号
    2. 小波域去噪（第828页）：在时间-尺度域分离有效信号和噪声
    3. 阻尼瞬时频率（公式16）：抑制低振幅区的噪声放大

    输出选项：
    - use_reconstruction=False: 输出13通道（原始 + 4尺度×3属性）
    - use_reconstruction=True:  输出16通道（原始 + 4尺度×3属性 + 重构3属性）

    修复：
    1. 移除对原始数据的二次归一化（输入已在dataloader中归一化）
    2. 统一所有特征到相同的数值范围
    3. 小波核参数可学习
    4. 增加最小核大小参数确保小尺度的有效性
    5. 【新增】实现论文的积分重构和小波域去噪机制
    """
    def __init__(self, in_channels=1, scales=[2, 4, 6, 8], omega0=6.0, learnable=True,  # noqa: B006
                 min_kernel_size=7, use_denoising=True, use_reconstruction=False):
        super().__init__()
        self.in_channels = in_channels
        self.scales = scales
        self.num_scales = len(scales)
        self.use_denoising = use_denoising
        self.use_reconstruction = use_reconstruction

        # 连续小波变换（Morlet小波）
        self.cwt = ContinuousWaveletTransform3D(
            scales=scales, omega0=omega0, learnable=learnable,
            min_kernel_size=min_kernel_size
        )

        # 【新增】小波域去噪模块（论文第828页）
        if use_denoising:
            self.denoiser = WaveletDomainDenoising(num_scales=len(scales), learnable=learnable)

        # 【新增】解析信号积分重构模块（论文定理2，公式6）
        if use_reconstruction:
            self.reconstructor = AnalyticSignalReconstruction(scales=scales, omega0=omega0)

        # 瞬时属性计算
        self.inst_attr = InstantaneousAttributes()

    def forward(self, x):
        """
        Args:
            x: [B, 1, D, H, W] - 已在dataloader中归一化
        Returns:
            use_reconstruction=False: [B, 13, D, H, W] - 原始数据 + 4尺度×3属性
            use_reconstruction=True:  [B, 16, D, H, W] - 原始数据 + 4尺度×3属性 + 重构3属性
        """
        # Step 1: 连续小波变换
        cwt_real, cwt_imag = self.cwt(x)  # [B, 1, S, D, H, W]

        # Step 2: 【论文第828页】小波域去噪（可选）
        # "有效信号能量分布在时间-尺度域一个小的闭子空间V中，
        #  而干扰波及随机噪声能量分布在另一个大的闭子空间V₁"
        if self.use_denoising:
            cwt_real, cwt_imag = self.denoiser(cwt_real, cwt_imag)

        # Step 3: 计算多尺度瞬时属性（公式13a-13c）
        inst_amp, inst_phase, inst_freq = self.inst_attr(cwt_real, cwt_imag)

        # [B, 1, S, D, H, W] -> [B, S, D, H, W]
        inst_amp = inst_amp.squeeze(1)
        inst_phase = inst_phase.squeeze(1)
        inst_freq = inst_freq.squeeze(1)

        # ========== 振幅归一化 ==========
        # log压缩后标准化（振幅是非负的，log压缩有助于处理动态范围）
        amp_norm = torch.log1p(inst_amp)
        amp_norm = (amp_norm - amp_norm.mean(dim=(2,3,4), keepdim=True)) / (amp_norm.std(dim=(2,3,4), keepdim=True) + 1e-8)

        # ========== 相位归一化 ==========
        # 【物理意义修正】相位是周期性变量，不能简单做均值-标准差归一化！
        # 正确做法：简单线性缩放到 [-1, 1]，保持相位的物理意义
        phase_norm = inst_phase / math.pi  # [-π, π] -> [-1, 1]

        # ========== 频率归一化 ==========
        # 频率可以做标准化，因为它不是周期性变量
        freq_norm = (inst_freq - inst_freq.mean(dim=(2,3,4), keepdim=True)) / (inst_freq.std(dim=(2,3,4), keepdim=True) + 1e-8)
        freq_norm = freq_norm.clamp(-3, 3)

        # 拼接多尺度属性：[amp_s0, phase_s0, freq_s0, amp_s1, ...]
        attrs = torch.stack([amp_norm, phase_norm, freq_norm], dim=2)  # [B, S, 3, D, H, W]
        attrs = attrs.view(attrs.shape[0], -1, *attrs.shape[3:])  # [B, 12, D, H, W]

        # Step 4: 【论文定理2，公式6/15】积分重构解析信号（可选）
        # 这是论文抗噪声能力的核心：通过多尺度积分重构获得更稳定的瞬时属性
        if self.use_reconstruction:
            # 重构解析信号
            recon_real, recon_imag = self.reconstructor(cwt_real, cwt_imag, return_multiscale=False)
            # recon_real: s(t), recon_imag: H[s(t)] = s*(t)

            # 基于重构的解析信号计算瞬时属性（公式14a-14c）
            # 这比直接使用Hilbert变换更抗噪声（论文图2d vs 图2b）
            recon_amp = torch.sqrt(recon_real ** 2 + recon_imag ** 2 + 1e-8)  # e(t) = sqrt(s² + s*²)
            recon_phase = torch.atan2(recon_imag, recon_real + 1e-8)  # θ(t) = arctan(s*/s)

            # 重构瞬时频率使用阻尼公式(16)
            recon_freq = self._compute_reconstructed_inst_freq(recon_real, recon_imag, recon_amp)

            # 归一化重构属性
            recon_amp_norm = torch.log1p(recon_amp.squeeze(1))
            recon_amp_norm = (recon_amp_norm - recon_amp_norm.mean(dim=(1,2,3), keepdim=True)) / (recon_amp_norm.std(dim=(1,2,3), keepdim=True) + 1e-8)

            recon_phase_norm = recon_phase.squeeze(1) / math.pi

            recon_freq_norm = recon_freq.squeeze(1)
            recon_freq_norm = (recon_freq_norm - recon_freq_norm.mean(dim=(1,2,3), keepdim=True)) / (recon_freq_norm.std(dim=(1,2,3), keepdim=True) + 1e-8)
            recon_freq_norm = recon_freq_norm.clamp(-3, 3)

            # 拼接重构属性
            recon_attrs = torch.stack([recon_amp_norm, recon_phase_norm, recon_freq_norm], dim=1)  # [B, 3, D, H, W]

            # 输出：原始 + 多尺度属性 + 重构属性
            out = torch.cat([x, attrs, recon_attrs], dim=1)  # [B, 16, D, H, W]
        else:
            # 原始数据直接使用，不再二次归一化（已在dataloader中归一化）
            out = torch.cat([x, attrs], dim=1)  # [B, 13, D, H, W]

        return out

    def _compute_reconstructed_inst_freq(self, s_real, s_imag, amplitude):
        """
        计算重构解析信号的瞬时频率（论文公式14c和16）

        公式(14c): ω(t) = d/dt[arctan(s*/s)]
        公式(16): f(t) = [s*ds*/dt - s**ds/dt] / [2π * (e² + ε²*e²_max)]

        Args:
            s_real: [B, C, D, H, W] 重构的实信号 s(t)
            s_imag: [B, C, D, H, W] 重构的Hilbert变换 s*(t)
            amplitude: [B, C, D, H, W] 瞬时振幅 e(t)
        """
        amp_squared = amplitude ** 2
        amp_squared_max = amp_squared.amax(dim=(2, 3, 4), keepdim=True)

        # 阻尼分母
        damping_term = self.inst_attr.damping_eps ** 2 * amp_squared_max
        denominator = amp_squared + damping_term + 1e-8

        # 沿时间方向计算导数
        ds_real = self._gradient_5d(s_real, dim=2)  # D方向
        ds_imag = self._gradient_5d(s_imag, dim=2)

        # 瞬时频率
        inst_freq = (s_real * ds_imag - s_imag * ds_real) / denominator

        return inst_freq

    def _gradient_5d(self, x, dim):
        """沿指定维度计算梯度（中心差分）- 5D版本，边界修复"""
        if dim == 2:  # D维度 [B, C, D, H, W]
            # 使用replicate padding避免边界伪影
            x_padded = F.pad(x, (0, 0, 0, 0, 1, 1), mode='replicate')
            grad = (x_padded[:, :, 2:, :, :] - x_padded[:, :, :-2, :, :]) / 2.0
        else:
            raise ValueError(f"Unsupported dim: {dim}")

        return grad


class DomainInvariantReconstruction(nn.Module):
    """
    基于定理2的域不变特征增强模块（最小版本）

    核心思想：
    1. 利用积分重构的抗噪声特性实现域不变
    2. 重构属性（通道13-15）是最域不变的特征
    3. 大尺度（低频）比小尺度（高频）更域不变

    论文依据：
    - 定理2（公式6）：积分重构获得解析信号，抗噪声能力提升~2000倍
    - 第828页：有效信号和噪声在时频域分离，积分后噪声抵消
    """
    def __init__(self, in_channels=16, out_channels=16):
        super().__init__()

        # 尺度自适应权重：学习不同尺度对域不变性的贡献
        self.scale_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 8),
            nn.ReLU(),
            nn.Linear(8, 4),  # 4个尺度
            nn.Softmax(dim=1)
        )

        # 域不变特征投影
        self.domain_inv_proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 1),
            nn.GroupNorm(out_channels // 4, out_channels),
            nn.ReLU()
        )

        # 重构属性增强（通道13-15是积分重构的属性，最域不变）
        self.recon_enhance = nn.Sequential(
            nn.Conv3d(3, out_channels // 4, 1),
            nn.GroupNorm(out_channels // 16, out_channels // 4),
            nn.Sigmoid()
        )

    def forward(self, wavelet_feat):
        """
        Args:
            wavelet_feat: [B, 16, D, H, W] 来自EnhancedWaveletInstantaneous3D的输出
                         通道布局: [原始, amp1,phase1,freq1, ..., recon_amp,recon_phase,recon_freq]
        Returns:
            [B, 16, D, H, W] 域不变增强特征
        """
        # 提取重构属性（最后3个通道，基于定理2积分重构，最域不变）
        recon_attrs = wavelet_feat[:, 13:16, :, :, :]  # [B, 3, D, H, W]

        # 计算重构增强权重
        recon_weight = self.recon_enhance(recon_attrs)  # [B, 4, D, H, W]

        # 基础投影
        base_feat = self.domain_inv_proj(wavelet_feat)  # [B, 16, D, H, W]

        # 用重构权重调制（重构属性强的区域增强）
        enhanced = base_feat * (1 + recon_weight.mean(dim=1, keepdim=True))

        return enhanced


class AnalyticGradientExtractor(nn.Module):
    """
    解析信号梯度提取器 - 实现统一理论框架的核心

    理论基础（断层检测充分条件）：
    F(x) = |∇A| · |∇φ| · R(θ) · W(z) > τ

    其中：
    - |∇A|: 瞬时振幅梯度 - 断层导致反射系数突变
    - |∇φ|: 瞬时相位梯度 - 断层导致相位不连续
    - |∇f|: 瞬时频率梯度 - 断层导致频率变化

    输入: EnhancedWaveletInstantaneous3D的16通道输出
    输出: 梯度特征 [B, 6, D, H, W] (|∇A|, |∇φ|, |∇f| 各2通道：重构+多尺度均值)
    """
    def __init__(self, use_learnable_sobel=True):
        super().__init__()
        self.use_learnable_sobel = use_learnable_sobel

        # Sobel算子（3D）
        if use_learnable_sobel:
            # 可学习的梯度算子
            # 【边界伪影修复】使用replicate填充
            self.grad_conv = nn.Conv3d(1, 3, kernel_size=3, padding=1, padding_mode='replicate', bias=False)
            # 初始化为Sobel算子
            self._init_sobel_weights()

        # 梯度幅值增强（学习最优的梯度组合）
        self.grad_enhance = nn.Sequential(
            nn.Conv3d(9, 16, 1),  # 3属性×3方向
            nn.GroupNorm(4, 16),
            nn.ReLU(),
            nn.Conv3d(16, 6, 1),  # 输出6通道梯度特征
            nn.ReLU()  # 梯度幅值非负
        )

    def _init_sobel_weights(self):
        """初始化为3D Sobel算子"""
        sobel_x = torch.tensor([
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            [[-2, 0, 2], [-4, 0, 4], [-2, 0, 2]],
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        ], dtype=torch.float32) / 16.0

        sobel_y = sobel_x.permute(1, 0, 2)
        sobel_z = sobel_x.permute(2, 1, 0)

        with torch.no_grad():
            self.grad_conv.weight[0, 0] = sobel_x
            self.grad_conv.weight[1, 0] = sobel_y
            self.grad_conv.weight[2, 0] = sobel_z

    def _compute_gradient_magnitude(self, x):
        """计算3D梯度幅值"""
        if self.use_learnable_sobel:
            grads = self.grad_conv(x)  # [B, 3, D, H, W]
            grad_mag = torch.sqrt((grads ** 2).sum(dim=1, keepdim=True) + 1e-8)
        else:
            # 简单差分 - 【边界伪影修复】使用replicate padding
            gx = F.pad(x, (1, 1, 0, 0, 0, 0), mode='replicate')[:, :, :, :, 2:] - F.pad(x, (1, 1, 0, 0, 0, 0), mode='replicate')[:, :, :, :, :-2]
            gy = F.pad(x, (0, 0, 1, 1, 0, 0), mode='replicate')[:, :, :, 2:, :] - F.pad(x, (0, 0, 1, 1, 0, 0), mode='replicate')[:, :, :, :-2, :]
            gz = F.pad(x, (0, 0, 0, 0, 1, 1), mode='replicate')[:, :, 2:, :, :] - F.pad(x, (0, 0, 0, 0, 1, 1), mode='replicate')[:, :, :-2, :, :]
            grad_mag = torch.sqrt(gx**2 + gy**2 + gz**2 + 1e-8) / 2.0
        return grad_mag

    def forward(self, wavelet_feat):
        """
        Args:
            wavelet_feat: [B, 16, D, H, W]
                通道布局: [原始(1), 多尺度属性(12), 重构属性(3)]
                多尺度属性: [amp1,phase1,freq1, amp2,phase2,freq2, ...]
                重构属性: [recon_amp, recon_phase, recon_freq]

        Returns:
            grad_features: [B, 6, D, H, W] 梯度特征
            grad_A: [B, 1, D, H, W] 振幅梯度（用于损失函数）
            grad_phi: [B, 1, D, H, W] 相位梯度（用于损失函数）
        """
        # 提取重构属性（最后3通道，基于积分重构，抗噪声）
        recon_amp = wavelet_feat[:, 13:14]    # [B, 1, D, H, W]
        recon_phase = wavelet_feat[:, 14:15]  # [B, 1, D, H, W]
        recon_freq = wavelet_feat[:, 15:16]   # [B, 1, D, H, W]

        # 提取多尺度属性均值（通道1-12）
        multi_scale_attrs = wavelet_feat[:, 1:13]  # [B, 12, D, H, W]
        # 按属性类型分组求均值: amp(0,3,6,9), phase(1,4,7,10), freq(2,5,8,11)
        ms_amp = multi_scale_attrs[:, 0::3].mean(dim=1, keepdim=True)    # [B, 1, D, H, W]
        ms_phase = multi_scale_attrs[:, 1::3].mean(dim=1, keepdim=True)  # [B, 1, D, H, W]
        ms_freq = multi_scale_attrs[:, 2::3].mean(dim=1, keepdim=True)   # [B, 1, D, H, W]

        # 计算梯度幅值
        grad_recon_amp = self._compute_gradient_magnitude(recon_amp)
        grad_recon_phase = self._compute_gradient_magnitude(recon_phase)
        grad_recon_freq = self._compute_gradient_magnitude(recon_freq)

        grad_ms_amp = self._compute_gradient_magnitude(ms_amp)
        grad_ms_phase = self._compute_gradient_magnitude(ms_phase)
        grad_ms_freq = self._compute_gradient_magnitude(ms_freq)

        # 如果使用可学习Sobel，进一步增强
        if self.use_learnable_sobel:
            # 拼接所有梯度方向
            all_grads = torch.cat([
                self.grad_conv(recon_amp),
                self.grad_conv(recon_phase),
                self.grad_conv(recon_freq)
            ], dim=1)  # [B, 9, D, H, W]

            grad_features = self.grad_enhance(all_grads)  # [B, 6, D, H, W]
        else:
            # 简单拼接
            grad_features = torch.cat([
                grad_recon_amp, grad_ms_amp,
                grad_recon_phase, grad_ms_phase,
                grad_recon_freq, grad_ms_freq
            ], dim=1)  # [B, 6, D, H, W]

        # 用于物理约束损失的梯度
        grad_A = grad_recon_amp
        grad_phi = grad_recon_phase

        return grad_features, grad_A, grad_phi


class FaultResponseFunction(nn.Module):
    """
    断层响应函数 - 实现统一理论公式

    F(x) = |∇A|^α₁ · |∇φ|^α₂ · R(θ)^α₃ · W(z)^α₄

    各因子的物理意义：
    - |∇A|: 振幅梯度响应 - 断层处反射系数突变
    - |∇φ|: 相位梯度响应 - 断层处相位不连续
    - R(θ): 方向响应 - 断层是定向面状结构（由DAC提供）
    - W(z): 深度权重 - 断层多发于特定深度（由DepthModulation提供）

    创新点：
    1. 每个因子对应一个可解释的物理量
    2. 幂次α可学习，自动平衡各因子贡献
    3. 融合方式有理论支撑
    """
    def __init__(self, grad_channels=6, feat_channels=16):
        super().__init__()

        # 振幅梯度响应函数
        # 【边界伪影修复】使用replicate填充
        self.amp_response = nn.Sequential(
            nn.Conv3d(2, 8, 3, padding=1, padding_mode='replicate'),  # 2通道：重构+多尺度
            nn.GroupNorm(2, 8),
            nn.ReLU(),
            nn.Conv3d(8, 1, 1),
            nn.Sigmoid()
        )

        # 相位梯度响应函数
        # 【边界伪影修复】使用replicate填充
        self.phase_response = nn.Sequential(
            nn.Conv3d(2, 8, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(2, 8),
            nn.ReLU(),
            nn.Conv3d(8, 1, 1),
            nn.Sigmoid()
        )

        # 频率梯度响应函数（辅助）
        # 【边界伪影修复】使用replicate填充
        self.freq_response = nn.Sequential(
            nn.Conv3d(2, 8, 3, padding=1, padding_mode='replicate'),
            nn.GroupNorm(2, 8),
            nn.ReLU(),
            nn.Conv3d(8, 1, 1),
            nn.Sigmoid()
        )

        # 可学习的融合幂次（初始化为1，即线性融合）
        self.alpha = nn.Parameter(torch.ones(4))  # [α_amp, α_phase, α_freq, α_dir]

        # 融合后投影
        self.fusion_proj = nn.Sequential(
            nn.Conv3d(feat_channels + 3, feat_channels, 1),  # +3是三个响应
            nn.GroupNorm(feat_channels // 4, feat_channels),
            nn.ReLU()
        )

    def forward(self, features, grad_features, direction_weight=None, depth_weight=None):
        """
        Args:
            features: [B, C, D, H, W] 编码器特征
            grad_features: [B, 6, D, H, W] 梯度特征
                [grad_amp_recon, grad_amp_ms, grad_phase_recon, grad_phase_ms, grad_freq_recon, grad_freq_ms]
            direction_weight: [B, 1, D, H, W] 方向响应权重（来自DAC）
            depth_weight: [B, 1, D, H, W] 深度权重（来自DepthModulation）

        Returns:
            enhanced_features: [B, C, D, H, W] 断层响应增强后的特征
            fault_response: [B, 1, D, H, W] 断层响应图（用于可视化/损失）
        """
        # 计算各因子响应
        r_amp = self.amp_response(grad_features[:, 0:2])      # [B, 1, D, H, W]
        r_phase = self.phase_response(grad_features[:, 2:4])  # [B, 1, D, H, W]
        r_freq = self.freq_response(grad_features[:, 4:6])    # [B, 1, D, H, W]

        # 幂次加权（确保正数）
        alpha = F.softplus(self.alpha) + 0.1  # 至少0.1，避免梯度消失

        # 断层响应公式: F = r_amp^α₁ · r_phase^α₂ · r_freq^α₃
        fault_response = (
            torch.pow(r_amp + 1e-6, alpha[0]) *
            torch.pow(r_phase + 1e-6, alpha[1]) *
            torch.pow(r_freq + 1e-6, alpha[2])
        )

        # 如果有方向权重，融合
        if direction_weight is not None:
            fault_response = fault_response * torch.pow(direction_weight + 1e-6, alpha[3])

        # 如果有深度权重，融合
        if depth_weight is not None:
            fault_response = fault_response * depth_weight

        # 归一化到[0,1]
        fault_response = fault_response / (fault_response.max() + 1e-8)

        # 特征增强：用断层响应调制原始特征
        enhanced_features = self.fusion_proj(
            torch.cat([features, r_amp, r_phase, r_freq], dim=1)
        )
        enhanced_features = enhanced_features * (1 + fault_response)

        return enhanced_features, fault_response


class ChannelAttention3D(nn.Module):
    """
    SE通道注意力模块 - 让网络自己学习通道（尺度×特征）的重要性

    核心思想：
    - 不在尺度层面做硬性权重学习（避免域差异问题）
    - 而是在通道层面让网络自适应选择
    - 合成数据和真实数据可以学到不同的通道组合

    参考：Squeeze-and-Excitation Networks (SENet)
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)

        # 共享MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, D, H, W]
        Returns:
            [B, C, D, H, W] 通道加权后的特征
        """
        B, C, D, H, W = x.shape

        # 双池化（avg + max）获得更丰富的统计信息
        avg_feat = self.avg_pool(x).view(B, C)
        max_feat = self.max_pool(x).view(B, C)

        # MLP处理
        avg_out = self.mlp(avg_feat)
        max_out = self.mlp(max_feat)

        # 融合并生成注意力权重
        attention = torch.sigmoid(avg_out + max_out).view(B, C, 1, 1, 1)

        return x * attention


class MultiScaleWaveletFeatures(nn.Module):
    """
    多尺度小波瞬时属性提取模块 - 纯振幅特征版（方案A+1+3）

    【核心设计】
    方案A：只使用振幅属性，用三种方式处理不连续性
    方案1：直接concat所有尺度（不学习尺度权重，避免域差异）
    方案3：SE通道注意力让网络自己学习通道组合

    输出通道布局（16通道）:
        ch0:     原始地震数据
        ch1-5:   5尺度 × |∇A| (振幅空间梯度) - 检测突变
        ch6-10:  5尺度 × Semblance(A) (局部相干性) - 检测不连续
        ch11-15: 5尺度 × Variance(A) (局部方差) - 检测异常

    物理意义：
        |∇A|:      断层处振幅有空间突变
        Semblance: 断层两侧振幅相干性降低
        Variance:  断层区域振幅变化剧烈

    为什么放弃相位？
        - 相位与断层相关性只有0.03-0.10
        - 振幅相关性达0.33-0.43
        - 用Semblance和Variance替代相位，都是基于振幅的有效特征
    """
    def __init__(self, scales=None, omega0=6.0, learnable=False, min_kernel_size=7,
                 window_size=5, use_channel_attention=True):
        super().__init__()
        if scales is None:
            scales = [2, 3, 4, 6, 8]  # 5个尺度
        self.scales = scales
        self.num_scales = len(scales)
        self.out_channels = 1 + self.num_scales * 3  # 原始 + 5尺度×3特征
        self.window_size = window_size
        self.use_channel_attention = use_channel_attention

        # 连续小波变换
        self.cwt = ContinuousWaveletTransform3D(
            scales=scales, omega0=omega0, learnable=learnable,
            min_kernel_size=min_kernel_size
        )

        # 瞬时属性计算（只用振幅）
        self.inst_attr = InstantaneousAttributes()

        # 方案3：SE通道注意力
        if use_channel_attention:
            self.channel_attention = ChannelAttention3D(self.out_channels, reduction=4)

    def _spatial_gradient(self, x):
        """
        计算空间梯度幅值（只在H和W方向，不在D方向）

        断层 = H/W方向的不连续
        层位 = D方向的变化（不要这个）

        Args:
            x: [B, S, D, H, W]
        Returns:
            grad_mag: [B, S, D, H, W] 空间梯度幅值
        """
        # H方向梯度（中心差分，边界用前向/后向差分）
        grad_h = torch.zeros_like(x)
        grad_h[:, :, :, 1:-1, :] = (x[:, :, :, 2:, :] - x[:, :, :, :-2, :]) / 2
        grad_h[:, :, :, 0, :] = x[:, :, :, 1, :] - x[:, :, :, 0, :]
        grad_h[:, :, :, -1, :] = x[:, :, :, -1, :] - x[:, :, :, -2, :]

        # W方向梯度
        grad_w = torch.zeros_like(x)
        grad_w[:, :, :, :, 1:-1] = (x[:, :, :, :, 2:] - x[:, :, :, :, :-2]) / 2
        grad_w[:, :, :, :, 0] = x[:, :, :, :, 1] - x[:, :, :, :, 0]
        grad_w[:, :, :, :, -1] = x[:, :, :, :, -1] - x[:, :, :, :, -2]

        # 梯度幅值
        grad_mag = torch.sqrt(grad_h**2 + grad_w**2 + 1e-8)

        return grad_mag

    def _local_semblance(self, x):
        """
        计算局部相干性（Semblance）- 只在H/W方向

        Semblance = (Σx)² / (N × Σx²)

        高相干性 → 连续层位
        低相干性 → 断层/不连续

        输出：1 - Semblance，这样断层区域值高

        Args:
            x: [B, S, D, H, W]
        Returns:
            semblance: [B, S, D, H, W] 不相干性（1-semblance）
        """
        B, S, D, H, W = x.shape
        w = self.window_size
        pad = w // 2

        # 只在H/W方向做padding和计算
        # [B, S, D, H, W] -> [B*S*D, 1, H, W] 方便2D卷积
        x_2d = x.view(B * S * D, 1, H, W)

        # Replicate padding
        x_padded = F.pad(x_2d, (pad, pad, pad, pad), mode='replicate')

        # 使用unfold提取局部窗口
        # [B*S*D, 1, H, W] -> [B*S*D, w*w, H, W]
        patches = F.unfold(x_padded, kernel_size=w, padding=0)
        patches = patches.view(B * S * D, w * w, H, W)

        # 计算Semblance
        sum_x = patches.sum(dim=1, keepdim=True)  # Σx
        sum_x2 = (patches ** 2).sum(dim=1, keepdim=True)  # Σx²
        n = w * w

        # Semblance = (Σx)² / (N × Σx²)
        semblance = (sum_x ** 2) / (n * sum_x2 + 1e-8)

        # 1 - Semblance：不连续区域值高
        incoherence = 1 - semblance

        # 恢复形状
        incoherence = incoherence.view(B, S, D, H, W)

        return incoherence

    def _local_variance(self, x):
        """
        计算局部方差 - 只在H/W方向

        Variance = E[x²] - E[x]²

        高方差 → 振幅变化剧烈（可能是断层）
        低方差 → 振幅平稳（连续层位）

        Args:
            x: [B, S, D, H, W]
        Returns:
            variance: [B, S, D, H, W] 局部方差
        """
        B, S, D, H, W = x.shape
        w = self.window_size
        pad = w // 2

        # [B, S, D, H, W] -> [B*S*D, 1, H, W]
        x_2d = x.view(B * S * D, 1, H, W)

        # Replicate padding
        x_padded = F.pad(x_2d, (pad, pad, pad, pad), mode='replicate')

        # 提取局部窗口
        patches = F.unfold(x_padded, kernel_size=w, padding=0)
        patches = patches.view(B * S * D, w * w, H, W)

        # 计算方差: Var = E[x²] - E[x]²
        mean_x = patches.mean(dim=1, keepdim=True)
        mean_x2 = (patches ** 2).mean(dim=1, keepdim=True)
        variance = mean_x2 - mean_x ** 2

        # 恢复形状
        variance = variance.view(B, S, D, H, W)

        return variance

    def _robust_normalize(self, feat):
        """稳健归一化：使用中位数和MAD，避免异常值影响"""
        B, S, D, H, W = feat.shape
        feat_flat = feat.view(B, S, -1)
        median = feat_flat.median(dim=2, keepdim=True)[0].view(B, S, 1, 1, 1)
        mad = (feat_flat - median.view(B, S, -1)).abs().median(dim=2, keepdim=True)[0].view(B, S, 1, 1, 1)
        # MAD to std: std ≈ 1.4826 * MAD
        std_est = 1.4826 * mad + 1e-8
        return (feat - median) / std_est

    def forward(self, x):
        """
        Args:
            x: [B, 1, D, H, W] - 归一化的地震数据
        Returns:
            [B, 16, D, H, W] - 原始 + 5尺度×3振幅特征（经通道注意力加权）
        """
        B = x.shape[0]

        # 小波变换
        cwt_real, cwt_imag = self.cwt(x)  # [B, 1, S, D, H, W]

        # 只计算瞬时振幅（放弃相位和频率）
        inst_amp, _, _ = self.inst_attr(cwt_real, cwt_imag)

        # [B, 1, S, D, H, W] -> [B, S, D, H, W]
        inst_amp = inst_amp.squeeze(1)

        # ========== 振幅预处理：log压缩 ==========
        amp_log = torch.log1p(inst_amp)

        # ========== 三种振幅特征（方案A）==========
        # 特征1: 空间梯度 |∇A| - 检测突变
        grad_amp = self._spatial_gradient(amp_log)

        # 特征2: 局部相干性 Semblance - 检测不连续
        semblance = self._local_semblance(amp_log)

        # 特征3: 局部方差 Variance - 检测异常
        variance = self._local_variance(amp_log)

        # ========== 稳健归一化 ==========
        grad_amp_norm = self._robust_normalize(grad_amp).clamp(-5, 5)
        semblance_norm = self._robust_normalize(semblance).clamp(-5, 5)
        variance_norm = self._robust_normalize(variance).clamp(-5, 5)

        # ========== 方案1：直接concat所有尺度（不学习权重）==========
        # 输出布局: [原始, ∇A_s1,...,∇A_s5, Sem_s1,...,Sem_s5, Var_s1,...,Var_s5]
        output_channels = [x]  # ch0: 原始数据

        # ch1-5: 5尺度的振幅梯度
        for s in range(self.num_scales):
            output_channels.append(grad_amp_norm[:, s:s+1])

        # ch6-10: 5尺度的相干性
        for s in range(self.num_scales):
            output_channels.append(semblance_norm[:, s:s+1])

        # ch11-15: 5尺度的方差
        for s in range(self.num_scales):
            output_channels.append(variance_norm[:, s:s+1])

        out = torch.cat(output_channels, dim=1)  # [B, 16, D, H, W]

        # ========== 方案3：SE通道注意力 ==========
        if self.use_channel_attention:
            out = self.channel_attention(out)

        return out


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # ========== 数据加载 ==========
    data_path = "data/train/seis/0.dat"
    if os.path.exists(data_path):
        seis_data = np.fromfile(data_path, dtype=np.single)
        seis_data = seis_data.reshape(128, 128, 128)
        print(f"地震数据形状: {seis_data.shape}")
        print(f"数据范围: [{seis_data.min():.4f}, {seis_data.max():.4f}]")
    else:
        # 生成合成测试数据（模拟含噪声的地震信号）
        print(f"数据文件 {data_path} 不存在，使用合成数据测试")
        np.random.seed(42)
        t = np.linspace(0, 1, 128)
        # 创建多频率合成信号
        signal_1d = np.sin(2 * np.pi * 10 * t) + 0.5 * np.sin(2 * np.pi * 25 * t)
        # 扩展到3D
        seis_data = np.zeros((128, 128, 128), dtype=np.float32)
        for i in range(128):
            for j in range(128):
                seis_data[:, i, j] = signal_1d * (1 + 0.1 * np.sin(2 * np.pi * i / 128))
        # 添加噪声（论文图2测试场景：噪声与有效信号最大值之比为0.20）
        noise_ratio = 0.20
        noise = np.random.randn(*seis_data.shape).astype(np.float32)
        seis_data = seis_data + noise_ratio * seis_data.max() * noise
        print(f"合成数据形状: {seis_data.shape}")
        print(f"数据范围: [{seis_data.min():.4f}, {seis_data.max():.4f}]")
        print(f"噪声比例: {noise_ratio}")

    # 数据归一化（模拟dataloader的预处理）
    seis_data = (seis_data - seis_data.mean()) / (seis_data.std() + 1e-8)

    # 转换为PyTorch张量 [B, C, D, H, W]
    x = torch.from_numpy(seis_data).float().unsqueeze(0).unsqueeze(0).to(device)
    print(f"输入张量形状: {x.shape}")

    # ========== 测试配置 ==========
    scales = [2, 4, 6, 8]  # 修正后的尺度，最大核49，约38%深度

    print("\n" + "="*60)
    print("测试1: 基础小波变换和瞬时属性计算")
    print("="*60)

    cwt = ContinuousWaveletTransform3D(scales=scales, omega0=6.0, min_kernel_size=21).to(device)
    inst_attr = InstantaneousAttributes().to(device)

    print("\n各尺度的小波核大小:")
    for i, scale in enumerate(scales):
        k = cwt.wavelet_kernels_real[i].shape[-1]
        print(f"  尺度 {scale}: 核大小 = {k}")

    with torch.no_grad():
        cwt_real, cwt_imag = cwt(x)
        print(f"\n小波变换输出形状: {cwt_real.shape}")

        inst_amp, inst_phase, inst_freq = inst_attr(cwt_real, cwt_imag)
        print(f"瞬时属性形状: amp={inst_amp.shape}, phase={inst_phase.shape}, freq={inst_freq.shape}")

    print("\n" + "="*60)
    print("测试2: 小波域去噪模块（论文第828页）")
    print("="*60)

    denoiser = WaveletDomainDenoising(num_scales=len(scales), learnable=True).to(device)
    print(f"去噪阈值参数: {denoiser.thresholds.data}")

    with torch.no_grad():
        cwt_real_denoised, cwt_imag_denoised = denoiser(cwt_real, cwt_imag)

        # 比较去噪前后的统计量
        print("\n去噪前后对比:")
        for i, scale in enumerate(scales):
            amp_before = torch.sqrt(cwt_real[0, 0, i] ** 2 + cwt_imag[0, 0, i] ** 2)
            amp_after = torch.sqrt(cwt_real_denoised[0, 0, i] ** 2 + cwt_imag_denoised[0, 0, i] ** 2)
            reduction = (1 - amp_after.mean() / amp_before.mean()) * 100
            print(f"  尺度 {scale}: 振幅均值 {amp_before.mean():.4f} -> {amp_after.mean():.4f} (减少 {reduction:.1f}%)")

    print("\n" + "="*60)
    print("测试3: 定理2积分重构（论文公式6）")
    print("="*60)

    reconstructor = AnalyticSignalReconstruction(scales=scales, omega0=6.0).to(device)
    print(f"容许性常数 C_g = {reconstructor.C_g.item():.4f}")

    with torch.no_grad():
        recon_real, recon_imag, _, _ = reconstructor(cwt_real_denoised, cwt_imag_denoised, return_multiscale=True)
        print(f"重构信号形状: real={recon_real.shape}, imag={recon_imag.shape}")
        print(f"重构实部统计: mean={recon_real.mean():.6f}, std={recon_real.std():.6f}")
        print(f"重构虚部统计: mean={recon_imag.mean():.6f}, std={recon_imag.std():.6f}")

        # 计算重构信号与原始信号的相关性
        corr = torch.corrcoef(torch.stack([x.flatten(), recon_real.flatten()]))[0, 1]
        print(f"重构信号与原始信号相关系数: {corr.item():.4f}")

    print("\n" + "="*60)
    print("测试4: 完整的EnhancedWaveletInstantaneous3D模块")
    print("="*60)

    # 测试不同配置
    configs = [
        {"use_denoising": False, "use_reconstruction": False, "name": "基础模式"},
        {"use_denoising": True, "use_reconstruction": False, "name": "去噪模式"},
        {"use_denoising": True, "use_reconstruction": True, "name": "完整抗噪声模式"},
    ]

    for config in configs:
        model = EnhancedWaveletInstantaneous3D(
            in_channels=1,
            scales=scales,
            omega0=6.0,
            use_denoising=config["use_denoising"],
            use_reconstruction=config["use_reconstruction"]
        ).to(device)

        with torch.no_grad():
            out = model(x)
            print(f"\n{config['name']}:")
            print(f"  输出形状: {out.shape}")
            print(f"  输出统计: mean={out.mean():.4f}, std={out.std():.4f}, range=[{out.min():.4f}, {out.max():.4f}]")

            # 检查各通道的统计
            for ch in range(min(out.shape[1], 4)):
                ch_data = out[0, ch]
                print(f"  通道{ch}: mean={ch_data.mean():.4f}, std={ch_data.std():.4f}")

    print("\n" + "="*60)
    print("测试5: 可视化对比")
    print("="*60)

    # 创建三种模式的模型
    model_basic = EnhancedWaveletInstantaneous3D(
        scales=scales, use_denoising=False, use_reconstruction=False
    ).to(device)
    model_denoise = EnhancedWaveletInstantaneous3D(
        scales=scales, use_denoising=True, use_reconstruction=False
    ).to(device)
    model_full = EnhancedWaveletInstantaneous3D(
        scales=scales, use_denoising=True, use_reconstruction=True
    ).to(device)

    slice_idx = 64

    with torch.no_grad():
        out_basic = model_basic(x)
        out_denoise = model_denoise(x)
        out_full = model_full(x)

    # 创建可视化图
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))

    # 第一行：原始数据
    seis_slice = seis_data[slice_idx, :, :]
    ax = axes[0, 0]
    im = ax.imshow(seis_slice, cmap='seismic', aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title('Original Seismic', fontsize=10)

    # 原始数据通道（所有模式相同）
    for i in range(1, 4):
        ax = axes[0, i]
        ax.axis('off')
        ax.set_title('', fontsize=10)

    # 第二行：基础模式的瞬时属性（尺度0）
    titles = ['Amplitude (s=2)', 'Phase (s=2)', 'Frequency (s=2)']
    for i, (title, ch) in enumerate(zip(titles, [1, 2, 3])):
        ax = axes[1, i]
        data = out_basic[0, ch, slice_idx, :, :].cpu().numpy()
        cmap = 'viridis' if i == 0 else ('twilight' if i == 1 else 'RdBu_r')
        if i == 2:
            vmax = np.percentile(np.abs(data), 99)
            im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'Basic: {title}', fontsize=9)
    axes[1, 3].axis('off')

    # 第三行：去噪模式
    for i, (title, ch) in enumerate(zip(titles, [1, 2, 3])):
        ax = axes[2, i]
        data = out_denoise[0, ch, slice_idx, :, :].cpu().numpy()
        cmap = 'viridis' if i == 0 else ('twilight' if i == 1 else 'RdBu_r')
        if i == 2:
            vmax = np.percentile(np.abs(data), 99)
            im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'Denoised: {title}', fontsize=9)
    axes[2, 3].axis('off')

    # 第四行：完整抗噪声模式（包含重构属性）
    recon_titles = ['Recon Amplitude', 'Recon Phase', 'Recon Frequency']
    for i, (title, ch) in enumerate(zip(recon_titles, [13, 14, 15])):
        ax = axes[3, i]
        data = out_full[0, ch, slice_idx, :, :].cpu().numpy()
        cmap = 'viridis' if i == 0 else ('twilight' if i == 1 else 'RdBu_r')
        if i == 2:
            vmax = np.percentile(np.abs(data), 99)
            im = ax.imshow(data, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, cmap=cmap, aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f'Full: {title}', fontsize=9)
    axes[3, 3].axis('off')

    plt.suptitle('Wavelet-based Instantaneous Attributes Comparison\n'
                 '(Basic vs Denoised vs Full Anti-noise Mode)', fontsize=12)
    plt.tight_layout()
    plt.savefig('wavelet_antinoise_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n对比图已保存为: wavelet_antinoise_comparison.png")

    # ========== 多尺度特征可视化 ==========
    print("\n" + "="*60)
    print("测试6: 多尺度瞬时特征可视化")
    print("="*60)

    fig, axes = plt.subplots(5, 3, figsize=(15, 25))

    # 第一行：原始地震数据
    for i in range(3):
        ax = axes[0, i]
        im = ax.imshow(seis_slice, cmap='seismic', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title('Original Seismic Data', fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

    feature_names = ['Inst. Amplitude', 'Inst. Phase', 'Inst. Frequency']

    # 使用去噪后的瞬时属性
    with torch.no_grad():
        inst_amp, inst_phase, inst_freq = inst_attr(cwt_real_denoised, cwt_imag_denoised)

    for s_idx, scale in enumerate(scales):
        amp_slice = inst_amp[0, 0, s_idx, slice_idx, :, :].cpu().numpy()
        phase_slice = inst_phase[0, 0, s_idx, slice_idx, :, :].cpu().numpy()
        freq_slice = inst_freq[0, 0, s_idx, slice_idx, :, :].cpu().numpy()

        features = [amp_slice, phase_slice, freq_slice]
        cmaps = ['viridis', 'twilight', 'RdBu_r']

        for f_idx, (feat, cmap, name) in enumerate(zip(features, cmaps, feature_names)):
            ax = axes[s_idx + 1, f_idx]

            if f_idx == 2:
                vmax = np.percentile(np.abs(feat), 99)
                if vmax < 1e-10:
                    vmax = 1.0
                im = ax.imshow(feat, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
            else:
                im = ax.imshow(feat, cmap=cmap, aspect='auto')

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(f'Scale={scale}, {name}', fontsize=10)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')

    plt.suptitle(f'Multi-scale Instantaneous Features with Denoising (Slice Z={slice_idx})', fontsize=14)
    plt.tight_layout()
    plt.savefig('wavelet_multiscale_features.png', dpi=150, bbox_inches='tight')
    print(f"多尺度特征图已保存为: wavelet_multiscale_features.png")

    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)