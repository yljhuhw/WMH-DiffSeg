"""
Haar DWT/IWT + learnable gated wavelet modulation for skip connections.
No external dependencies (pywt not required).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2d(nn.Module):
    """Fixed Haar wavelet 2D DWT via depthwise convolution."""

    def __init__(self):
        super().__init__()
        # Haar filters: [LL, LH, HL, HH], shape [4, 1, 2, 2]
        haar = torch.tensor([
            [[[1,  1], [ 1,  1]]],   # LL
            [[[1,  1], [-1, -1]]],   # LH
            [[[1, -1], [ 1, -1]]],   # HL
            [[[1, -1], [-1,  1]]],   # HH
        ], dtype=torch.float32) * 0.5
        self.register_buffer('filters', haar)

    def forward(self, x):
        B, C, H, W = x.shape
        # reshape to apply per-channel
        x_r = x.reshape(B * C, 1, H, W)
        # replicate filters for all channels
        f = self.filters  # [4, 1, 2, 2]
        out = F.conv2d(x_r, f, stride=2)  # [B*C, 4, H/2, W/2]
        out = out.reshape(B, C, 4, H // 2, W // 2)
        LL = out[:, :, 0]
        LH = out[:, :, 1]
        HL = out[:, :, 2]
        HH = out[:, :, 3]
        return LL, LH, HL, HH


class HaarIWT2d(nn.Module):
    """Fixed Haar wavelet 2D IWT via transposed convolution."""

    def __init__(self):
        super().__init__()
        haar = torch.tensor([
            [[[1,  1], [ 1,  1]]],
            [[[1,  1], [-1, -1]]],
            [[[1, -1], [ 1, -1]]],
            [[[1, -1], [-1,  1]]],
        ], dtype=torch.float32) * 0.5
        # IWT filters: transpose of DWT
        self.register_buffer('filters', haar)

    def forward(self, LL, LH, HL, HH):
        B, C, H, W = LL.shape
        # stack subbands: [B, C, 4, H, W] -> [B*C, 4, H, W]
        x = torch.stack([LL, LH, HL, HH], dim=2).reshape(B * C, 4, H, W)
        f = self.filters  # [4, 1, 2, 2]
        # use conv_transpose2d: each subband reconstructs via its filter
        out = torch.zeros(B * C, 1, H * 2, W * 2, device=LL.device, dtype=LL.dtype)
        for i in range(4):
            out += F.conv_transpose2d(x[:, i:i+1], f[i:i+1], stride=2)
        return out.reshape(B, C, H * 2, W * 2)


class WaveletModulation(nn.Module):
    """
    Drop-in skip-connection modulation via Haar DWT -> learnable gates -> IWT.
    Shape-preserving: output same as input.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.dwt = HaarDWT2d()
        self.iwt = HaarIWT2d()
        # Per-channel learnable gates, initialized to preserve structure
        self.gate_ll = nn.Parameter(torch.full((1, channels, 1, 1), 0.9))
        self.gate_lh = nn.Parameter(torch.full((1, channels, 1, 1), 0.5))
        self.gate_hl = nn.Parameter(torch.full((1, channels, 1, 1), 0.5))
        self.gate_hh = nn.Parameter(torch.full((1, channels, 1, 1), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        LL, LH, HL, HH = self.dwt(x)
        LL = LL * torch.sigmoid(self.gate_ll)
        LH = LH * torch.sigmoid(self.gate_lh)
        HL = HL * torch.sigmoid(self.gate_hl)
        HH = HH * torch.sigmoid(self.gate_hh)
        return self.iwt(LL, LH, HL, HH)
