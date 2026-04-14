"""
Hybrid CNN-Transformer encoder for WMH-DiffSeg.
Replaces Generic_UNet highway with ~15M param encoder.
Interface: (features_list, calibration_map) same as highway_forward.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtBlock(nn.Module):
    def __init__(self, ch, expand=4):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 7, padding=3, groups=ch)
        self.norm = nn.LayerNorm(ch)
        self.pw1 = nn.Linear(ch, ch * expand)
        self.pw2 = nn.Linear(ch * expand, ch)

    def forward(self, x):
        h = self.dw(x)
        h = h.permute(0, 2, 3, 1)
        h = self.norm(h)
        h = self.pw2(F.gelu(self.pw1(h)))
        h = h.permute(0, 3, 1, 2)
        return x + h


class WindowAttention(nn.Module):
    def __init__(self, ch, num_heads=8, window_size=7):
        super().__init__()
        self.ws = window_size
        self.num_heads = num_heads
        self.head_dim = ch // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(ch, ch * 3)
        self.proj = nn.Linear(ch, ch)
        # relative position bias
        self.rel_bias = nn.Parameter(torch.zeros((2 * window_size - 1) ** 2, num_heads))
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing='ij'
        ))  # [2, ws, ws]
        coords_flat = coords.flatten(1)  # [2, ws*ws]
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # [2, ws*ws, ws*ws]
        rel = rel.permute(1, 2, 0).contiguous()
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer('rel_idx', rel.sum(-1))

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.ws
        # pad to multiple of window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape
        # partition windows
        x = x.permute(0, 2, 3, 1)  # B,H,W,C
        x = x.reshape(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, C)
        # attention
        qkv = self.qkv(x).reshape(-1, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.rel_bias[self.rel_idx.reshape(-1)].reshape(ws * ws, ws * ws, self.num_heads)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn.softmax(-1)
        x = (attn @ v).transpose(1, 2).reshape(-1, ws * ws, C)
        x = self.proj(x)
        # merge windows
        nw_h, nw_w = Hp // ws, Wp // ws
        x = x.reshape(B, nw_h, nw_w, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        x = x[:, :H, :W, :].permute(0, 3, 1, 2).contiguous()
        return x


class SwinBlock(nn.Module):
    def __init__(self, ch, num_heads=8, window_size=7, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(ch)
        self.attn = WindowAttention(ch, num_heads, window_size)
        self.norm2 = nn.LayerNorm(ch)
        self.mlp = nn.Sequential(
            nn.Linear(ch, ch * mlp_ratio),
            nn.GELU(),
            nn.Linear(ch * mlp_ratio, ch),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        h = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        h = self.norm1(h).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.attn(h)
        h = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        h = self.norm2(h)
        h = self.mlp(h).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + h


class PatchMerging(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm = nn.LayerNorm(in_ch * 4)
        self.proj = nn.Linear(in_ch * 4, out_ch)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B,H,W,C
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = self.norm(x)
        x = self.proj(x)
        return x.permute(0, 3, 1, 2)


class HybridEncoder(nn.Module):
    """
    Hybrid CNN-Transformer encoder (~15M params).
    Input: [B, in_ch, 256, 256]
    Output: (features_list, calibration_map [B,1,H,W])
    features_list[i] has channels matching main UNet skip levels [L1-L4].
    """
    # channels at each level (L0-L5 output channels)
    # DIRECTLY match UNet [128, 128, 256, 256, 512, 512] so adapters are near-identity
    CHANNELS = [128, 128, 256, 256, 512, 512]

    def __init__(self, in_ch=3, unet_channels=None):
        """
        unet_channels: list of 6 channel counts matching UNet skip levels [L0-L5]
        """
        super().__init__()
        C = self.CHANNELS

        # Level 0: 256x256, 128ch
        self.l0 = nn.Sequential(
            nn.Conv2d(in_ch, C[0], 3, padding=1), nn.BatchNorm2d(C[0]), nn.GELU(),
            ConvNeXtBlock(C[0]),
            ConvNeXtBlock(C[0]),
        )
        # Level 1: 128x128, 128ch
        self.down1 = nn.Sequential(
            nn.Conv2d(C[0], C[1], 3, stride=2, padding=1), nn.BatchNorm2d(C[1]), nn.GELU()
        )
        self.l1 = nn.Sequential(ConvNeXtBlock(C[1]), ConvNeXtBlock(C[1]))
        # Level 2: 64x64, 256ch
        self.down2 = nn.Sequential(
            nn.Conv2d(C[1], C[2], 3, stride=2, padding=1), nn.BatchNorm2d(C[2]), nn.GELU()
        )
        self.l2 = nn.Sequential(SwinBlock(C[2], num_heads=8), SwinBlock(C[2], num_heads=8))
        # Level 3: 32x32, 256ch
        self.down3 = nn.Sequential(
            nn.Conv2d(C[2], C[3], 3, stride=2, padding=1), nn.BatchNorm2d(C[3]), nn.GELU()
        )
        self.l3 = nn.Sequential(SwinBlock(C[3], num_heads=8), SwinBlock(C[3], num_heads=8))
        # Level 4: 16x16, 512ch
        self.down4 = nn.Sequential(
            nn.Conv2d(C[3], C[4], 3, stride=2, padding=1), nn.BatchNorm2d(C[4]), nn.GELU()
        )
        self.l4 = nn.Sequential(SwinBlock(C[4], num_heads=8), SwinBlock(C[4], num_heads=8))
        # Level 5: 8x8, 512ch (bottleneck)
        self.down5 = nn.Sequential(
            nn.Conv2d(C[4], C[5], 3, stride=2, padding=1), nn.BatchNorm2d(C[5]), nn.GELU()
        )
        self.l5 = nn.Sequential(SwinBlock(C[5], num_heads=8), SwinBlock(C[5], num_heads=8))

        # Calibration head: decode from 512ch 8x8 bottleneck to 256x256 binary map
        self.cal_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C[5], C[4], 3, padding=1), nn.ReLU(),   # 512->512
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C[4], C[3], 3, padding=1), nn.ReLU(),   # 512->256
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C[3], C[2], 3, padding=1), nn.ReLU(),   # 256->256
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C[2], C[1], 3, padding=1), nn.ReLU(),   # 256->128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C[1], C[0], 3, padding=1), nn.ReLU(),   # 128->128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(C[0], 1, 1), nn.Sigmoid(),
        )

        # Adapters: map HybridEncoder channels to UNet channels
        # unet_channels = [128, 128, 256, 256, 512, 512] = HybridEncoder.CHANNELS, so 1x1 is identity
        if unet_channels is None:
            unet_channels = C
        self.adapters = nn.ModuleList([
            nn.Conv2d(C[i], unet_channels[i], 1) for i in range(6)
        ])

    def forward(self, x):
        f0 = self.l0(x)                    # [B, 128, 256, 256]
        f1 = self.l1(self.down1(f0))       # [B, 128, 128, 128]
        f2 = self.l2(self.down2(f1))       # [B, 256, 64, 64]
        f3 = self.l3(self.down3(f2))       # [B, 256, 32, 32]
        f4 = self.l4(self.down4(f3))       # [B, 512, 16, 16]
        f5 = self.l5(self.down5(f4))       # [B, 512, 8, 8]

        # features[0]=L1(f1=128ch->adapter[1]), features[1]=L2(f2=256ch->adapter[2]), etc.
        features = [self.adapters[i+1](f) for i, f in enumerate([f1, f2, f3, f4])]
        cal = self.cal_up(f5)
        return features, cal
