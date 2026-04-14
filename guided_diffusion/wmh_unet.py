"""
WMH-DiffSeg main model: UNetModel_WMH
Integrates HybridEncoder + WaveletModulation + CoarseDiffusionHead.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wmh_encoder import HybridEncoder
from .wavelet_modulation import WaveletModulation
from .nn import (
    conv_nd, linear, zero_module, normalization,
    timestep_embedding, checkpoint,
)
from .unet import (
    TimestepEmbedSequential, ResBlock, AttentionBlock,
    Upsample, Downsample,
)
from .fp16_util import convert_module_to_f16, convert_module_to_f32


class CoarseDiffusionHead(nn.Module):
    """
    Lightweight 3-level UNet for 64x64 coarse segmentation.
    Input: [B, 4, 64, 64] (image_down 3ch + noisy_mask 1ch)
    Encoder features injected from HybridEncoder L2/L3/L4.
    Output: [B, 1, 64, 64] v-prediction
    """
    def __init__(self, model_channels=64, enc_channels=None):
        super().__init__()
        mc = model_channels
        # enc_channels: [highway_L2=128ch, highway_L3=256ch, highway_L4=512ch]
        if enc_channels is None:
            enc_channels = [128, 256, 512]

        time_embed_dim = mc * 4
        self.time_embed = nn.Sequential(
            linear(mc, time_embed_dim), nn.SiLU(), linear(time_embed_dim, time_embed_dim)
        )

        # encoder
        self.enc0 = TimestepEmbedSequential(conv_nd(2, 4, mc, 3, padding=1))
        self.enc1 = TimestepEmbedSequential(
            ResBlock(mc, time_embed_dim, 0, out_channels=mc * 2),
            Downsample(mc * 2, True),
        )
        self.enc2 = TimestepEmbedSequential(
            ResBlock(mc * 2, time_embed_dim, 0, out_channels=mc * 4),
            Downsample(mc * 4, True),
        )
        self.bottleneck = TimestepEmbedSequential(
            ResBlock(mc * 4, time_embed_dim, 0),
        )

        # encoder feature adapters (align HybridEncoder features to coarse head channels)
        # HybridEncoder features[1]=256ch(L2), [2]=256ch(L3), [3]=512ch(L4)
        # -> map to highway channels L2=128ch, L3=256ch, L4=512ch, then to mc*2, mc*4, mc*4
        self.enc_adapt = nn.ModuleList([
            nn.Conv2d(256, mc * 2, 1),   # L2: 256ch -> 128ch highway (for coarse head at 32x32)
            nn.Conv2d(256, mc * 4, 1),   # L3: 256ch -> 256ch highway (for coarse head at 16x16)
            nn.Conv2d(512, mc * 4, 1),   # L4: 512ch -> 512ch highway (for coarse head at 16x16)
        ])

        # decoder
        self.dec2 = TimestepEmbedSequential(
            ResBlock(mc * 4 + mc * 4, time_embed_dim, 0, out_channels=mc * 2),
            Upsample(mc * 2, True),
        )
        self.dec1 = TimestepEmbedSequential(
            ResBlock(mc * 2 + mc * 2, time_embed_dim, 0, out_channels=mc),
            Upsample(mc, True),
        )
        self.dec0 = TimestepEmbedSequential(
            ResBlock(mc + mc, time_embed_dim, 0, out_channels=mc),
        )
        self.out = nn.Sequential(
            normalization(mc), nn.SiLU(),
            conv_nd(2, mc, 1, 3, padding=1),
        )

    def forward(self, x, emb, enc_features):
        """
        x: [B, 4, 64, 64]
        emb: timestep embedding [B, time_embed_dim]
        enc_features: [f_l2, f_l3, f_l4] from HybridEncoder (downsampled to 64/32/16)
        """
        temb = self.time_embed(emb)

        h0 = self.enc0(x, temb)                                    # [B, mc, 64, 64]
        h1 = self.enc1(h0, temb)                                   # [B, mc*2, 32, 32]
        h1 = h1 + self.enc_adapt[0](
            F.interpolate(enc_features[0], size=h1.shape[-2:], mode='bilinear', align_corners=False)
        )
        h2 = self.enc2(h1, temb)                                   # [B, mc*4, 16, 16]
        h2 = h2 + self.enc_adapt[1](
            F.interpolate(enc_features[1], size=h2.shape[-2:], mode='bilinear', align_corners=False)
        )
        hb = self.bottleneck(h2, temb)                             # [B, mc*4, 16, 16]
        hb = hb + self.enc_adapt[2](
            F.interpolate(enc_features[2], size=hb.shape[-2:], mode='bilinear', align_corners=False)
        )

        d2 = self.dec2(torch.cat([hb, h2], dim=1), temb)          # [B, mc*2, 32, 32]
        d1 = self.dec1(torch.cat([d2, h1], dim=1), temb)          # [B, mc, 64, 64]
        d0 = self.dec0(torch.cat([d1, h0], dim=1), temb)          # [B, mc, 64, 64]
        return self.out(d0)                                         # [B, 1, 64, 64]


class UNetModel_WMH(nn.Module):
    """
    WMH-DiffSeg main model.
    - HybridEncoder replaces Generic_UNet highway
    - CoarseDiffusionHead for 64x64 coarse stage
    - WaveletModulation on skip connections
    - in_channels=5: image(3) + noisy_mask(1) + coarse_pred_up(1)
    Returns: ((fine_out, cal), coarse_pred)
    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        conv_resample=True,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels      # should be 5
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        # Compute UNet skip channel counts for encoder adapter sizing
        # channel_mult default (1,1,2,2,4,4) with model_channels=128 -> 128,128,256,256,512,512
        # hs[3], hs[6], hs[9], hs[12] correspond to levels 1,2,3,4 last blocks
        ch = model_channels
        input_block_chans = [ch]
        ds = 1
        skip_chans = []
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                ch = mult * model_channels
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                input_block_chans.append(ch)
                ds *= 2
        # pick every (num_res_blocks+1)-th entry as injection points
        # channel_mult=(1,1,2,2,4,4): levels 0-5
        # unet_skip_chans[i] = ch at input_block[2+3*i] = ch at level i
        unet_skip_chans = []
        idx = 0
        for level, mult in enumerate(channel_mult):
            idx += num_res_blocks
            unet_skip_chans.append(mult * model_channels)
            if level != len(channel_mult) - 1:
                idx += 1
        # keep all 6 levels; HybridEncoder provides 4 adapters (L1-L4 → levels 1-4)
        # levels 0 and 5 use direct skip (identity)
        unet_skip_chans = unet_skip_chans[:6]

        # Hybrid encoder (replaces highway)
        # HybridEncoder.CHANNELS = [128, 128, 256, 256, 512, 512]
        # unet_skip_chans = [128, 128, 256, 256, 512, 512] - direct match
        img_in_ch = in_channels - 2  # image channels only (no noisy mask, no coarse pred)
        self.encoder = HybridEncoder(in_ch=img_in_ch, unet_channels=unet_skip_chans)

        # Coarse diffusion head: enc_channels = [highway_L2, highway_L3, highway_L4]
        # With new HybridEncoder: features = [f1(128), f2(256), f3(256), f4(512)]
        # Highway channels: L2=128, L3=256, L4=512
        self.coarse_head = CoarseDiffusionHead(
            model_channels=64,
            enc_channels=[128, 256, 512],
        )
        self.coarse_time_proj = linear(time_embed_dim, 64)  # project to coarse head time embed input (mc=64, not model_channels)

        # Main UNet input blocks
        ch = model_channels
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(2, in_channels, model_channels, 3, padding=1))
        ])
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        self._enc_inject_indices = []  # indices in input_blocks where encoder features are injected

        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [ResBlock(
                    ch, time_embed_dim, dropout,
                    out_channels=mult * model_channels,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(
                        ch, use_checkpoint=use_checkpoint,
                        num_heads=num_heads, num_head_channels=num_head_channels,
                        use_new_attention_order=use_new_attention_order,
                    ))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(TimestepEmbedSequential(
                    Downsample(ch, conv_resample, dims=2, out_channels=ch)
                ))
                input_block_chans.append(ch)
                ds *= 2

        # Encoder feature injection: 1x1 conv to align channels, injected additively
        # Inject at levels 1-4 (skip levels 0 and 5)
        inject_positions = []
        pos = 0
        for level, mult in enumerate(channel_mult):
            pos += num_res_blocks
            inject_positions.append(pos)
            if level != len(channel_mult) - 1:
                pos += 1
        # inject at levels 1-4 (indices 1-4 in inject_positions)
        self._inject_positions = inject_positions[1:5]  # [5, 8, 11, 14] for levels 1-4

        # 6 adapters for all 6 levels; levels 0 and 5 are identity (not injected)
        # enc_inject_convs maps HybridEncoder output channels to UNet hidden state channels
        # Using identity since h and ef channels already match at each injection point
        self.enc_inject_convs = nn.ModuleList([
            nn.Identity(),  # level 0: not used
            nn.Identity(),  # level 1 (inj_idx=0): h=128ch, ef=128ch
            nn.Identity(),  # level 2 (inj_idx=1): h=256ch, ef=256ch
            nn.Identity(),  # level 3 (inj_idx=2): h=256ch, ef=256ch
            nn.Identity(),  # level 4 (inj_idx=3): h=512ch, ef=512ch
            nn.Identity(),  # level 5: not used
        ])

        # Middle block
        self.middle_block = TimestepEmbedSequential(
            ResBlock(ch, time_embed_dim, dropout, use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
            AttentionBlock(ch, use_checkpoint=use_checkpoint, num_heads=num_heads,
                           num_head_channels=num_head_channels,
                           use_new_attention_order=use_new_attention_order),
            ResBlock(ch, time_embed_dim, dropout, use_checkpoint=use_checkpoint,
                     use_scale_shift_norm=use_scale_shift_norm),
        )

        # Output blocks with wavelet modulation on skip connections
        self.output_blocks = nn.ModuleList([])
        self.wavelet_modules = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                self.wavelet_modules.append(WaveletModulation(ich))
                layers = [ResBlock(
                    ch + ich, time_embed_dim, dropout,
                    out_channels=model_channels * mult,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(
                        ch, use_checkpoint=use_checkpoint,
                        num_heads=num_heads_upsample, num_head_channels=num_head_channels,
                        use_new_attention_order=use_new_attention_order,
                    ))
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=2))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch), nn.SiLU(),
            conv_nd(2, model_channels, out_channels, 3, padding=1),
        )

    def convert_to_fp16(self):
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def load_part_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)

    def forward(self, x, timesteps, y=None):
        """
        x: [B, 5, H, W]  (image 3ch + noisy_mask 1ch + coarse_pred 1ch)
           On first call coarse_pred channel is zeros; model fills it internally.
        Returns: ((fine_out, cal), coarse_pred)
        """
        B = x.shape[0]
        image = x[:, :self.in_channels - 2, ...]   # [B, 3, H, W]
        noisy_mask = x[:, -2:-1, ...]               # [B, 1, H, W]

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        # --- Hybrid encoder ---
        enc_features, cal = self.encoder(image)     # enc_features: 4 levels

        # --- Coarse stage ---
        img_small = F.interpolate(image, scale_factor=0.25, mode='bilinear', align_corners=False)
        mask_small = F.interpolate(noisy_mask, scale_factor=0.25, mode='bilinear', align_corners=False)
        coarse_in = torch.cat([img_small, mask_small], dim=1)   # [B, 4, H/4, W/4]
        coarse_emb = self.coarse_time_proj(emb)
        coarse_pred = self.coarse_head(coarse_in, coarse_emb, enc_features[1:4])  # [B, 1, H/4, W/4]

        # Upsample coarse pred and concat to input
        coarse_up = F.interpolate(coarse_pred, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x_fine = torch.cat([image, noisy_mask, coarse_up], dim=1)  # [B, 5, H, W]

        # --- Main UNet ---
        hs = []
        h = x_fine.type(self.dtype)
        inject_idx = 0

        for ind, module in enumerate(self.input_blocks):
            h = module(h, emb)
            hs.append(h)
            # inject encoder features at designated positions
            if inject_idx < len(self._inject_positions) and ind == self._inject_positions[inject_idx]:
                ef = enc_features[inject_idx]
                ef_resized = F.interpolate(ef, size=h.shape[-2:], mode='bilinear', align_corners=False)
                h = h + self.enc_inject_convs[inject_idx](ef_resized)
                hs[-1] = h  # update stored skip
                inject_idx += 1

        h = self.middle_block(h, emb)

        for i, module in enumerate(self.output_blocks):
            skip = hs.pop()
            skip = self.wavelet_modules[i](skip)    # wavelet modulation
            h = torch.cat([h, skip], dim=1)
            h = module(h, emb)

        h = h.type(x.dtype)
        fine_out = self.out(h)                      # [B, out_channels, H, W]

        return (fine_out, cal), coarse_pred
