"""
WMH-DiffSeg loss functions: v-prediction, Dice+BCE, cross-scale consistency.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_v_target(x_start, noise, sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t):
    """v = sqrt_alpha * noise - sqrt(1-alpha) * x_0"""
    return sqrt_alphas_cumprod_t * noise - sqrt_one_minus_alphas_cumprod_t * x_start


def predict_xstart_from_v(x_t, v, sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t):
    """x_0 = sqrt_alpha * x_t - sqrt(1-alpha) * v"""
    return sqrt_alphas_cumprod_t * x_t - sqrt_one_minus_alphas_cumprod_t * v


def dice_bce_loss(pred, target, smooth=1e-5):
    """Dice + BCE joint loss. pred/target in [0,1]."""
    pred = torch.sigmoid(pred)
    bce = F.binary_cross_entropy(pred, target.float(), reduction='mean')
    pred_flat = pred.reshape(-1)
    tgt_flat = target.float().reshape(-1)
    intersection = (pred_flat * tgt_flat).sum()
    dice = 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + tgt_flat.sum() + smooth)
    return bce + dice


def cross_scale_consistency_loss(coarse_pred, fine_pred):
    """Downsample fine to coarse resolution, then MSE."""
    h, w = coarse_pred.shape[-2], coarse_pred.shape[-1]
    fine_down = F.interpolate(fine_pred, size=(h, w), mode='bilinear', align_corners=False)
    return F.mse_loss(fine_down, coarse_pred.detach())
