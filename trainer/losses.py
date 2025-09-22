import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.visualizations import render_srgb_preview
from utils.helpers import spectral_total_variation

def l1_loss(pred, target, mask=None):
    if mask is None:
        return F.l1_loss(pred, target)
    m = mask[:, None, :, :].float()   # (N,1,H,W)
    num = torch.clamp(m.sum() * pred.shape[1], min=1.0)
    return (torch.abs(pred - target) * m).sum() / num

def sam_loss(pred, target, eps=1e-8, mask=None):
    # pred/target: (N,C,H,W)
    N,C,H,W = pred.shape
    p = pred.permute(0,2,3,1).reshape(-1, C)
    t = target.permute(0,2,3,1).reshape(-1, C)
    if mask is not None:
        m = mask.reshape(-1).bool()
        p = p[m]; t = t[m]
    p = p + eps; t = t + eps
    num = (p * t).sum(dim=1)
    den = torch.norm(p, dim=1) * torch.norm(t, dim=1) + eps
    cos = torch.clamp(num / den, -1 + 1e-7, 1 - 1e-7)
    ang = torch.acos(cos)  # radians
    return ang.mean()

class ReconLoss(nn.Module):
    def __init__(self, lambda_sam=0.1):
        super().__init__()
        self.lambda_sam = lambda_sam
    def forward(self, pred, target, mask=None):
        return l1_loss(pred, target, mask) + self.lambda_sam * sam_loss(pred, target, mask=mask)


class ReconLossWithColorSmooth(nn.Module):
    """
    L = L1(pred, target) + λ_sam*SAM + λ_col*L1(sRGB(pred), sRGB(target)) + λ_tv*SpectralTV
    """
    def __init__(self, wl_nm, lambda_sam=0.1, lambda_color=0.2, lambda_tv=0.005):
        super().__init__()
        self.lambda_sam = float(lambda_sam)
        self.lambda_color = float(lambda_color)
        self.lambda_tv = float(lambda_tv)
        # store as numpy for renderer
        import numpy as np
        self.wl_nm = np.asarray(wl_nm, dtype=float)

    def _srgb(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,C,H,W) in [0,1]
        import numpy as np
        outs = []
        for i in range(x.size(0)):
            rgb = render_srgb_preview(x[i].detach().cpu().numpy(), self.wl_nm, show_fig=False)
            outs.append(torch.from_numpy(rgb).permute(2,0,1))  # -> CHW
        return torch.stack(outs, dim=0).to(x.device)

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask=None):
        base = l1_loss(pred, target, mask) + self.lambda_sam * sam_loss(pred, target, mask=mask)
        # Color term on small random crops to reduce CPU cost
        N, C, H, W = pred.shape
        r0 = 0
        c0 = 0
        hh = min(H, 128)
        ww = min(W, 128)
        pred_crop = pred[:, :, r0:r0+hh, c0:c0+ww]
        tgt_crop  = target[:, :, r0:r0+hh, c0:c0+ww]
        srgb_pred = self._srgb(pred_crop)
        srgb_tgt  = self._srgb(tgt_crop)
        color_l1 = F.l1_loss(srgb_pred, srgb_tgt)
        tv = spectral_total_variation(pred)
        return base + self.lambda_color * color_l1 + self.lambda_tv * tv
