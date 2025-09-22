import torch
import torch.nn as nn


class LinearRaw2HSI(nn.Module):
    """
    Ultra-simple linear mapper for Track 1 (mosaic -> HSI cube).

    Pipeline:
      - PixelUnshuffle(2): (N,1,H,W) -> (N,4,H/2,W/2)
      - 1x1 Conv: 4 -> (out_bands*4) in packed space
      - PixelShuffle(2): (N,out_bands*4,H/2,W/2) -> (N,out_bands,H,W)

    This mirrors a per-Bayer-tile linear regression to spectrum, shared across spatial locations.
    """

    def __init__(self, out_bands: int = 61, bias: bool = True):
        super().__init__()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.linear_1x1 = nn.Conv2d(4, out_bands * 4, kernel_size=1, stride=1, padding=0, bias=bias)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, mosaic: torch.Tensor) -> torch.Tensor:
        x = self.unshuffle(mosaic)              # (N,4,H/2,W/2)
        y = self.linear_1x1(x)                  # (N,out_bands*4,H/2,W/2)
        out = self.shuffle(y)                   # (N,out_bands,H,W)
        out = torch.clamp(out, 0.0, 1.0)
        return out

