# Track 1
from baselines.raw2hsi import Raw2HSI
print(sum(p.numel() for p in Raw2HSI(base_ch=64, n_blocks=8, out_bands=61).parameters()))

# Track 2 (example)
from baselines.mstpp_up import MST_Plus_Plus_LateUpsample
print(sum(p.numel() for p in MST_Plus_Plus_LateUpsample(in_channels=3, out_channels=61, n_feat=61, stage=3, upscale_factor=2).parameters()))
