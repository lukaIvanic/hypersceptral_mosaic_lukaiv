import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride=stride)


def shift_back(inputs,step=2):          # input [bs,28,256,310]  output [bs, 28, 256, 256]
    [bs, nC, row, col] = inputs.shape
    down_sample = 256//row
    step = float(step)/float(down_sample*down_sample)
    out_col = row
    for i in range(nC):
        inputs[:,i,:,:out_col] = \
            inputs[:,i,:,int(step*i):int(step*i)+out_col]
    return inputs[:, :, :, :out_col]

class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        
        q_inp = self.to_q(x_in)
        k_inp = self.to_k(x_in)
        v_inp = self.to_v(x_in)
        
        q, k, v = map(lambda t: rearrange(t, 'b h w (heads c) -> b heads c (h w)', heads=self.num_heads),
                                (q_inp, k_inp, v_inp))
        
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        
        attn = (q @ k.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        
        x = attn @ v
        x = rearrange(x, 'b heads c (h w) -> b h w (heads c)', h=h, w=w)
        
        out_c = self.proj(x)
        out_p = self.pos_emb(v_inp.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

import torch.utils.checkpoint as checkpoint

class MSAB(nn.Module):
    """
    Multi-Spectral Attention Block (SAB in the paper).
    
    Paper Reference (Section 3.1):
    ─────────────────────────────
    "Based on this nature [HSI spatially sparse, spectrally self-similar], 
    we adopt the Spectral-wise Multi-head Self-Attention (S-MSA) to compose 
    the basic unit, Spectral-wise Attention Block (SAB)."
    
    Block Structure (standard Transformer block with residual):
    ─────────────────────────────────────────────────────────
    
        Input x
          │
          ├─────────────┐
          ▼             │
    ┌───────────┐       │
    │  MS_MSA   │       │  Spectral-wise Multi-head Self-Attention
    └───────────┘       │
          │             │
          + ◄───────────┘  Residual connection #1
          │
          ├─────────────┐
          ▼             │
    ┌───────────┐       │
    │ LayerNorm │       │  
    │    +      │       │  Pre-norm + Feed-Forward Network
    │    FFN    │       │  (1×1 conv → GELU → 3×3 depthwise → GELU → 1×1 conv)
    └───────────┘       │
          │             │
          + ◄───────────┘  Residual connection #2
          │
          ▼
       Output
    
    Multiple blocks are stacked (num_blocks parameter).
    
    Args:
        dim: Feature dimension
        dim_head: Dimension per attention head
        heads: Number of attention heads
        num_blocks: Number of SAB blocks to stack
        use_checkpoint: Enable gradient checkpointing for memory efficiency
    """
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            num_blocks,
            use_checkpoint=False
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.use_checkpoint = use_checkpoint
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                # Spectral-wise Multi-head Self-Attention
                MS_MSA(dim=dim, dim_head=dim_head, heads=heads),
                # Pre-norm + Feed-Forward Network
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        Forward pass through stacked SAB blocks.
        
        Args:
            x: [B, C, H, W] - channels first (standard PyTorch conv format)
        
        Returns:
            out: [B, C, H, W]
        """
        # Convert to channels-last for attention: [B, C, H, W] → [B, H, W, C]
        x = x.permute(0, 2, 3, 1)
        
        for (attn, ff) in self.blocks:
            if self.use_checkpoint and x.requires_grad:
                # Gradient checkpointing: trade compute for memory
                # Recomputes forward pass during backward instead of storing activations
                x = checkpoint.checkpoint(self._run_block, attn, ff, x)
            else:
                # Standard transformer block with residual connections
                x = attn(x) + x   # Self-attention + residual
                x = ff(x) + x     # FFN + residual
        
        # Convert back to channels-first: [B, H, W, C] → [B, C, H, W]
        out = x.permute(0, 3, 1, 2)
        return out

    def _run_block(self, attn, ff, x):
        """Helper for gradient checkpointing."""
        x = attn(x) + x
        x = ff(x) + x
        return x


class MST(nn.Module):
    """
    Single-stage Spectral-wise Transformer (SST) with U-shaped encoder-decoder.
    
    Paper Reference (Section 3.2):
    ─────────────────────────────
    "Our SABs build up our proposed Single-stage Spectral-wise Transformer (SST) 
    that exploits a U-shaped structure to extract multi-resolution spectral 
    contextual information which is critical for HSI restoration."
    
    Architecture (U-Net style):
    ──────────────────────────
    
    Input: [B, in_dim, H, W]
           │
           ▼
    ┌──────────────┐
    │  Embedding   │  3×3 Conv: in_dim → dim
    └──────────────┘
           │
           │ ENCODER PATH (resolution decreases, channels increase)
           ▼
    ┌──────────────┐
    │   MSAB #0    │  dim channels, H×W resolution
    │  + Downsample│  4×4 Conv stride 2: dim → 2×dim, H/2 × W/2
    └──────┬───────┘
           │ skip connection ─────────────────────┐
           ▼                                      │
    ┌──────────────┐                              │
    │   MSAB #1    │  2×dim channels, H/2×W/2     │
    │  + Downsample│  4×4 Conv: 2×dim → 4×dim     │
    └──────┬───────┘                              │
           │ skip connection ─────────┐           │
           ▼                          │           │
    ┌──────────────┐                  │           │
    │  Bottleneck  │  4×dim, H/4×W/4  │           │
    │    MSAB      │                  │           │
    └──────────────┘                  │           │
           │ DECODER PATH             │           │
           ▼                          │           │
    ┌──────────────┐                  │           │
    │   Upsample   │  TransConv: 4×dim → 2×dim    │
    │   + Fusion   │  Concat skip + 1×1 conv ◄────┘
    │   + MSAB     │                              │
    └──────────────┘                              │
           ▼                                      │
    ┌──────────────┐                              │
    │   Upsample   │  TransConv: 2×dim → dim      │
    │   + Fusion   │  Concat skip + 1×1 conv ◄────┘
    │   + MSAB     │
    └──────────────┘
           │
           ▼
    ┌──────────────┐
    │   Mapping    │  3×3 Conv: dim → out_dim
    └──────────────┘
           │
           + ◄───────── Input x (residual connection)
           │
           ▼
    Output: [B, out_dim, H, W]
    
    Args:
        in_dim: Input channels
        out_dim: Output channels  
        dim: Base channel dimension (doubles at each encoder stage)
        stage: Number of encoder/decoder levels (default 1 → 2× downsampling)
        num_blocks: List of MSAB blocks at each level [enc0, ..., bottleneck]
        use_checkpoint: Gradient checkpointing for memory efficiency
    
    Minimal Config (for fast testing):
        in_dim=8, out_dim=8, dim=8, stage=1, num_blocks=[1, 1]
        → ~50K parameters, processes 1024×1024 in seconds
    """
    def __init__(self, in_dim=8, out_dim=8, dim=8, stage=1, num_blocks=[1, 1], use_checkpoint=False):
        super(MST, self).__init__()
        self.dim = dim
        self.stage = stage

        # Input projection: project input channels to internal feature dimension
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # ENCODER: progressively downsample and increase channels
        # At each stage: MSAB for feature extraction, then 2× downsample
        self.encoder_layers = nn.ModuleList([])
        dim_stage = dim
        for i in range(stage):
            self.encoder_layers.append(nn.ModuleList([
                # MSAB: Spectral attention blocks
                # heads = dim_stage // dim → more heads at higher channels
                MSAB(dim=dim_stage, num_blocks=num_blocks[i], dim_head=dim, 
                     heads=dim_stage // dim, use_checkpoint=use_checkpoint),
                # Downsample: 4×4 conv with stride 2 (halves spatial, doubles channels)
                nn.Conv2d(dim_stage, dim_stage * 2, 4, 2, 1, bias=False),
            ]))
            dim_stage *= 2  # Double channels at each stage

        # BOTTLENECK: deepest layer with smallest spatial resolution
        # Has the most channels (dim × 2^stage)
        self.bottleneck = MSAB(
            dim=dim_stage, dim_head=dim, heads=dim_stage // dim, 
            num_blocks=num_blocks[-1], use_checkpoint=use_checkpoint)

        # DECODER: progressively upsample and decrease channels
        # Uses skip connections from encoder for detail preservation
        self.decoder_layers = nn.ModuleList([])
        for i in range(stage):
            self.decoder_layers.append(nn.ModuleList([
                # Upsample: 2× spatial increase, halve channels
                nn.ConvTranspose2d(dim_stage, dim_stage // 2, stride=2, kernel_size=2, 
                                   padding=0, output_padding=0),
                # Fusion: 1×1 conv to fuse upsampled + skip connection
                # Input: concatenated (dim_stage//2 + dim_stage//2) = dim_stage
                # Output: dim_stage // 2
                nn.Conv2d(dim_stage, dim_stage // 2, 1, 1, bias=False),
                # MSAB for feature refinement at this resolution
                MSAB(dim=dim_stage // 2, num_blocks=num_blocks[stage - 1 - i], 
                     dim_head=dim, heads=(dim_stage // 2) // dim, use_checkpoint=use_checkpoint),
            ]))
            dim_stage //= 2  # Halve channels at each decoder stage

        # Output projection: map back to output channels
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        # Activation function (used elsewhere, not in main forward)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass through U-shaped SST.
        
        Args:
            x: [B, in_dim, H, W]
        
        Returns:
            out: [B, out_dim, H, W] with residual connection to input
        """
        # Project input to feature space
        fea = self.embedding(x)  # [B, dim, H, W]

        # ENCODER: extract multi-scale features
        fea_encoder = []  # Store for skip connections
        for (MSAB, FeaDownSample) in self.encoder_layers:
            fea = MSAB(fea)                    # Spectral attention at this scale
            fea_encoder.append(fea)            # Save for skip connection
            fea = FeaDownSample(fea)           # Downsample: 2× spatial reduction

        # BOTTLENECK: process at lowest resolution
        fea = self.bottleneck(fea)

        # DECODER: reconstruct with skip connections
        for i, (FeaUpSample, Fusion, LeWinBlock) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)             # Upsample: 2× spatial increase
            # Concatenate with encoder features (skip connection) and fuse
            skip_idx = self.stage - 1 - i     # Reverse order for skip connections
            fea = Fusion(torch.cat([fea, fea_encoder[skip_idx]], dim=1))
            fea = LeWinBlock(fea)              # Refine with spectral attention

        # Map to output channels + RESIDUAL to input
        # This residual enables learning the "correction" to input
        out = self.mapping(fea) + x

        return out

class MST_Plus_Plus(nn.Module):
    """
    Multi-stage Spectral-wise Transformer (MST++) for Spectral Reconstruction.
    
    Paper Reference: "MST++: Multi-stage Spectral-wise Transformer for Efficient 
    Spectral Reconstruction" (CVPRW 2022, NTIRE Challenge Winner)
    
    Architecture Overview (Paper Section 3, Figure 2):
    ─────────────────────────────────────────────────
    MST++ cascades multiple Single-stage Spectral-wise Transformers (SSTs).
    Each SST progressively refines the reconstruction from coarse to fine.
    
        RGB Input [B, 3, H, W]
             │
             ▼
        ┌─────────────┐
        │   conv_in   │  Project: 3 → n_feat channels
        └─────────────┘
             │
             ▼
        ┌─────────────┐
        │   SST #1    │  (MST with U-shaped encoder-decoder)
        │   (MST)     │  Each SST has internal residual: out = mapping(fea) + x
        └─────────────┘
             │
             ▼
        ┌─────────────┐
        │   SST #2    │  Cascaded SSTs progressively refine features
        │   (MST)     │
        └─────────────┘
             │
             ▼
        ┌─────────────┐
        │   SST #3    │  (stage parameter controls number of SSTs)
        │   (MST)     │
        └─────────────┘
             │
         ┌───┴───┐
         │       │
         │   + ◄─┼──── x_feat (global residual from conv_in output)
         │       │
         └───┬───┘
             │
             ▼
        ┌─────────────┐
        │  conv_out   │  Project: n_feat → out_channels (e.g., 31 HSI bands)
        └─────────────┘
             │
             ▼
        HSI Output [B, 31, H, W]
    
    Complexity Analysis:
    ───────────────────
    The key innovation is Spectral-wise Multi-head Self-Attention (S-MSA) in MS_MSA.
    Standard spatial attention: O((H×W)² × C) - quadratic in spatial dimension
    Spectral-wise attention:    O(C² × H×W)   - LINEAR in spatial dimension H×W
    
    This is achieved by treating each spectral channel as a token and computing
    attention between channels (C×C matrix) rather than spatial locations.
    For HSI with C=31 channels and H×W=256×256 pixels:
      - Standard: O(65536² × 31) ≈ 133 trillion operations
      - Spectral: O(31² × 65536) ≈ 63 million operations (2000x reduction!)
    
    Args:
        in_channels (int): Number of input channels (default: 3 for RGB)
        out_channels (int): Number of output HSI spectral bands (default: 61)
        n_feat (int): Internal feature dimension (default: 8 for fast testing)
        stage (int): Number of cascaded SST stages (default: 1)
        use_checkpoint (bool): Enable gradient checkpointing for memory efficiency
    
    Minimal Config (for fast testing with 1024×1024 input):
        in_channels=3, out_channels=61, n_feat=8, stage=1
        → ~15K parameters total, very fast inference
    
    Production Config (for quality):
        in_channels=3, out_channels=61, n_feat=31, stage=3
        → ~1M+ parameters, better reconstruction quality
    """
    def __init__(self, in_channels=3, out_channels=61, n_feat=8, stage=1, use_checkpoint=False):
        super(MST_Plus_Plus, self).__init__()
        self.stage = stage
        
        # Input projection: RGB (3 channels) → n_feat feature channels
        # Uses 3×3 conv for local spatial context aggregation
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        
        # Body: Cascade of SSTs (Single-stage Spectral-wise Transformers)
        # Each MST module is one SST with U-shaped encoder-decoder structure
        # Paper: "MST++, cascaded by several SSTs, develops a multi-stage learning 
        # scheme to progressively improve the reconstruction quality from coarse to fine"
        #
        # MST parameters:
        #   - dim=n_feat: Base channel dimension for attention heads
        #   - in_dim/out_dim=n_feat: Input/output channels (internal feature space)
        #   - stage=1: Number of encoder/decoder levels in U-Net structure (minimal)
        #   - num_blocks=[1, 1]: Number of SABs at each resolution level (minimal)
        modules_body = [
            MST(dim=n_feat, in_dim=n_feat, out_dim=n_feat, stage=1, 
                num_blocks=[1, 1], use_checkpoint=use_checkpoint) 
            for _ in range(stage)
        ]
        self.body = nn.Sequential(*modules_body)
        
        # Output projection: n_feat → out_channels (spectral bands)
        # Maps learned features back to target HSI spectral dimension
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=(3 - 1) // 2, bias=False)

    def forward(self, x):
        """
        Forward pass for spectral reconstruction.
        
        Args:
            x: Input RGB image [B, in_channels, H, W]
        
        Returns:
            Reconstructed HSI [B, out_channels, H, W]
        
        Memory Complexity: O(B × n_feat × H × W) for feature maps
        Compute Complexity: O(B × stage × (C² × H × W)) - linear in H×W
        """
        b, c, h_inp, w_inp = x.shape
        
        # Padding to ensure H and W are divisible by 2^stage_mst
        # Required because the U-shaped MST uses 2× downsampling per stage
        # Default stage=1 in MST means 2× total, but we pad to 8 for flexibility
        # Note: 1024×1024 is already divisible by 8, so no padding needed
        hb, wb = 8, 8
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x_pad = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        # Project input to feature space
        # x_feat: [B, n_feat, H_padded, W_padded]
        x_feat = self.conv_in(x_pad)
        
        # Pass through cascaded SSTs (MST modules)
        # Each MST has its own internal residual: out = mapping(fea) + input
        # This creates progressive refinement: coarse → fine reconstruction
        h = self.body(x_feat)

        # Global residual connection on features (before final projection)
        # This helps gradient flow and enables learning residual corrections
        # Note: Each MST already has internal residual, this is an outer residual
        h = h + x_feat

        # Project features to output spectral bands
        # h: [B, out_channels, H_padded, W_padded]
        h = self.conv_out(h)
        
        # Remove padding and return original spatial dimensions
        return h[:, :, :h_inp, :w_inp]

