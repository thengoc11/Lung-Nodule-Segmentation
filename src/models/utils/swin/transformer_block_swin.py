import sys
sys.path.insert(0, "/home/dupham/Documents/self_learning/")
import torch.nn as nn
from src.models.utils.swin.transformer_layer_swin import SwinTransformerLayer
import torch.utils.checkpoint as checkpoint

class SwinTransformerBlock(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """
    
    def __init__(self,
                 depth,
                 downsample,
                 use_checkpoint,
                 embed_dim, 
                 input_resolution, 
                 num_heads, 
                 window_size,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop = 0.0,
                 drop_paths = 0.0,
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        
        self.blocks = nn.ModuleList([
            SwinTransformerLayer(embed_dim=embed_dim,
                                 input_resolution=input_resolution,
                                 num_heads=num_heads,
                                 window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop_rate=drop_rate,
                                 attn_drop=attn_drop,
                                 drop_path=drop_paths[i],
                                 act_layer=act_layer,
                                 norm_layer=norm_layer)
        for i in range(depth)])
        
        if downsample is not None:
            self.downsample = downsample(input_resolution, embed_dim, norm_layer)
        else:
            self.downsample = None
    
    # x: (B, N, E)
    # output: (B, N/4, E*2)
    def forward(self, x):
        for block in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(block, x)
            else:
                x = block(x)
                
        if self.downsample is not None:
            x = self.downsample(x)
        return x

# from src.models.utils.swin.patch_merging import PatchMerging
# import torch
# x = torch.randn(2, 16 * 16, 4)
# y = SwinTransformerBlock(2, PatchMerging, True, 4, (16, 16), 2, 8, mlp_ratio=4.0, qkv_bias=False, qk_scale=None, drop_rate=0.0, drop_paths=[0.0, 0.0])(x)
# print(y.shape)