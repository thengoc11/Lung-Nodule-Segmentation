import sys
sys.path.insert(0, "/home/dupham/Documents/self_learning/")
import torch
import torch.nn as nn
from src.models.utils.swin.mlp import MLP
from src.models.utils.swin.window_attention_v2 import WindowMultipleSelfAttention
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class SwinTransformerLayer(nn.Module):
    """ Swin Transformer Block.
    Args:
        embed_dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, 
                 embed_dim, 
                 input_resolution, 
                 num_heads, 
                 window_size=7,
                 shift_size=0,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_rate=0.0,
                 attn_drop = 0.0,
                 drop_path = 0.0,
                 act_layer = nn.GELU,
                 norm_layer = nn.LayerNorm
                 ):
        super().__init__()

        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.shift_size = shift_size
        
        # normalization
        self.norm1 = norm_layer(embed_dim)

        # attention
        self.attn = WindowMultipleSelfAttention(embed_dim=embed_dim,
                                                window_size=window_size,
                                                num_heads=num_heads,
                                                qkv_bias=qkv_bias,
                                                qk_scale=qk_scale,
                                                attn_drop=attn_drop,
                                                proj_drop=drop_rate)
        
        # drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # normalization
        self.norm2 = norm_layer(embed_dim)

        # multi-layer perceptron
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(in_channels=embed_dim, 
                       hidden_channels=mlp_hidden_dim, 
                       act_layer=act_layer, 
                       dropout=drop_rate)
        
        if min(self.input_resolution) < self.window_size:
            # set shift_size as 0 means don't partition and shift windows
            self.shift_size = 0 
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0 - window_size"

        if self.shift_size > 0:
            H, W = self.input_resolution
            
            # generate attention mask for SW-MSA
            img_mask = torch.zeros((1, H, W, 1))
            slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
            count = 0
            for h in slices:
                for w in slices: 
                    img_mask[:, h, w, :] = count 
                    count += 1
            
            # partition windows
            # 1, H, W, 1 -> nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask, self.window_size)
            
            # nW, window_size, window_size, 1 -> nW, window_size * window_size
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            
            # mask_windows.unsqueeze(1): nW, 1, window_size * window_size
            # mask_windows.unsqueeze(2): nW, window_size * window_size, 1
            # attn_mask: nW, window_size * window_size, window_size * window_size
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            
            # change value of attn_mask from != 0 to -Inf and == 0 to 0
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None 
        
        # register buffer for attn_mask
        self.register_buffer("attn_mask", attn_mask)


    def forward(self, x):
        print("x.shape", x.shape)
        # H is height, W is width
        H, W = self.input_resolution
        # B is batch size, N is number of patches, C is embedding dimension
        B, N, C = x.shape
        assert N == H * W, "input feature has wrong size"

        # shortcut for residual connection
        shortcut = x

        # view B, N, C -> B, H, W, C
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            x_shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) 
        else:
            x_shifted = x
        
        # window partition
        # B, H, W, C -> B, nW, window_size, window_size, C
        x_windows = self.window_partition(x_shifted, self.window_size)

        # flatten window_size
        # B, nW, window_size, window_size, C -> nW * B, window_size * window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # unflatten window_size 
        # nW * B, window_size * window_size, C -> B, nW, window_size, window_size, C
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # window reverse
        # B, nW, window_size, window_size, C -> B, H, W, C
        x_shifted = self.window_reverse(attn_windows, self.window_size, H, W)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x_shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = x_shifted
        
        # view B, H, W, C -> B, N, C
        x = x.view(B, N, C)

        # residual connection
        ## different from Swin and Swinv2
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN (MLP)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

    def window_partition(self, x, window_size):
        # x: [B, H, W, C] -> windows: [B * nW, window_size, window_size, C]
        B, H, W, C = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)    
        return windows 
    
    def window_reverse(self, windows, window_size, H, W):
        # windows: [B * nW, window_size, window_size, C] -> x: [B, H, W, C]
        nW = H * W / window_size / window_size
        B = int(windows.shape[0] / nW)
        x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x 
    

# block = SwinTransformerLayer(4, (16, 16), 2, 8, 4, mlp_ratio=4, qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm)
# patches = torch.zeros((1, 16 * 16, 4))
# print(block(patches).shape)