import torch.nn as nn
import torch
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class WindowMultipleSelfAttention(nn.Module):
    """ Window based multi-head self attention (W-MSA) module with relative position bias.
        Supports both of shifted and non-shifted window.
    Args:
        embed_dim (int): Number of embedding dimensions before the attention layer.
        window_size (int): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, embed_dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0):
        super().__init__()

        self.embed_dim = embed_dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # if qk_scale is None, use sqrt(embed_dim)
        self.scale = qk_scale or embed_dim ** -0.5

        # Table: [(2 * Ww - 1) * (2 * Wh - 1), nH]
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1), num_heads))
        
        # get pair-wise relative position index for each token inside the window
        # relative_position_index: [Wh * Ww, Wh * Ww]
        relative_position_index = self.get_relative_position_bias() 
        self.register_buffer("relative_position_index", relative_position_index)
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    # input:
    ## x: B, N, C
    ## mask: B, N, N
    # output:
    ## x (W-MSA): B, N, C
    def forward(self, x, mask=None):
        # B: batch size, N: number of patches, C: embedding dimension, nH: number of heads, nW: number of windows
        B, N, C = x.shape
        # B, N, C -> B, N, 3C -> B, N, 3, nH, C/nH -> 3, B, nH, N, C/nH
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        # Q, K, V of attention: B, nH, N, C/nH
        q, k, v = qkv[0], qkv[1], qkv[2]

        # scale factor
        q = q * self.scale

        # calculate attention score from query and key
        # attn: B, nH, N, N. 
        attn = (q @ k.transpose(-2, -1))
        
        # take relative position bias from pre-computed table
        # relative_position_bias: B, Wh * Ww * Wh * Ww, nH => B, Wh*Ww, Wh*Ww, nH
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)
        # B, nH, Wh*Ww, Wh*Ww = B, nH, N, N
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  

        # add relative position bias to attn
        attn = attn + relative_position_bias.unsqueeze(0)

        
        if mask is not None:
            # nW: number of windows
            nW = mask.shape[0]
            
            # attn: B, nH, N, N -> B / nW, nW, nH, N, N
            # mask: B, N, N -> B / nW, nW, 1, N, N
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            
            # attn: B / nW, nW, nH, N, N -> B, nH, N, N
            attn = attn.view(-1, self.num_heads, N, N)
            
            # softmax
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)
        
        # dropout
        attn = self.attn_drop(attn)

        # calculate attention output from attention score and value
        # x: B, nH, N, C/nH -> B, N, nH, C/nH -> B, N, C
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # projection and dropout
        # x: B, N, C -> B, N, C
        x = self.projection(x)
        x = self.proj_drop(x)

        return x

    # TODO: add comments
    def get_relative_position_bias(self):
        # Wh: window height, Ww: window width
        
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
    
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        
        return relative_position_index
    
# x = torch.randn(1, 16, 256)
# window_attention = WindowMultipleSelfAttention(256, 4, 4, 4)
# y = window_attention(x)
# print(y.shape)
