import torch.nn as nn
import torch

class PatchMerging(nn.Module):
    """ Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, input_resolution, new_embed_dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.new_embed_dim = new_embed_dim
        self.norm = norm_layer(4 * new_embed_dim)
        self.reduction = nn.Linear(4 * new_embed_dim, 2 * new_embed_dim, bias=False)
    
    
    # B: batch size, N: num_patches, C: embed_dim, H: input_resolution[0], W: input_resolution[1]
    # x: (B, N, C) 
    # output: (B, N/4, C * 2)
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        
        # view x from (B, N, E) (B, H, W, C)
        x = x.view(B, H, W, C)
        
        x00 = x[:, 0::2, 0::2, :] # (B, H/2, W/2, C)
        x10 = x[:, 1::2, 0::2, :] # (B, H/2, W/2, C)
        x01 = x[:, 0::2, 1::2, :] # (B, H/2, W/2, C)
        x11 = x[:, 1::2, 1::2, :] # (B, H/2, W/2, C)
        x = torch.cat([x00, x10, x01, x11], dim=-1) # (B, H/2, W/2, 4*C)
        x = x.view(B, -1, 4 * C) # (B, H/2*W/2, 4*C)
        
        # x: (B, H/2*W/2, 4*C) -> (B, H/2*W/2, 2*C)
        x = self.reduction(x)
        x = self.norm(x)
        
        
        return x
        
# import torch
# x = torch.randn(1, 196, 768)
# y = PatchMerging((14, 14), 768)(x)
# print(y.shape)
