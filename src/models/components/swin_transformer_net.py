import sys
sys.path.insert(0, "/home/dupham/Documents/self_learning/")
import torch.nn as nn
import torch
from src.models.utils.swin.patch_embed import PatchEmbed
from src.models.utils.swin.patch_merging import PatchMerging
from src.models.utils.swin.transformer_block_swin import SwinTransformerBlock
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class SwinTransformer(nn.Module):
    """
    Args:
        img_size (int): Input image size. Default 224
        patch_size (int): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part
    """
    
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False):
        
        super().__init__()
        
        if norm_layer == "norm_layer":
            norm_layer = nn.LayerNorm
        
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.ape = ape
        
        # patch embedding by convolution
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim,
                                      norm_layer=norm_layer if self.patch_norm else None)
        
        self.num_patches = self.patch_embed.num_patches
        self.patches_resolution = self.patch_embed.patch_resolution 
        
        # absolute position embedding
        if ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        # add block transformer
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SwinTransformerBlock(depth=depths[i_layer],
                                         downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                         use_checkpoint=use_checkpoint,
                                         embed_dim=self.embed_dim * 2 ** i_layer,
                                         input_resolution=(self.patches_resolution[0] // (2 ** i_layer), 
                                                           self.patches_resolution[1] // (2 ** i_layer)),
                                         num_heads=num_heads[i_layer],
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias,
                                         qk_scale=qk_scale,
                                         drop_rate=drop_rate,
                                         attn_drop=attn_drop_rate,
                                         drop_paths=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                         norm_layer=norm_layer,
                                         act_layer=nn.GELU,)
            self.layers.append(layer)
            
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
    
    # B: batch size, C: channel, H: height, W: width, E: embedding dim, N: number of patches
    def forward(self, x):
        # patch embedding: (B, C, H, W) -> (B, N, E)
        x = self.patch_embed(x)
        
        # add absolute position embedding
        if self.ape:
            x = x + self.absolute_pos_embed
        
        # add position dropout
        x = self.pos_drop(x)
        
        # forward transformer blocks
        for i, layer in enumerate(self.layers):
            x = layer(x)
        
        # add norm layer
        x = self.norm(x)
        
        # pooling
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        
        # classification head   
        x = self.head(x)
        return x
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            

# import torch
# x = torch.randn(2, 3, 64, 64)
# swin_transformer = SwinTransformer(img_size=64,
#                                     patch_size=4,
#                                     in_chans=3,
#                                     num_classes=2,
#                                     embed_dim=16,
#                                     depths=[2, 2, 6, 2],
#                                     num_heads=[2, 2, 2, 2],
#                                     window_size=2,
#                                     mlp_ratio=2,
#                                     qkv_bias=True,
#                                     qk_scale=None,
#                                     drop_rate=0,
#                                     drop_path_rate=0.1,
#                                     attn_drop_rate=0,
#                                     norm_layer=nn.LayerNorm,
#                                     ape=False,
#                                     patch_norm=True,
#                                     use_checkpoint=False)
# print(swin_transformer(x).shape)                              
                                         
            