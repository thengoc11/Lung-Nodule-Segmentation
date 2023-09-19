import torch.nn as nn

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_resolution = [img_size // patch_size, img_size // patch_size]
        self.embed_dim = embed_dim
        self.num_patches = self.patch_resolution[0] * self.patch_resolution[1]
                
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None 
        
        self.conv = nn.Conv2d(in_channels=in_chans,
                              out_channels=embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
    # x: (B, C, W, H)
    # B: batch size, C: in_chans, W: img_size, H: img_size, E: embed_dim, P: patch_size, N: num_patches
    def forward(self, x):
        # x: (B, C, W, H) -> (B, E, W/P, H/P)
        x = self.conv(x)
        
        # x_prj: (B, E, W/P, H/P) -> (B, E, W/P*H/P) -> (B, W/P*H/P, E) = (B, N, E)
        x = x.flatten(2).transpose(1, 2)
       
        # normalize
        if self.norm is not None:
            x = self.norm(x)

        return x

# import torch
# x = torch.randn(1, 3, 224, 224)
# patch_embed = PatchEmbed()
# y = patch_embed(x)
# print(y.shape)