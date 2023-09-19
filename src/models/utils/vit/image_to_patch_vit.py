import torch


# Function image to patch (input of transformer)
def image_to_patch(x, patch_size):
    # x: [B, C, H, W]
    # patch size P: size of a patch to transformer 

    B, C, H, W = x.shape
    # [B, C, H/P, P, W/P, P]
    x = x.reshape(
        B, 
        C, 
        torch.div(H, patch_size, rounding_mode='floor'), 
        patch_size, 
        torch.div(W, patch_size, rounding_mode='floor'),
        patch_size,
    )
    # [B, H/P, W/P, C, P, P]
    x = x.permute(0, 2, 4, 1, 3, 5)
    # [B, H * W / P^2 , C, P, P]
    x= x.flatten(1, 2)
    # [B, H * W / P^2, C * P^2] = [B, N, C * P^2]
    x = x.flatten(2, 4)

    return x



