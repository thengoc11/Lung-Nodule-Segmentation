import torch
import torch.nn as nn
from src.models.utils.vit.image_to_patch_vit import image_to_patch
from src.models.utils.vit.transformer_block_vit import TransformerBlock

class VisionTransformer(nn.Module):
    def __init__(
        self,
        embbed_dim,
        hidden_dim,
        num_channels,
        num_heads,
        num_layers,
        num_classes,
        patch_size,
        num_patches,
        dropout=0.0
    ):
        super().__init__()
        self.patch_size = patch_size

        self.input_layer = nn.Linear(num_channels * patch_size * patch_size, embbed_dim)
        self.transformer = nn.Sequential(
            *(TransformerBlock(embbed_dim, hidden_dim, num_heads, dropout=dropout)
              for _ in range(num_layers))
        )
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embbed_dim),
            nn.Linear(embbed_dim, num_classes)
        )
        self.dropout = nn.Dropout(dropout)
        
        self.cls_token = nn.Parameter(
            torch.rand(1, 1, embbed_dim)
        )
        self.pos_embbeding = nn.Parameter(
            torch.rand(1, 1 + num_patches, embbed_dim)
        )


    def forward(self, x):
        x = image_to_patch(x, self.patch_size)

        B, N, num_patches = x.shape
        x = self.input_layer(x)

        cls_token = self.cls_token.repeat(B, 1, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embbeding[:, : N + 1]

        x = self.dropout(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)

        cls = x[0]
        out_put = self.mlp_head(cls)
        return out_put 