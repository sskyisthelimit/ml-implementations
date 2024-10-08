import torch
import torch.nn as nn
from config import (EMBED_DIM, PATCH_SIZE, NUM_PATCHES, DROPOUT, IN_CHANNELS,
                    device, NUM_CLASSES, IMG_SIZE, NUM_ENCODERS, NUM_HEADS,
                    HIDDEN_DIM, ACTIVATION)
import warnings


class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size,
                 num_patches, dropout, in_channels):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=embed_dim,
                kernel_size=patch_size,
                stride=patch_size,
            ), 
            nn.Flatten(2)
            )

        self.cls_token = nn.Parameter(
            torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.position_embeddings = nn.Parameter(
            torch.randn(size=(1, num_patches+1, embed_dim)),
            requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.position_embeddings + x 
        x = self.dropout(x)
        return x
    

class ViT(nn.Module):
    def __init__(self, num_patches, img_size, num_classes,
                 patch_size, embed_dim, num_encoders,
                 num_heads, hidden_dim, dropout, activation, in_channels):
        super().__init__()
        self.embeddings_block = PatchEmbedding(
            embed_dim, patch_size,
            num_patches, dropout, in_channels)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dropout=dropout,
            activation=activation, batch_first=True, norm_first=True)
        
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer,
                                                    num_layers=num_encoders)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :])  # Apply MLP on the CLS token only
        return x


if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    model = ViT(NUM_PATCHES, IMG_SIZE, NUM_CLASSES, PATCH_SIZE, EMBED_DIM,
                NUM_ENCODERS, NUM_HEADS, HIDDEN_DIM, DROPOUT,
                ACTIVATION, IN_CHANNELS).to(device)
    
    x = torch.randn(512, 1, 28, 28).to(device)
    print(model(x).shape)  # BATCH_SIZE X NUM_CLASSES