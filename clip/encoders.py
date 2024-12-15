import torch.nn as nn

from transformers import ViTConfig, ViTModel
from torchvision.models import resnet50, ResNet50_Weights


class CLIPViTEncoder(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=32,
        hidden_size=512,
        num_hidden_layers=12,
        num_attention_heads=8,
        projection_dim=512
    ):
        super().__init__()
        
        # Configure ViT
        self.config = ViTConfig(
            image_size=image_size,
            patch_size=patch_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_channels=3,
            qkv_bias=True,
            layer_norm_eps=1e-6
        )
        
        # Initialize ViT backbone
        self.vit = ViTModel(self.config)
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection_dim)
        )

    def forward(self, pixel_values):
        # Get ViT outputs
        outputs = self.vit(pixel_values)
        pooled_output = outputs.pooler_output
        
        # Project to final dimension
        projected = self.projection(pooled_output)
        
        # Normalize embeddings
        image_features = projected / projected.norm(dim=-1, keepdim=True)
        
        return image_features


class CLIPResNetEncoder(nn.Module):
    def __init__(self, projection_dim=512):
        super().__init__()
        
        # Load pretrained ResNet
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # Remove final classification layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        
        # Add projection layers
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, projection_dim),
            nn.LayerNorm(projection_dim)
        )
        
    def forward(self, x):
        # Get ResNet features
        features = self.resnet(x)
        features = features.squeeze(-1).squeeze(-1)
        
        # Project to final dimension
        projected = self.projection(features)
        
        # Normalize embeddings
        image_features = projected / projected.norm(dim=-1, keepdim=True)
        
        return image_features