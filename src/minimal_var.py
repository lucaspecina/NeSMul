import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        B, L, C = x.shape
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        return x

class ProgressiveVAR(nn.Module):
    def __init__(self, num_embeddings=1024, embed_dim=256, num_heads=8, num_layers=6):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings, embed_dim)
        
        # Define patch sizes for progressive generation
        self.patch_sizes = [16, 8, 4, 2]  # From coarse to fine
        
        # Position embeddings for each resolution
        self.pos_embeddings = nn.ParameterDict({
            str(size): nn.Parameter(torch.randn(1, (256//size)**2, embed_dim) * 0.02)
            for size in self.patch_sizes
        })
        
        # Level embeddings to distinguish different resolutions
        self.level_embedding = nn.Parameter(torch.randn(len(self.patch_sizes), 1, embed_dim) * 0.02)
        
        self.transformer_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                MultiHeadAttention(embed_dim, num_heads),
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim * 4),
                nn.GELU(),
                nn.Linear(embed_dim * 4, embed_dim),
            ) for _ in range(num_layers)
        ])
        
        self.final_ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_embeddings)
    
    def forward(self, indices_list, current_level, mask=None):
        """
        indices_list: list of token indices at different resolutions
        current_level: which resolution we're currently training
        """
        batch_size = indices_list[0].size(0)
        
        # Embed all available tokens
        embeddings = []
        position = 0
        for level, indices in enumerate(indices_list[:current_level + 1]):
            # Token embedding
            x = self.token_embedding(indices)
            
            # Add position embedding for this resolution
            patch_size = self.patch_sizes[level]
            x = x + self.pos_embeddings[str(patch_size)][:, :x.size(1)]
            
            # Add level embedding
            x = x + self.level_embedding[level]
            
            embeddings.append(x)
            position += x.size(1)
        
        # Concatenate all embeddings
        x = torch.cat(embeddings, dim=1)
        
        # Create causal attention mask
        if mask is None:
            mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool()
            mask = mask.to(x.device)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = x + layer(x)
        
        x = self.final_ln(x)
        logits = self.head(x)
        
        return logits
