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

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
    
    def forward(self, x, mask=None):
        attn_out = self.attn(self.ln1(x), mask=mask)
        x = x + attn_out
        mlp_out = self.mlp(self.ln2(x))
        x = x + mlp_out
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
        
        # Use custom TransformerBlock modules that accept a mask.
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        
        self.final_ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_embeddings)
    
    def forward(self, indices_list, current_level, mask=None):
        """
        indices_list: list of token indices at different resolutions
        current_level: which resolution we're currently training
        """
        # Embed all available tokens (each token tensor might be a 2D grid; flatten if needed)
        embeddings = []
        for level, indices in enumerate(indices_list[:current_level + 1]):
            # If indices have spatial dimensions, flatten to shape [B, num_tokens]
            if indices.dim() > 2:
                indices = indices.view(indices.size(0), -1)
            # Token embedding
            x = self.token_embedding(indices)
            
            # Add positional embedding for this resolution
            patch_size = self.patch_sizes[level]
            x = x + self.pos_embeddings[str(patch_size)][:, :x.size(1)]
            
            # Add level embedding
            x = x + self.level_embedding[level]
            
            embeddings.append(x)
        
        # Concatenate all embeddings along the sequence dimension.
        x = torch.cat(embeddings, dim=1)
        
        # Create a causal attention mask if none is provided.
        if mask is None:
            mask = torch.triu(torch.ones(x.size(1), x.size(1)), diagonal=1).bool().to(x.device)
        
        # Apply transformer layers with mask.
        for layer in self.transformer_layers:
            x = layer(x, mask=mask)
        
        x = self.final_ln(x)
        logits = self.head(x)
        
        return logits
