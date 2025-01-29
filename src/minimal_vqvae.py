import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, latent_dim, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)

class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=32, out_channels=3):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return torch.tanh(self.conv3(x))

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=32):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, z):
        # Reshape z -> (batch, height, width, channel)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, z.shape[-1])
        
        # Calculate distances to embeddings
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        # Find nearest embedding
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # Reshape back
        z_q = z_q.permute(0, 3, 1, 2)
        
        # Use straight-through estimator
        z_q = z + (z_q - z).detach()
        return z_q, min_encoding_indices

# Add patch-based encoding to VQVAE
class SimpleVQVAE(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=32):
        super().__init__()
        self.encoder = SimpleEncoder(latent_dim=embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = SimpleDecoder(latent_dim=embedding_dim)
        
    def encode_to_patches(self, x, patch_size):
        """Encode image into patches of different sizes"""
        B, C, H, W = x.shape
        # Ensure H and W are divisible by patch_size
        assert H % patch_size == 0 and W % patch_size == 0
        
        # Reshape into patches
        patches = x.view(B, C, H//patch_size, patch_size, W//patch_size, patch_size)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(B, (H//patch_size)*(W//patch_size), C*patch_size*patch_size)
        
        return patches

    def decode_from_patches(self, patches, patch_size, H, W):
        """Decode patches back to image"""
        B = patches.size(0)
        patches = patches.view(B, H//patch_size, W//patch_size, -1, patch_size, patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patches.view(B, -1, H, W)