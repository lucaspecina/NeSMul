import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEncoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, latent_dim, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return self.conv3(x)

class SimpleDecoder(nn.Module):
    def __init__(self, latent_dim=256, out_channels=3):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(latent_dim, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return torch.tanh(self.conv3(x))

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=256):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
        
    def forward(self, z):
        # Save original shape
        orig_shape = z.shape
        
        # Reshape z -> (batch, height * width, channel)
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, z.shape[-1])
        
        # Calculate distances to embeddings
        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.embedding.weight**2, dim=1) - \
            2 * torch.matmul(z_flattened, self.embedding.weight.t())
        
        # Find nearest embedding
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)
        
        # Reshape back to original shape
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        
        # Make sure z_q has the same shape as input
        assert z_q.shape == orig_shape, f"Shape mismatch: {z_q.shape} vs {orig_shape}"
        
        # Use straight-through estimator
        z_q = z.permute(0, 3, 1, 2).contiguous() + (z_q - z.permute(0, 3, 1, 2).contiguous()).detach()
        
        # Reshape indices to match spatial dimensions
        min_encoding_indices = min_encoding_indices.view(orig_shape[0], orig_shape[2], orig_shape[3])
        
        return z_q, min_encoding_indices

# Add patch-based encoding to VQVAE
class SimpleVQVAE(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=256):
        super().__init__()
        self.encoder = SimpleEncoder(in_channels=3, latent_dim=embedding_dim)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = SimpleDecoder(latent_dim=embedding_dim, out_channels=3)
        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        """Forward pass through the VQVAE"""
        # Encode the input
        z = self.encoder(x)  # Shape: [B, embedding_dim, H/4, W/4]
        
        # Quantize the latent representations
        z_q, indices = self.quantizer(z)  # Same shape as z
        
        # Decode
        x_recon = self.decoder(z_q)  # Shape: [B, 3, H, W]
        
        return x_recon, z_q, indices
    
    def encode_to_indices(self, x):
        """Encode input to indices only using full resolution"""
        z = self.encoder(x)
        _, indices = self.quantizer(z)
        return indices

    def encode_to_indices_with_scale(self, x, image_size, patch_size):
        """
        Encode the image into token indices at a specific scale.
        The target token grid resolution is computed as image_size // patch_size.
        """
        z = self.encoder(x)  # z: [B, embedding_dim, H', W']
        target_resolution = image_size // patch_size  # e.g., 256//16 = 16, 256//8 = 32, etc.
        if z.shape[2] != target_resolution or z.shape[3] != target_resolution:
            z = F.adaptive_avg_pool2d(z, (target_resolution, target_resolution))
        _, indices = self.quantizer(z)  # indices shape: [B, target_resolution, target_resolution]
        return indices
    
    def decode_from_indices(self, indices):
        """Decode from token indices using the quantizer and decoder.
        If indices is a flattened tensor (i.e. 2D), it is automatically reshaped into a grid.
        """
        # If indices are flattened, reshape to a square grid if possible.
        if indices.dim() == 2:  # shape: [B, num_tokens]
            num_tokens = indices.size(1)
            grid_size = int(num_tokens ** 0.5)
            indices = indices.view(-1, grid_size, grid_size)
        # Get embeddings and reshape to match expected decoder input shape
        z_q = self.quantizer.embedding(indices)  # [B, H, W, embedding_dim]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()  # [B, embedding_dim, H, W]
        x_recon = self.decoder(z_q)
        return x_recon

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
        """(Deprecated) Decode patches back to image."""
        B = patches.size(0)
        patches = patches.view(B, H//patch_size, W//patch_size, -1, patch_size, patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return patches.view(B, -1, H, W)