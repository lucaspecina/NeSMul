import torch
import pytest
from src.minimal_vqvae import SimpleVQVAE

def test_vqvae_shapes():
    vqvae = SimpleVQVAE()
    x = torch.randn(2, 3, 256, 256)  # Test batch of 2 images
    
    # Test encode_to_patches
    patches_16 = vqvae.encode_to_patches(x, patch_size=16)
    assert patches_16.shape == (2, 256, 768)  # (batch, 16x16 patches, 3*16*16)
    
    patches_8 = vqvae.encode_to_patches(x, patch_size=8)
    assert patches_8.shape == (2, 1024, 192)  # (batch, 32x32 patches, 3*8*8)
    
    # Test full forward pass
    z_q, indices = vqvae(patches_16)
    assert z_q.shape == patches_16.shape
    assert indices.shape == (2, 256)  # (batch, num_patches)
    
    # Test decode_from_patches
    decoded = vqvae.decode_from_patches(z_q, patch_size=16, H=256, W=256)
    assert decoded.shape == (2, 3, 256, 256) 