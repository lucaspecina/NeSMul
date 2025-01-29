import torch
import pytest
from src.minimal_vqvae import SimpleVQVAE

def test_vqvae_shapes():
    print("\nTesting VQVAE component...")
    
    # Initialize model
    print("Initializing VQVAE model...")
    vqvae = SimpleVQVAE(embedding_dim=32)
    x = torch.randn(2, 3, 256, 256)  # Test batch of 2 images
    print(f"Input shape: {x.shape}")
    
    # Test forward pass
    print("\nTesting VQVAE forward pass...")
    x_recon, z_q, indices = vqvae(x)
    print(f"Reconstruction shape: {x_recon.shape}")
    print(f"Quantized vectors shape: {z_q.shape}")
    print(f"Indices shape: {indices.shape}")
    
    # Test shapes
    assert x_recon.shape == x.shape, "Wrong reconstruction shape"
    assert z_q.shape[1] == vqvae.embedding_dim, "Wrong quantized vectors dimension"
    assert indices.shape == (x.shape[0], x.shape[2]//4, x.shape[3]//4), "Wrong indices shape"
    
    print("\nAll VQVAE shape tests passed!")

def test_vqvae_reconstruction():
    print("\nTesting VQVAE reconstruction...")
    
    vqvae = SimpleVQVAE(embedding_dim=32)
    x = torch.randn(1, 3, 256, 256)
    
    # Test reconstruction pipeline
    x_recon, _, _ = vqvae(x)
    
    # Check if reconstruction has reasonable values
    print(f"Input range: [{x.min():.2f}, {x.max():.2f}]")
    print(f"Output range: [{x_recon.min():.2f}, {x_recon.max():.2f}]")
    assert not torch.isnan(x_recon).any(), "NaN values in reconstruction"
    assert x_recon.shape == x.shape, "Reconstruction shape mismatch"
    
    print("\nAll VQVAE reconstruction tests passed!")

def test_vqvae_encoding_decoding():
    print("\nTesting VQVAE encoding/decoding...")
    
    vqvae = SimpleVQVAE(embedding_dim=32)
    x = torch.randn(1, 3, 256, 256)
    
    # Test encode-decode pipeline
    indices = vqvae.encode_to_indices(x)
    print(f"Input shape: {x.shape}")
    print(f"Indices shape: {indices.shape}")
    
    # Expected shapes after downsampling in encoder (factor of 4)
    expected_indices_shape = (x.shape[0], x.shape[2]//4, x.shape[3]//4)
    assert indices.shape == expected_indices_shape, f"Expected indices shape {expected_indices_shape}, got {indices.shape}"
    
    x_recon = vqvae.decode_from_indices(indices)
    print(f"Reconstruction shape: {x_recon.shape}")
    assert x_recon.shape == x.shape, f"Expected reconstruction shape {x.shape}, got {x_recon.shape}"
    
    # Check value ranges
    print(f"Reconstruction range: [{x_recon.min():.2f}, {x_recon.max():.2f}]")
    assert not torch.isnan(x_recon).any(), "NaN values in reconstruction"
    
    print("\nAll VQVAE encoding/decoding tests passed!") 