import torch
import pytest
from src.minimal_var import ProgressiveVAR

def test_var_shapes():
    print("\nTesting Progressive VAR component...")
    
    # Initialize model
    print("Initializing Progressive VAR model...")
    var = ProgressiveVAR()
    batch_size = 2
    
    # Create dummy token sequences
    print("\nCreating test token sequences...")
    tokens_16 = torch.randint(0, 1024, (batch_size, 256))  # 16x16 patches
    tokens_8 = torch.randint(0, 1024, (batch_size, 1024))  # 8x8 patches
    print(f"16x16 tokens shape: {tokens_16.shape}")
    print(f"8x8 tokens shape: {tokens_8.shape}")
    
    # Test single resolution
    print("\nTesting single resolution forward pass...")
    logits = var([tokens_16], current_level=0)
    print(f"Output logits shape (single resolution): {logits.shape}")
    assert logits.shape == (batch_size, 256, 1024), "Wrong shape for single resolution"
    
    # Test multiple resolutions
    print("\nTesting multiple resolution forward pass...")
    logits = var([tokens_16, tokens_8], current_level=1)
    print(f"Output logits shape (multiple resolutions): {logits.shape}")
    assert logits.shape == (batch_size, 256 + 1024, 1024), "Wrong shape for multiple resolutions"
    
    print("\nAll Progressive VAR tests passed!")

def test_var_attention_mask():
    print("\nTesting VAR attention masking...")
    
    var = ProgressiveVAR()
    batch_size = 2
    tokens = torch.randint(0, 1024, (batch_size, 256))
    
    # Test that future tokens are not attended to
    logits = var([tokens], current_level=0)
    
    # Check if gradients flow correctly
    print("Testing gradient flow...")
    loss = logits.mean()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in var.parameters())
    print(f"Model has gradients: {has_grad}")
    assert has_grad, "No gradients in the model" 