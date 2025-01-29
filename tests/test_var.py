import torch
import pytest
from src.minimal_var import ProgressiveVAR

def test_var_shapes():
    var = ProgressiveVAR()
    batch_size = 2
    
    # Create dummy token sequences for different resolutions
    tokens_16 = torch.randint(0, 1024, (batch_size, 256))  # 16x16 patches
    tokens_8 = torch.randint(0, 1024, (batch_size, 1024))  # 8x8 patches
    
    # Test progressive generation
    logits = var([tokens_16], current_level=0)
    assert logits.shape == (batch_size, 256, 1024)  # (batch, sequence_length, vocab_size)
    
    # Test with multiple resolutions
    logits = var([tokens_16, tokens_8], current_level=1)
    assert logits.shape == (batch_size, 256 + 1024, 1024) 