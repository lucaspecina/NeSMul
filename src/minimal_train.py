import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from minimal_vqvae import SimpleVQVAE
from minimal_var import ProgressiveVAR
import torch.nn.functional as F
import numpy as np
from torchvision.utils import save_image
import os
from pathlib import Path

def save_checkpoint(model, optimizer, epoch, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, filename)

def load_checkpoint(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch']

def train_progressive(vqvae, var, optimizer, images, device):
    var.train()
    losses = []
    batch_size = images.size(0)
    
    # Get tokens at different resolutions
    tokens_at_scales = []
    with torch.no_grad():
        for patch_size in var.patch_sizes:
            patches = vqvae.encode_to_patches(images, patch_size)
            _, indices = vqvae(patches)
            tokens_at_scales.append(indices)
    
    # Progressive training through different resolutions
    for current_level in range(len(var.patch_sizes)):
        # Prepare input and target sequences
        input_sequences = tokens_at_scales[:current_level + 1]
        target_indices = tokens_at_scales[current_level][:, 1:]  # All except first token
        
        # Forward pass
        logits = var(input_sequences, current_level)
        
        # Calculate loss only on the current resolution's predictions
        current_level_logits = logits[:, -target_indices.size(1):]
        loss = F.cross_entropy(
            current_level_logits.view(-1, logits.size(-1)),
            target_indices.view(-1)
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    return np.mean(losses)

def generate_progressive(vqvae, var, device, image_size=256, temperature=1.0):
    var.eval()
    with torch.no_grad():
        batch_size = 1
        generated_tokens = []
        
        # Generate from coarse to fine
        for level, patch_size in enumerate(var.patch_sizes):
            num_patches = (image_size // patch_size) ** 2
            current_tokens = torch.zeros(batch_size, num_patches, dtype=torch.long).to(device)
            
            # Generate tokens for current resolution
            for position in range(num_patches):
                # Forward pass through VAR
                logits = var(generated_tokens + [current_tokens], level)
                
                # Sample from the logits with temperature
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                current_tokens[:, position] = next_token.squeeze()
            
            generated_tokens.append(current_tokens)
        
        # Decode the final result
        final_tokens = generated_tokens[-1]
        patches = vqvae.quantizer.embedding(final_tokens)
        reconstructed_image = vqvae.decode_from_patches(
            patches,
            var.patch_sizes[-1],
            image_size,
            image_size
        )
        
        return reconstructed_image

def main():
    # Create output directories
    output_dir = Path("outputs")
    checkpoint_dir = output_dir / "checkpoints"
    sample_dir = output_dir / "samples"
    
    for dir in [output_dir, checkpoint_dir, sample_dir]:
        dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models
    vqvae = SimpleVQVAE().to(device)
    var = ProgressiveVAR().to(device)
    
    # Load VQVAE weights (assuming it's pretrained)
    vqvae.load_state_dict(torch.load("vqvae_checkpoint.pt"))
    vqvae.eval()
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    dataset = datasets.ImageFolder("path/to/dataset", transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Training setup
    optimizer = torch.optim.Adam(var.parameters(), lr=1e-4)
    start_epoch = 0
    
    # Load checkpoint if exists
    checkpoint_path = checkpoint_dir / "var_latest.pt"
    if checkpoint_path.exists():
        start_epoch = load_checkpoint(var, optimizer, checkpoint_path)
    
    # Training loop
    for epoch in range(start_epoch, 100):
        epoch_losses = []
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            loss = train_progressive(vqvae, var, optimizer, images, device)
            epoch_losses.append(loss)
            
            if batch_idx % 100 == 0:
                avg_loss = np.mean(epoch_losses[-100:])
                print(f"Epoch: {epoch}, Batch: {batch_idx}, Loss: {avg_loss:.4f}")
                
                # Generate sample images with different temperatures
                for temp in [0.5, 1.0, 2.0]:
                    sample = generate_progressive(vqvae, var, device, temperature=temp)
                    save_image(
                        sample,
                        sample_dir / f"sample_e{epoch}_b{batch_idx}_t{temp}.png",
                        normalize=True
                    )
        
        # Save checkpoint
        save_checkpoint(var, optimizer, epoch, checkpoint_dir / "var_latest.pt")
        if epoch % 10 == 0:
            save_checkpoint(var, optimizer, epoch, checkpoint_dir / f"var_epoch_{epoch}.pt")

if __name__ == "__main__":
    main()