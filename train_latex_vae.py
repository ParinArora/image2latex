import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader, random_split
import os
import json
from tqdm import tqdm
import numpy as np
from AutoEncoder_Latex import TransformerVAE, vae_loss, ImageEncoder
import torch.nn.functional as F

class LatexDataset(Dataset):
    def __init__(self, file_path, max_len=256):
        self.file_path = file_path
        self.max_len = max_len
        self.data = self._load_data()
        self.vocab = self._build_vocab()
        
    def _build_vocab(self):
        vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        idx = 3
        for formula in self.data:
            for char in formula:
                if char not in vocab:
                    vocab[char] = idx
                    idx += 1
        return vocab
    
    def _load_data(self):
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                formula = line.strip()
                if formula:
                    data.append(formula)
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        # Start with SOS
        tokens = [self.vocab['<sos>']]
        
        # Add content tokens up to max_len-2 (leaving room for EOS)
        for c in text:
            if len(tokens) >= self.max_len - 1:
                break
            tokens.append(self.vocab.get(c, self.vocab['<pad>']))
        
        # Add EOS if there's room
        if len(tokens) < self.max_len:
            tokens.append(self.vocab['<eos>'])
        
        # Record true length before padding
        true_len = len(tokens)
        
        # Pad to max_len
        tokens = tokens + [self.vocab['<pad>']] * (self.max_len - true_len)
        
        # Create mask: True for real tokens, False for padding
        length_mask = torch.zeros(self.max_len, dtype=torch.bool)
        length_mask[:true_len] = True
        
        return torch.tensor(tokens, dtype=torch.long), length_mask


def train_epoch(model, loader, optimizer, scheduler, device, pad_idx, sos_idx, eos_idx, scaler, clip_grad=1.0, beta=0.01, epoch=0, num_epochs=10):
    model.train()
    total_loss = total_recon = total_kl = total_length = total_diversity = total_eos = 0.0
    
    # Improved KL annealing: faster warm-up
    current_beta = beta * min(1.0, epoch / (num_epochs // 2))
    
    pbar = tqdm(loader, desc='Train')
    for batch, _ in pbar:  # We don't need length_mask anymore
        batch = batch.to(device)
        
        # prepare decoder inputs and shifted targets
        tgt_input = batch[:, :-1]           # drop last token
        tgt_output = batch[:, 1:]           # drop first token
        
        # Use mixed precision training
        with torch.amp.autocast(device_type='cuda'):
            logits, mu, logvar = model(batch, tgt_input, pad_idx)
            loss, recon_loss, kl_loss = vae_loss(logits, tgt_output, mu, logvar, pad_idx, sos_idx, eos_idx, beta=current_beta)
            
            # Calculate length and diversity penalties
            token_probs = F.softmax(logits, dim=-1)
            pad_token_probs = token_probs[:, :, pad_idx]
            length_penalty = torch.mean(pad_token_probs)
            
            # Calculate EOS penalty
            eos_token_probs = token_probs[:, :, eos_idx]
            eos_penalty = torch.mean((eos_token_probs - 0.5).pow(2))
            
            token_entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-10), dim=-1)
            diversity_penalty = -torch.mean(token_entropy)
            
            # Log token distribution
            top_preds = logits.argmax(dim=-1)
            pad_frac = (top_preds == pad_idx).float().mean().item()
            eos_frac = (top_preds == eos_idx).float().mean().item()
            sos_frac = (top_preds == sos_idx).float().mean().item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_length += length_penalty.item()
        total_diversity += diversity_penalty.item()
        total_eos += eos_penalty.item()
        
        pbar.set_postfix({
            'loss': total_loss / (pbar.n+1),
            'recon': total_recon / (pbar.n+1),
            'kl': total_kl / (pbar.n+1),
            'length': total_length / (pbar.n+1),
            'diversity': total_diversity / (pbar.n+1),
            'eos': total_eos / (pbar.n+1),
            'beta': current_beta,
            'pad%': f'{pad_frac:.2%}',
            'eos%': f'{eos_frac:.2%}',
            'sos%': f'{sos_frac:.2%}'
        })
    return total_loss / len(loader)


def validate_epoch(model, loader, device, pad_idx, sos_idx, eos_idx, beta=0.01, epoch=0, num_epochs=10):
    model.eval()
    total_loss = total_recon = total_kl = total_length = total_diversity = total_eos = 0.0
    
    # Use same beta as training
    current_beta = beta * min(1.0, epoch / (num_epochs // 2))
    
    with torch.no_grad():
        pbar = tqdm(loader, desc='Val  ')
        for batch, _ in pbar:  # We don't need length_mask anymore
            batch = batch.to(device)
            
            # prepare decoder inputs and shifted targets
            tgt_input = batch[:, :-1]           # drop last token
            tgt_output = batch[:, 1:]           # drop first token
            
            with torch.amp.autocast(device_type='cuda'):
                logits, mu, logvar = model(batch, tgt_input, pad_idx)
                loss, recon_loss, kl_loss = vae_loss(logits, tgt_output, mu, logvar, pad_idx, sos_idx, eos_idx, beta=current_beta)
                
                # Calculate length and diversity penalties
                token_probs = F.softmax(logits, dim=-1)
                pad_token_probs = token_probs[:, :, pad_idx]
                length_penalty = torch.mean(pad_token_probs)
                
                # Calculate EOS penalty
                eos_token_probs = token_probs[:, :, eos_idx]
                eos_penalty = torch.mean((eos_token_probs - 0.5).pow(2))
                
                token_entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-10), dim=-1)
                diversity_penalty = -torch.mean(token_entropy)
                
                # Log token distribution
                top_preds = logits.argmax(dim=-1)
                pad_frac = (top_preds == pad_idx).float().mean().item()
                eos_frac = (top_preds == eos_idx).float().mean().item()
                sos_frac = (top_preds == sos_idx).float().mean().item()
                
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
            total_length += length_penalty.item()
            total_diversity += diversity_penalty.item()
            total_eos += eos_penalty.item()
            
            pbar.set_postfix({
                'loss': total_loss / (pbar.n+1),
                'recon': total_recon / (pbar.n+1),
                'kl': total_kl / (pbar.n+1),
                'length': total_length / (pbar.n+1),
                'diversity': total_diversity / (pbar.n+1),
                'eos': total_eos / (pbar.n+1),
                'beta': current_beta,
                'pad%': f'{pad_frac:.2%}',
                'eos%': f'{eos_frac:.2%}',
                'sos%': f'{sos_frac:.2%}'
            })
    return total_loss / len(loader)


def main():
    # Hyperparameters
    batch_size = 128
    num_epochs = 10
    learning_rate = 5e-5
    latent_dim = 128
    d_model = 512
    nhead = 8
    num_layers = 6
    max_len = 512
    weight_decay = 1e-2
    warmup_ratio = 0.1
    dropout = 0.2
    checkpoint_every = 1  # Save checkpoint every N epochs
    beta = 0.01  # KL divergence weight (reduced from 0.1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = LatexDataset(
        '/home/parin.arora_ug2023/text2latex/PRINTED_TEX_230k/final_png_formulas.txt',
        max_len=max_len
    )
    print(f"Total formulas: {len(dataset)} | Vocab size: {len(dataset.vocab)}")

    # Get special token indices
    pad_idx = dataset.vocab['<pad>']
    sos_idx = dataset.vocab['<sos>']
    eos_idx = dataset.vocab['<eos>']

    # Train/Val split
    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=4, pin_memory=True)

    model = TransformerVAE(
        vocab_size=len(dataset.vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        latent_dim=latent_dim,
        max_len=max_len,
        dropout=dropout
    ).to(device)

    # Use mixed precision training
    scaler = torch.amp.GradScaler()
    
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = LambdaLR(optimizer, lambda step: min((step+1)/warmup_steps, 1.0))

    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)

    best_val = float('inf')
    for epoch in range(1, num_epochs+1):
        print(f"Epoch {epoch}/{num_epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, pad_idx, sos_idx, eos_idx, scaler, beta=beta, epoch=epoch, num_epochs=num_epochs)
        val_loss = validate_epoch(model, val_loader, device, pad_idx, sos_idx, eos_idx, beta=beta, epoch=epoch, num_epochs=num_epochs)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save checkpoint every N epochs
        if epoch % checkpoint_every == 0:
            checkpoint_path = f'checkpoints/latex_vae_epoch_{epoch:03d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab': dataset.vocab,
                'config': {
                    'd_model': d_model,
                    'nhead': nhead,
                    'num_layers': num_layers,
                    'latent_dim': latent_dim,
                    'max_len': max_len,
                    'dropout': dropout,
                    'vocab_size': len(dataset.vocab)
                }
            }, checkpoint_path)
            print(f"  Saved checkpoint to {checkpoint_path}")

        # Save best model if validation loss improved
        if val_loss < best_val:
            best_val = val_loss
            best_model_path = 'checkpoints/best_latex_vae.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'vocab': dataset.vocab,
                'config': {
                    'd_model': d_model,
                    'nhead': nhead,
                    'num_layers': num_layers,
                    'latent_dim': latent_dim,
                    'max_len': max_len,
                    'dropout': dropout,
                    'vocab_size': len(dataset.vocab)
                }
            }, best_model_path)
            print(f"  Saved new best model with val_loss: {val_loss:.4f}")

    # Save final model
    final_model_path = 'checkpoints/final_latex_vae.pth'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'vocab': dataset.vocab,
        'config': {
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'latent_dim': latent_dim,
            'max_len': max_len,
            'dropout': dropout,
            'vocab_size': len(dataset.vocab)
        }
    }, final_model_path)
    print(f"Saved final model to {final_model_path}")

if __name__ == '__main__':
    main()