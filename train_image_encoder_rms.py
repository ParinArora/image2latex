from pathlib import Path
from typing import Dict, List, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from AutoEncoder_Latex import ImageEncoder, vae_loss
import os
import numpy as np
import random
import json
from datetime import datetime
import torch.nn.functional as F

# Try to import tensorboard, but make it optional
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Will use basic file logging instead.")

class Logger:
    def __init__(self, log_dir='logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(log_dir, f'training_{timestamp}.jsonl')
        self.metrics = {}
        
    def add_scalar(self, tag, value, step):
        if tag not in self.metrics:
            self.metrics[tag] = []
        self.metrics[tag].append({'step': step, 'value': value})
        
    def add_scalars(self, tag, metrics, step):
        if tag not in self.metrics:
            self.metrics[tag] = []
        self.metrics[tag].append({'step': step, **metrics})
        
    def flush(self):
        with open(self.log_file, 'a') as f:
            for tag, values in self.metrics.items():
                for entry in values:
                    f.write(json.dumps({'tag': tag, **entry}) + '\n')
        self.metrics = {}
    
    def close(self):
        self.flush()

# ────────────────────────────────────────────────────────────
# 1. DATASET  -  reuse VAE vocab, never create new indices
# ────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    """PNG  ↔  LaTeX  → token‑ids (fixed vocab)."""

    def __init__(
        self,
        latex_file: str,
        png_names_file: str,
        png_dir: str,
        vocab: Dict[str, int],
        max_len: int,
        transform=None,
    ) -> None:
        self.formulas = self._read_lines(latex_file)
        self.png_names = self._read_lines(png_names_file)
        assert len(self.formulas) == len(self.png_names), "#lines mismatch"

        self.png_dir = Path(png_dir)
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                              std=(0.229, 0.224, 0.225))
        ])
        self.max_len = max_len

        # Use the VAE's vocabulary directly, without adding <unk>
        self.vocab = vocab
        self.pad = vocab["<pad>"]
        self.sos = vocab["<sos>"]
        self.eos = vocab["<eos>"]

    @staticmethod
    def _read_lines(path: str) -> List[str]:
        with open(path, encoding="utf-8") as f:
            return [ln.rstrip("\n") for ln in f]

    def _encode(self, formula: str) -> torch.Tensor:
        # Map OOV characters to <pad> instead of using <unk>
        ids = [self.sos] + [self.vocab.get(c, self.pad) for c in formula] + [self.eos]
        if len(ids) > self.max_len:  # truncate
            ids = ids[: self.max_len]
            ids[-1] = self.eos
        else:  # pad
            ids += [self.pad] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    # ‑‑ Dataset API ‑‑
    def __len__(self):
        return len(self.formulas)

    def __getitem__(self, idx):
        img = Image.open(self.png_dir / self.png_names[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        tokens = self._encode(self.formulas[idx])
        return img, tokens


# ────────────────────────────────────────────────────────────
# 2.  Load VAE checkpoint *with original hyper‑params*
# ────────────────────────────────────────────────────────────

def load_vae(ckpt_path: str):
    from collections import Counter
    from AutoEncoder_Latex import TransformerVAE  # local import

    ckpt = torch.load(ckpt_path, map_location="cpu")
    vocab: Dict[str, int] = ckpt["vocab"]
    sd = ckpt["model_state_dict"]
    # cfg = ckpt["cfg"]

    cfg = {
        "batch_size": 128,
        "num_epochs": 10,
        "learning_rate": 5e-5,
        "latent_dim": 128,
        "d_model": 512,
        "nhead": 8,
        "num_layers": 6,
        "max_len": 512,
        "weight_decay": 1e-2,
        "warmup_ratio": 0.1,
        "dropout": 0.2,
    }

    vae = TransformerVAE(
        vocab_size=len(vocab),
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_layers=cfg["num_layers"],
        latent_dim=cfg["latent_dim"],
        max_len=cfg["max_len"],
        dropout=cfg["dropout"]
    )
    # Filter out unexpected keys from state dict
    expected_keys = set(vae.state_dict().keys())
    filtered_sd = {k: v for k, v in sd.items() if k in expected_keys}
    
    # Load the filtered state dict
    vae.load_state_dict(filtered_sd, strict=False)
    vae.eval()
    return vae, vocab, cfg


# ────────────────────────────────────────────────────────────
# 3.  Batch‑level sanity‑check on token IDs
# ────────────────────────────────────────────────────────────

def assert_tokens_in_vocab(t: torch.Tensor, emb: nn.Embedding):
    if int(t.max()) >= emb.num_embeddings:
        bad = torch.unique(t[t >= emb.num_embeddings]).tolist()
        raise RuntimeError(f"Token id(s) {bad} out of range (vocab={emb.num_embeddings})")


# ────────────────────────────────────────────────────────────
# 4.  Training stub
# ────────────────────────────────────────────────────────────

def train_epoch(model, loader, vae, device, optimizer, scaler, scheduler, vocab, epoch, num_epochs, kl_weight, lr_warmup_steps, writer):
    model.train()
    vae.eval()  # Keep VAE in eval mode
    total_loss = total_kl = total_recon = total_latex = 0.0
    
    pbar = tqdm(loader, desc=f'Train (Epoch {epoch}/{num_epochs})')
    for i, (images, tokens) in enumerate(pbar, start=1):
        # Calculate global step for current batch
        current_batch_global_step = (epoch - 1) * len(loader) + i
        
        # KL Annealing: Linearly ramp up KL weight from 0 to kl_weight over lr_warmup_steps
        anneal_factor = min(1.0, current_batch_global_step / max(1, lr_warmup_steps))
        current_kl_weight = kl_weight * anneal_factor
        
        images = images.to(device)
        tokens = tokens.to(device)
        
        # Sanity check token IDs
        assert_tokens_in_vocab(tokens, vae.embedding)
        
        # Get VAE encoder outputs (fixed)
        with torch.no_grad():
            memory, text_mu, text_logvar, src_mask = vae.encode(tokens, vocab['<pad>'])
        
        # Get image encoder outputs
        img_mu, img_logvar = model(images)
        
        # Symmetric KL divergence between image encoder and VAE encoder
        kl_img2txt = (
            img_logvar - text_logvar + 
            (text_logvar.exp() + (img_mu - text_mu).pow(2)) / img_logvar.exp() - 1
        )
        kl_txt2img = (
            text_logvar - img_logvar + 
            (img_logvar.exp() + (text_mu - img_mu).pow(2)) / text_logvar.exp() - 1
        )
        kl_loss = 0.5 * (kl_img2txt.mean(0).sum() + kl_txt2img.mean(0).sum()) / img_mu.size(1)
        
        # Get reconstruction using VAE decoder
        z = model.reparameterize(img_mu, img_logvar)
        
        # Inject latent into memory
        latent_mem = vae.latent_to_dec(z).unsqueeze(0).expand(memory.size(0), -1, -1)
        memory = memory + latent_mem
        
        # Teacher-force decode: feed the full token sequence as 'tgt'
        logits = vae.decode(memory, tokens, vocab['<pad>'], src_mask)
        
        # Reconstruction loss using RMS
        logits = logits.contiguous()
        tokens = tokens.contiguous()
        
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)
        
        # Create one-hot encoding of target tokens
        target_one_hot = F.one_hot(tokens, num_classes=logits.size(-1)).float()
        
        # Calculate mean squared error
        mse = (probs - target_one_hot).pow(2)
        
        # Apply padding mask to ignore padding tokens
        padding_mask = (tokens != vocab['<pad>']).float().unsqueeze(-1)
        masked_mse = mse * padding_mask
        
        # Calculate RMS loss
        recon_loss = torch.sqrt(masked_mse.mean())
        
        # LaTeX-specific loss components
        # 1. Command completion loss: Penalize incomplete LaTeX commands
        command_tokens = [vocab.get(c, vocab['<pad>']) for c in ['\\', '{', '}', '^', '_']]
        command_mask = torch.zeros_like(tokens, dtype=torch.bool)
        for cmd in command_tokens:
            command_mask |= (tokens == cmd)
        
        # Find incomplete commands (e.g., \ without closing brace)
        incomplete_commands = torch.zeros_like(tokens, dtype=torch.float)
        for b in range(tokens.size(0)):
            for t in range(tokens.size(1)-1):
                if tokens[b,t] == vocab['\\']:
                    # Check if next token is a valid command character
                    if tokens[b,t+1] not in [vocab.get(c, vocab['<pad>']) for c in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']]:
                        incomplete_commands[b,t] = 1.0
        
        command_loss = (incomplete_commands * command_mask.float()).mean()
        
        # 2. Brace matching loss: Penalize unmatched braces
        brace_stack = torch.zeros_like(tokens, dtype=torch.float)
        for b in range(tokens.size(0)):
            stack = 0
            for t in range(tokens.size(1)):
                if tokens[b,t] == vocab['{']:
                    stack += 1
                elif tokens[b,t] == vocab['}']:
                    stack -= 1
                brace_stack[b,t] = abs(stack)
        
        brace_loss = brace_stack.mean()
        
        # 3. Length penalty: Encourage proper sequence lengths
        # Calculate token distribution entropy
        token_entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        length_penalty = -torch.mean(token_entropy)  # negative because we want to maximize entropy
        
        # 4. EOS placement loss: Encourage proper sequence termination
        eos_token_probs = probs[:, :, vocab['<eos>']]
        eos_penalty = torch.mean((eos_token_probs - 0.5).pow(2))
        
        # Combine LaTeX-specific losses
        latex_loss = (
            0.1 * command_loss +    # Penalize incomplete commands
            0.1 * brace_loss +      # Penalize unmatched braces
            0.1 * length_penalty +  # Encourage diverse outputs
            0.1 * eos_penalty       # Encourage proper EOS placement
        )
        
        # Total loss with weighted components
        loss = recon_loss + current_kl_weight * kl_loss + latex_loss
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Update running averages
        total_loss += loss.item()
        total_kl += kl_loss.item()
        total_recon += recon_loss.item()
        total_latex += latex_loss.item()
        
        # Log to TensorBoard / Logger
        writer.add_scalar('Loss/total', loss.item(), current_batch_global_step)
        writer.add_scalar('Loss/reconstruction', recon_loss.item(), current_batch_global_step)
        writer.add_scalar('Loss/kl', kl_loss.item(), current_batch_global_step)
        writer.add_scalar('Loss/latex', latex_loss.item(), current_batch_global_step)
        writer.add_scalar('Loss/command', command_loss.item(), current_batch_global_step)
        writer.add_scalar('Loss/brace', brace_loss.item(), current_batch_global_step)
        writer.add_scalar('Loss/length', length_penalty.item(), current_batch_global_step)
        writer.add_scalar('Loss/eos', eos_penalty.item(), current_batch_global_step)
        writer.add_scalar('Loss/kl_weight', current_kl_weight, current_batch_global_step)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], current_batch_global_step)
        
        pbar.set_postfix({
            'loss': total_loss / i,
            'kl': total_kl / i,
            'recon': total_recon / i,
            'latex': total_latex / i,
            'kl_weight': current_kl_weight,
            'lr': optimizer.param_groups[0]['lr']
        })
    
    # Log epoch averages
    avg_loss = total_loss / len(loader)
    avg_kl = total_kl / len(loader)
    avg_recon = total_recon / len(loader)
    avg_latex = total_latex / len(loader)
    
    writer.add_scalars('Epoch', {
        'total_loss': avg_loss,
        'kl_loss': avg_kl,
        'recon_loss': avg_recon,
        'latex_loss': avg_latex
    }, epoch)
    
    if hasattr(writer, 'flush') and callable(writer.flush):
        writer.flush()
    
    return avg_loss, avg_kl, avg_recon, avg_latex


# ────────────────────────────────────────────────────────────
# 5.  CLI
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser("train image‑encoder with frozen VAE")
    p.add_argument("--latex_file", required=True)
    p.add_argument("--png_names_file", required=True)
    p.add_argument("--png_dir", required=True)
    p.add_argument("--vae_ckpt", required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--kl_weight", type=float, default=0.05, help="Target weight for KL divergence loss after annealing.")
    p.add_argument("--warmup_ratio", type=float, default=0.1, help="Ratio of total steps for learning rate warmup (also used for KL annealing).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume training from.")

    args = p.parse_args()

    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Set deterministic behavior for CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create logger (TensorBoard or basic file logger)
    if TENSORBOARD_AVAILABLE:
        writer = SummaryWriter('runs/image_encoder')
    else:
        writer = Logger('logs/image_encoder')
        print("Using basic file logging. Logs will be saved to logs/image_encoder/")

    # VAE & vocab
    vae, vocab, cfg = load_vae(args.vae_ckpt)
    vae.to(device)
    vae.eval()  # Keep VAE in eval mode
    # Disable gradients for all VAE parameters
    for p in vae.parameters():
        p.requires_grad_(False)
    print("Loaded VAE cfg:", cfg)

    # Data
    ds = ImageDataset(
        latex_file=args.latex_file,
        png_names_file=args.png_names_file,
        png_dir=args.png_dir,
        vocab=vocab,
        max_len=cfg["max_len"],
    )
    
    # Create generator with fixed seed for DataLoader
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    dl = DataLoader(
        ds, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4, 
        pin_memory=True,
        generator=g
    )

    # Use proper ImageEncoder from AutoEncoder_Latex
    model = ImageEncoder(latent_dim=cfg["latent_dim"]).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler()  # For mixed precision

    # Learning rate scheduler with warmup
    total_training_steps = len(dl) * args.epochs
    lr_warmup_steps = int(total_training_steps * args.warmup_ratio)
    scheduler = LambdaLR(opt, lambda step: min((step+1)/lr_warmup_steps, 1.0) if lr_warmup_steps > 0 else 1.0)

    start_epoch = 1
    best_loss = float('inf')

    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print(f"=> loading checkpoint '{args.resume_from}'")
            checkpoint = torch.load(args.resume_from, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            best_loss = checkpoint['loss'] # Assumes 'loss' is the total loss used for determining best model
            model.load_state_dict(checkpoint['model_state_dict'])
            opt.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                 scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            else:
                print("Warning: Scheduler state not found in checkpoint. Initializing new scheduler.")
            print(f"=> loaded checkpoint '{args.resume_from}' (epoch {checkpoint['epoch']})")
            print(f"   Resuming training from epoch {start_epoch}, best_loss: {best_loss:.4f}")
        else:
            print(f"=> no checkpoint found at '{args.resume_from}'. Starting from scratch.")

    for epoch_num in range(start_epoch, args.epochs + 1):
        loss, kl_loss_val, recon_loss_val, latex_loss_val = train_epoch(
            model, dl, vae, device, opt, scaler, scheduler, 
            vocab, epoch_num, args.epochs, args.kl_weight, lr_warmup_steps, writer
        )
        
        print(f"epoch {epoch_num:02d} | loss={loss:.4f} | kl={kl_loss_val:.4f} | recon={recon_loss_val:.4f} | latex={latex_loss_val:.4f}")
        
        # Save checkpoint after each epoch
        checkpoint_path = f"image_encoder_resnet18_epoch_{epoch_num:02d}.pth"
        torch.save({
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss,
            'kl_loss': kl_loss_val,
            'recon_loss': recon_loss_val,
            'latex_loss': latex_loss_val,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

        # Save best model if current loss is better
        if loss < best_loss:
            best_loss = loss
            best_model_path = "image_encoder_best.pth"
            torch.save({
                'epoch': epoch_num,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss,
                'kl_loss': kl_loss_val,
                'recon_loss': recon_loss_val,
                'latex_loss': latex_loss_val,
            }, best_model_path)
            print(f"New best model saved with loss {loss:.4f}")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'final_loss': loss,
        'final_kl_loss': kl_loss_val,
        'final_recon_loss': recon_loss_val,
        'final_latex_loss': latex_loss_val,
        'kl_weight': args.kl_weight
    }, "image_encoder.pth")
    
    # Close logger
    if hasattr(writer, 'close') and callable(writer.close):
        writer.close()
