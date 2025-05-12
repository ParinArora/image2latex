import argparse
import json
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torchvision import transforms
from PIL import Image
from AutoEncoder_Latex import TransformerVAE, ImageEncoder

def load_vae(ckpt_path: str, device):
    from collections import Counter
    from AutoEncoder_Latex import TransformerVAE  # local import

    ckpt = torch.load(ckpt_path, map_location=device)  # Load checkpoint to correct device
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
    ).to(device)  # Move model to device
    
    # Filter out unexpected keys from state dict
    expected_keys = set(vae.state_dict().keys())
    filtered_sd = {k: v for k, v in sd.items() if k in expected_keys}
    
    # Load the filtered state dict
    vae.load_state_dict(filtered_sd, strict=False)
    vae.eval()
    return vae, vocab, cfg['latent_dim']


def load_image_encoder(enc_path, device, latent_dim):
    print(f"Loading image encoder with latent_dim={latent_dim}")  # Debug print
    # Instantiate image encoder with VAE's latent dimension
    img_enc = ImageEncoder(latent_dim=latent_dim).to(device)
    ckpt = torch.load(enc_path, map_location=device)
    # support both key conventions
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    
    # Print the shapes of the state dict for debugging
    # print("Image encoder state dict shapes:")
    # for k, v in state.items():
    #     print(f"{k}: {v.shape}")
    
    img_enc.load_state_dict(state)
    img_enc.eval()
    return img_enc


def preprocess_image(image_path, device):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                           std=(0.229, 0.224, 0.225))
    ])
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    return img


def greedy_decode(vae, z, sos_idx, eos_idx, pad_idx, device, temperature=1.0):
    batch_size = 1
    max_len = vae.max_len
    
    # Initialize with SOS token
    tokens = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
    
    # Create initial memory by projecting z
    latent_mem = vae.latent_to_dec(z).unsqueeze(0)  # (1, batch, d_model)
    
    for _ in range(max_len - 1):
        # Get padding mask for current sequence
        tgt_key_padding_mask = (tokens == pad_idx)
        
        # Create causal mask
        seq_len = tokens.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        
        # Decode
    with torch.no_grad():
        logits = vae.decode(latent_mem, tokens, pad_idx, tgt_key_padding_mask)
            
        # Get next token probabilities
        next_token_logits = logits[:, -1, :] / temperature
        next_token_probs = F.softmax(next_token_logits, dim=-1)
            
        # Sample next token
        next_token = torch.multinomial(next_token_probs, num_samples=1)
        
        # Append to sequence
        tokens = torch.cat([tokens, next_token], dim=1)
        
        # Stop if EOS token is generated
        if (next_token == eos_idx).any():
                break
    
    return tokens[0]  # Return first (and only) sequence


def tokens_to_string(tokens, vocab):
    # Skip initial <sos> and after <eos>
    chars = []
    print(f"Processing tokens: {tokens[:10]}...")  # Print first 10 tokens
    for t in tokens:
        if t == vocab['<sos>']:
            print("Found <sos> token")
            continue
        if t == vocab['<eos>']:
            print("Found <eos> token")
            break
        # Find the character for this token
        found = False
        for ch, idx in vocab.items():
            if idx == t:
                chars.append(ch)
                found = True
                break
        if not found:
            print(f"Token {t} not found in vocabulary")
    result = ''.join(chars)
    print(f"Decoded LaTeX: {result}")
    return result


def main():
    parser = argparse.ArgumentParser("Image to LaTeX Inference")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--vae', type=str, default='best_latex_vae.pth', help='Path to VAE checkpoint')
    parser.add_argument('--encoder', type=str, default='image_encoder_resnet18_epoch_03.pth', help='Path to image encoder checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--temperature', type=float, default=1.5, help='Sampling temperature')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load models
    print("Loading VAE...")
    vae, vocab, latent_dim = load_vae(args.vae, device)
    print(f"VAE latent_dim: {latent_dim}")
    print(f"Vocabulary size: {len(vocab)}")
    
    # Print special tokens
    print("\nSpecial tokens:")
    for token in ['<sos>', '<eos>', '<pad>', '<unk>']:
        if token in vocab:
            print(f"{token}: {vocab[token]}")
    
    # Print sample vocabulary items
    print("\nSample vocabulary items:")
    for i, (k, v) in enumerate(list(vocab.items())[:10]):
        print(f"{k}: {v}")
    
    pad_idx = vocab['<pad>']
    sos_idx = vocab['<sos>']
    eos_idx = vocab['<eos>']
    print(f"\nSpecial token indices: pad={pad_idx}, sos={sos_idx}, eos={eos_idx}")

    print("Loading image encoder...")
    img_enc = load_image_encoder(args.encoder, device, latent_dim)

    # Preprocess and encode image
    print(f"Processing image: {args.image}")
    img = preprocess_image(args.image, device)
    with torch.no_grad():
        mu, logvar = img_enc(img)
        print(f"Image encoder output shape: {mu.shape}")
        print(f"Image encoder output range: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
        # Use reparameterization trick instead of normalization
        z = img_enc.reparameterize(mu, logvar)
        print(f"Reparameterized z range: [{z.min().item():.4f}, {z.max().item():.4f}]")

    # Decode to token sequence
    print("Decoding image to LaTeX...")
    token_seq = greedy_decode(vae, z, sos_idx, eos_idx, pad_idx, device, args.temperature)
    latex_str = tokens_to_string(token_seq, vocab)

    # Save output
    output_path = 'output.tex'
    with open(output_path, 'w') as f:
        f.write(latex_str)
    print(f"Saved LaTeX to {output_path}")


if __name__ == '__main__':
    main()
