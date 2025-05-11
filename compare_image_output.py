from AutoEncoder_Latex import TransformerVAE, ImageEncoder
from train_image_encoder import ImageDataset, load_vae
import argparse
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

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
    """
    Autoregressive greedy decode from latent z.
    Returns a list of token-IDs from <sos> up to and including <eos>,
    or the full max_len sequence if no <eos> is generated.
    """
    device = z.device
    max_len = vae.max_len

    # prepare output buffer full of pad, set position 0 = SOS
    gen = torch.full((1, max_len), pad_idx, dtype=torch.long, device=device)
    gen[0, 0] = sos_idx

    # Create memory from latent vector
    memory = vae.latent_to_dec(z).unsqueeze(0)  # (1, batch, d_model)
    
    # Create source padding mask (all False since we're using the full memory)
    src_key_padding_mask = torch.zeros(1, 1, dtype=torch.bool, device=device)

    with torch.no_grad():
        for i in range(1, max_len):
            # only pass the prefix [0:i]
            prefix = gen[:, :i]                  # shape (1, i)
            logits = vae.decode(memory, prefix, pad_idx, src_key_padding_mask)  # shape (1, i, V)
            next_logits = logits[0, -1]          # shape (V,)

            # pick the highest‑probability token
            next_t = next_logits.argmax().item()
            gen[0, i] = next_t

            if next_t == eos_idx:
                # return up through the EOS token
                return gen[0, :i+1].tolist()

    # no EOS found → return entire buffer
    return gen[0].tolist()

def tokens_to_string(tokens, inv_vocab, sos_idx, eos_idx):
    """Convert list of token IDs to a string, stripping special tokens."""
    chars = []
    for t in tokens:
        if t == sos_idx:    continue
        if t == eos_idx:    break
        chars.append(inv_vocab.get(t, ''))
    return ''.join(chars)

def main():
    parser = argparse.ArgumentParser("Compare Image to LaTeX Output")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--vae', type=str, default='best_latex_vae.pth', help='Path to VAE checkpoint')
    parser.add_argument('--encoder', type=str, default='image_encoder_resnet18_epoch_03.pth', help='Path to image encoder checkpoint')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load VAE
    print("Loading VAE...")
    vae, vocab, cfg = load_vae(args.vae)
    vae = vae.to(device)
    vae.eval()
    # Freeze VAE parameters to save memory
    for p in vae.parameters():
        p.requires_grad_(False)
    
    # Load image encoder
    print("Loading image encoder...")
    img_enc = ImageEncoder(latent_dim=cfg['latent_dim']).to(device)
    ckpt = torch.load(args.encoder, map_location=device)
    state = ckpt.get('model_state_dict', ckpt.get('model', ckpt))
    img_enc.load_state_dict(state)
    img_enc.eval()

    # Process image
    print(f"Processing image: {args.image}")
    img = preprocess_image(args.image, device)
    
    # Get latent representation
    with torch.no_grad():
        mu, logvar = img_enc(img)
        z = img_enc.reparameterize(mu, logvar)

    # Decode to LaTeX
    print("Decoding to LaTeX...")
    pad_idx = vocab['<pad>']
    sos_idx = vocab['<sos>']
    eos_idx = vocab['<eos>']
    
    # Create inverse vocabulary mapping
    inv_vocab = {i: ch for ch, i in vocab.items()}
    
    token_seq = greedy_decode(vae, z, sos_idx, eos_idx, pad_idx, device, args.temperature)
    latex_str = tokens_to_string(token_seq, inv_vocab, sos_idx, eos_idx)

    # Save output
    output_path = 'output.tex'
    with open(output_path, 'w') as f:
        f.write(latex_str)
    print(f"\nGenerated LaTeX: {latex_str}")
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()