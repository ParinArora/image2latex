import argparse
import torch
import torch.nn.functional as F
from AutoEncoder_Latex import TransformerVAE

def load_vae(ckpt_path: str, device):
    """Load VAE model and vocabulary from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = ckpt["vocab"]
    sd    = ckpt["model_state_dict"]

    # static config (could also load from ckpt["cfg"] if present)
    cfg = {
        "batch_size":    128,
        "num_epochs":    10,
        "learning_rate": 5e-5,
        "latent_dim":    128,
        "d_model":       512,
        "nhead":         8,
        "num_layers":    6,
        "max_len":       512,
        "dropout":       0.2,
    }

    vae = TransformerVAE(
        vocab_size = len(vocab),
        d_model     = cfg["d_model"],
        nhead       = cfg["nhead"],
        num_layers  = cfg["num_layers"],
        latent_dim  = cfg["latent_dim"],
        max_len     = cfg["max_len"],
        dropout     = cfg["dropout"]
    ).to(device)

    # filter out any unexpected keys (e.g. if code changed)
    expected = set(vae.state_dict().keys())
    filtered = {k: v for k, v in sd.items() if k in expected}
    vae.load_state_dict(filtered, strict=False)
    vae.eval()
    return vae, vocab, cfg

def encode_latex(latex_str, vocab, max_len, device):
    """Convert LaTeX string to token sequence of length max_len."""
    # build token IDs (use <pad> for unknown chars)
    pad_idx = vocab['<pad>']
    tokens = [vocab['<sos>']]
    for ch in latex_str:
        tokens.append(vocab.get(ch, pad_idx))
    tokens.append(vocab['<eos>'])

    # truncate or pad
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
        tokens[-1] = vocab['<eos>']
    else:
        tokens += [pad_idx] * (max_len - len(tokens))

    return torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

def greedy_decode(vae, z, sos_idx, eos_idx, pad_idx):
    """
    Autoregressive greedy decode from latent z.
    Returns a list of token-IDs from <sos> up to and including <eos>,
    or the full max_len sequence if no <eos> is generated.
    """
    device  = z.device
    max_len = vae.max_len

    # prepare output buffer full of pad, set position 0 = SOS
    gen = torch.full((1, max_len), pad_idx, dtype=torch.long, device=device)
    gen[0, 0] = sos_idx

    with torch.no_grad():
        for i in range(1, max_len):
            # only pass the prefix [0:i]
            prefix = gen[:, :i]                  # shape (1, i)
            logits = vae.decode(z, prefix)       # shape (1, i, V)
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
    parser = argparse.ArgumentParser("LaTeX ↔ VAE Round-trip")
    parser.add_argument('--input',       type=str,   required=True)
    parser.add_argument('--vae',         type=str,   default='best_latex_vae.pth')
    parser.add_argument('--device',      type=str,   default='cuda')
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1) load
    vae, vocab, cfg = load_vae(args.vae, device)
    print(f"Loaded VAE; latent_dim={cfg['latent_dim']}  max_len={cfg['max_len']}")
    print("Vocab size:", len(vocab))

    # special tokens
    sos_idx = vocab['<sos>']; eos_idx = vocab['<eos>']; pad_idx = vocab['<pad>']
    inv_vocab = {i:ch for ch,i in vocab.items()}

    # 2) encode
    print("Encoding:", args.input)
    tokens = encode_latex(args.input, vocab, vae.max_len, device)

    # 3) get z
    with torch.no_grad():
        mu, logvar = vae.encode(tokens)
        z = vae.reparameterize(mu, logvar)
    print("Encoded to z shape:", z.shape)

    # 4) decode
    print("Decoding with temperature =", args.temperature)
    out_tokens = greedy_decode(vae, z, sos_idx, eos_idx, pad_idx)
    out_str    = tokens_to_string(out_tokens, inv_vocab, sos_idx, eos_idx)
    print("→", out_str)

    # 5) write
    with open('output.tex','w') as f:
        f.write(out_str)
    print("Saved output.tex")

    # 6) quick round-trip test (optional)
    if len(args.input) < cfg['max_len']-2:
        rec_tokens = encode_latex(out_str, vocab, vae.max_len, device)
        assert rec_tokens[0,0].item()==sos_idx and rec_tokens[0,(len(out_str)+1)].item()==eos_idx
        print("Round-trip self-check passed.")

if __name__ == '__main__':
    main()
