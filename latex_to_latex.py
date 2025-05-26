import argparse
import torch
import torch.nn.functional as F
from AutoEncoder_Latex import TransformerVAE

# ────────────────────── utilities ──────────────────────
def load_vae(ckpt_path: str, device):
    ckpt  = torch.load(ckpt_path, map_location=device)
    vocab = ckpt["vocab"]
    sd    = ckpt["model_state_dict"]

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
        d_model    = cfg["d_model"],
        nhead      = cfg["nhead"],
        num_layers = cfg["num_layers"],
        latent_dim = cfg["latent_dim"],
        max_len    = cfg["max_len"],
        dropout    = cfg["dropout"],
    ).to(device)

    # keep only parameters that still exist
    vae.load_state_dict({k: v for k, v in sd.items() if k in vae.state_dict()}, strict=False)
    vae.eval()
    return vae, vocab, cfg


def encode_latex(latex_str, vocab, max_len, device):
    pad = vocab["<pad>"]
    ids = [vocab["<sos>"]] + [vocab.get(ch, pad) for ch in latex_str] + [vocab["<eos>"]]

    if len(ids) > max_len:        # truncate
        ids = ids[:max_len]
        ids[-1] = vocab["<eos>"]
    else:                         # pad
        ids += [pad] * (max_len - len(ids))

    return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)


def greedy_decode(vae, memory, src_key_padding_mask,
                  sos_idx, eos_idx, pad_idx):
    """
    Autoregressive greedy decoding given **frozen encoder memory**.
    Returns list of token-ids from <sos> up to and incl. <eos>.
    """
    max_len = vae.max_len
    gen = torch.full((1, max_len), pad_idx, dtype=torch.long,
                     device=memory.device)
    gen[0, 0] = sos_idx

    with torch.no_grad():
        for i in range(1, max_len):
            prefix = gen[:, :i]                                       # (1, i)
            logits = vae.decode(memory, prefix, pad_idx,              # (1, i, V)
                                 src_key_padding_mask)
            next_id = logits[0, -1].argmax().item()
            gen[0, i] = next_id
            if next_id == eos_idx:
                return gen[0, :i+1].tolist()

    return gen[0].tolist()  # no EOS produced


def tokens_to_string(tokens, inv_vocab, sos_idx, eos_idx):
    chars = []
    for t in tokens:
        if t == sos_idx: continue
        if t == eos_idx: break
        chars.append(inv_vocab.get(t, ""))
    return "".join(chars)

# ───────────────────────── main ─────────────────────────
def main():
    p = argparse.ArgumentParser("LaTeX ↔ VAE round-trip")
    p.add_argument("--input",  required=True)
    p.add_argument("--vae",    default="best_latex_vae.pth")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)

    vae, vocab, cfg = load_vae(args.vae, device)
    print("VAE loaded – latent_dim:", cfg["latent_dim"],
          " max_len:", cfg["max_len"])
    print("vocab size:", len(vocab))

    # special tokens & helpers
    pad_idx = vocab["<pad>"]
    sos_idx = vocab["<sos>"]
    eos_idx = vocab["<eos>"]
    inv_vocab = {i: ch for ch, i in vocab.items()}

    # 1) encode LaTeX to token ids
    src = encode_latex(args.input, vocab, cfg["max_len"], device)

    # 2) run encoder  (⚠️ pad_idx now passed!)
    memory, mu, logvar, src_mask = vae.encode(src, pad_idx)

    # 3) sample latent & inject into memory
    with torch.no_grad():
        z = vae.reparameterize(mu, logvar)                           # (1, latent_dim)
        latent = vae.latent_to_dec(z).unsqueeze(0)                   # (1, 1, d_model)
        memory = memory + latent.expand_as(memory)                   # broadcast

    # 4) greedy decode
    out_ids = greedy_decode(vae, memory, src_mask,
                            sos_idx, eos_idx, pad_idx)
    out_str = tokens_to_string(out_ids, inv_vocab, sos_idx, eos_idx)
    print(" →", out_str)

    # 5) write result
    with open("output.tex", "w") as f:
        f.write(out_str)
    print("Saved output.tex")

    # 6) simple round-trip self-check
    if len(args.input) < cfg["max_len"] - 2:
        re_src = encode_latex(out_str, vocab, cfg["max_len"], device)
        assert re_src[0, 0] == sos_idx and re_src[0, len(out_str)+1] == eos_idx
        print("round-trip check ✅")

if __name__ == "__main__":
    main()
