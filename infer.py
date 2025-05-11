import torch
import torch.nn.functional as F
from AutoEncoder_Latex import TransformerVAE
from train_latex_vae import LatexDataset
import argparse

@torch.no_grad()
def generate(
    vae,
    src,                # (1, seq_len) LongTensor of input token IDs
    pad_idx: int,
    sos_idx: int,
    eos_idx: int,
    max_steps: int = 512,
    temperature: float = 1.0,
    sample: bool = False
):
    """
    Returns a list of token IDs (including sos and eos) decoded greedily (or sampled if sample=True).
    """
    vae.eval()
    device = src.device

    # 1) Encode + get padding mask
    memory, mu, logvar, src_key_padding_mask = vae.encode(src, pad_idx)

    # 2) Reparameterize z and inject into memory
    z = vae.reparameterize(mu, logvar)                              # (1, latent_dim)
    latent_mem = vae.latent_to_dec(z)                                # (1, d_model)
    latent_mem = latent_mem.unsqueeze(0).expand(memory.size(0), -1, -1)
    memory = memory + latent_mem                                    # (seq_len, 1, d_model)

    # 3) Prepare generation buffer
    seq = torch.full((1, vae.max_len), pad_idx, dtype=torch.long, device=device)
    seq[0, 0] = sos_idx

    # 4) Decode one token at a time
    for i in range(1, min(max_steps, vae.max_len)):
        prefix = seq[:, :i]                                          # (1, i)
        # note: decode signature is decode(memory, tgt, pad_idx, src_mask)
        logits = vae.decode(memory, prefix, pad_idx, src_key_padding_mask)  # (1, i, V)
        next_logits = logits[0, -1] / temperature                    # (V,)

        if sample:
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()
        else:
            next_id = torch.argmax(next_logits, dim=-1).item()

        seq[0, i] = next_id

        if next_id == eos_idx:
            return seq[0, :i+1].tolist()

    # if we never saw EOS, return full generated prefix
    return seq[0, :max_steps].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_latex_vae.pth',
                      help='Path to model checkpoint')
    parser.add_argument('--temperature', type=float, default=0.8,
                      help='Temperature for sampling (higher = more random)')
    parser.add_argument('--max_steps', type=int, default=512,
                      help='Maximum number of decoding steps')
    parser.add_argument('--sample', action='store_true',
                      help='Use sampling instead of greedy decoding')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1) load your dataset (for vocab & an example)
    ds = LatexDataset(
        '/home/parin.arora_ug2023/text2latex/PRINTED_TEX_230k/final_png_formulas.txt',
        max_len=512
    )

    # 2) rebuild & load your trained VAE (must match hyperparams!)
    vae = TransformerVAE(
        vocab_size=len(ds.vocab),
        d_model=512, nhead=8, num_layers=6,
        latent_dim=128, max_len=512, dropout=0.2
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    vae.load_state_dict(ckpt['model_state_dict'])
    vae.eval()

    # 3) pick an example and get its latent z
    pad_idx = ds.vocab['<pad>']
    sos_idx = ds.vocab['<sos>']
    eos_idx = ds.vocab['<eos>']

    # Get a batch of examples
    tokens, _ = ds[273]
    src = tokens.unsqueeze(0).to(device)  # shape (1, max_len)
    
    # 4) Generate using the new generate function
    seq = generate(
        vae,
        src,
        pad_idx=pad_idx,
        sos_idx=sos_idx,
        eos_idx=eos_idx,
        max_steps=args.max_steps,
        temperature=args.temperature,
        sample=args.sample
    )

    # 5) convert IDs back to chars
    inv_vocab = {i:t for t,i in ds.vocab.items()}
    decoded = ''.join(inv_vocab[i] for i in seq if i not in {pad_idx, sos_idx, eos_idx})
    print("\nGenerated string:", repr(decoded))
    
    # 6) Print the original string for comparison
    original = ''.join(inv_vocab[i] for i in src[0].tolist() if i not in {pad_idx, sos_idx, eos_idx})
    print("Original string:", repr(original))
