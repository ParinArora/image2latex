import sys
import torch

def main(checkpoint_path):
    # Load checkpoint (works for CPU even if it was saved from GPU)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    
    # If it's a plain tensor or number
    if isinstance(ckpt, (float, int, torch.Tensor)):
        print("Loaded object is a single value:", ckpt)
        return

    # If it's a dict, look for common loss keys
    if isinstance(ckpt, dict):
        # print all keys so you know what's inside
        print("Keys in checkpoint:", list(ckpt.keys()))
        
        # common names people use for loss
        for key in ("loss", "total_loss", "train_loss", "val_loss"):
            if key in ckpt:
                print(f"{key} =", ckpt[key])
        else:
            print("No standard loss key found.  You might inspect manually:")
            for k, v in ckpt.items():
                # show small items
                if isinstance(v, (float, int, torch.Tensor)):
                    print(f"  {k} → {v}")
    else:
        print("Checkpoint is neither a dict nor a scalar. It's a:", type(ckpt))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_loss.py path/to/checkpoint.pth")
        sys.exit(1)
    main(sys.argv[1])
