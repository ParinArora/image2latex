import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        # Debug: Print shapes
        # print(f"Input shape: {x.shape}")
        # print(f"PE shape: {self.pe.shape}")
        # print(f"Slice shape: {self.pe[:, :x.size(1)].shape}")
        return x + self.pe[:, :x.size(1)].to(x.device)

class TransformerVAE(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=4, latent_dim=64, max_len=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Project encoder output to latent space
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)
        
        # Project latent vector back to sequence space
        self.latent_to_dec = nn.Linear(latent_dim, d_model)
        
        # Decoder
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output = nn.Linear(d_model, vocab_size)
        
        self.max_len = max_len
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

    def encode(self, x, pad_idx):
        # Create padding mask (True for padding tokens)
        src_key_padding_mask = (x == pad_idx)  # shape: (batch, seq_len)
        
        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Transform to (seq_len, batch, d_model) for transformer
        x = x.transpose(0, 1)
        
        # Get contextualized representations
        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)  # (seq_len, batch, d_model)
        
        # Use the last hidden state for latent space
        last_hidden = memory[-1]  # (batch, d_model)
        mu = self.to_mu(last_hidden)
        logvar = self.to_logvar(last_hidden)
        
        return memory, mu, logvar, src_key_padding_mask

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, memory, tgt, pad_idx, src_key_padding_mask):
        # Create target padding mask (True for padding tokens)
        tgt_key_padding_mask = (tgt == pad_idx)  # shape: (batch, seq_len)
        
        # Embed and add positional encoding to target
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt)
        tgt = self.dropout(tgt)
        
        # Transform to (seq_len, batch, d_model) for transformer
        tgt = tgt.transpose(0, 1)
        
        # Create causal mask (seq_len, seq_len)
        seq_len = tgt.size(0)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=tgt.device), diagonal=1).bool()
        
        # Decode using memory from encoder
        out = self.decoder(
            tgt, 
            memory,
            tgt_mask=causal_mask,  # Causal mask for self-attention
            memory_key_padding_mask=src_key_padding_mask,  # Mask for encoder memory
            tgt_key_padding_mask=tgt_key_padding_mask  # Mask for target padding
        )
        
        # Transform back to (batch, seq_len, d_model)
        out = out.transpose(0, 1)
        
        # Project to vocabulary
        return self.output(out)

    def forward(self, src, tgt, pad_idx):
        # Encode source sequence
        memory, mu, logvar, src_key_padding_mask = self.encode(src, pad_idx)
        
        # Sample from latent space
        z = self.reparameterize(mu, logvar)
        
        # Inject latent information into memory
        latent_memory = self.latent_to_dec(z)  # (batch, d_model)
        latent_memory = latent_memory.unsqueeze(0).expand(memory.size(0), -1, -1)  # (seq_len, batch, d_model)
        memory = memory + latent_memory
        
        # Decode
        logits = self.decode(memory, tgt, pad_idx, src_key_padding_mask)
        
        return logits, mu, logvar

def vae_loss(logits, targets, mu, logvar, pad_idx, sos_idx=1, eos_idx=2, beta=0.01):
    # Ensure tensors are contiguous
    logits = logits.contiguous()
    targets = targets.contiguous()
    
    # Create masks for different token types
    non_special_mask = (targets != pad_idx) & (targets != sos_idx) & (targets != eos_idx)
    eos_mask = (targets == eos_idx)
    
    # Reconstruction loss with ignore_index for padding
    recon_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), 
        targets.reshape(-1),
        ignore_index=pad_idx,  # Ignore padding tokens in loss
        reduction='none'
    ).reshape_as(targets)
    
    # Apply weights only to non-padding tokens
    token_weights = torch.ones_like(targets, dtype=torch.float)
    token_weights[non_special_mask] = 2.0  # Double weight for actual content
    token_weights[eos_mask] = 1.5  # Higher weight for EOS tokens
    
    # Apply weights and take mean (only over non-padding tokens)
    recon_loss = (recon_loss * token_weights * (targets != pad_idx).float()).sum() / (targets != pad_idx).float().sum()
    
    # Length penalty: encourage non-empty sequences
    # Use softmax to get proper token probabilities
    token_probs = F.softmax(logits, dim=-1)  # shape: (batch, seq_len, vocab_size)
    pad_token_probs = token_probs[:, :, pad_idx]  # shape: (batch, seq_len)
    length_penalty = torch.mean(pad_token_probs)
    
    # EOS penalty: encourage proper sequence termination
    eos_token_probs = token_probs[:, :, eos_idx]  # shape: (batch, seq_len)
    # We want exactly one EOS token near the end of non-padding sequence
    eos_penalty = torch.mean((eos_token_probs - 0.5).pow(2))
    
    # Diversity penalty: encourage diverse token usage
    # Calculate token distribution entropy
    token_entropy = -torch.sum(token_probs * torch.log(token_probs + 1e-10), dim=-1)  # shape: (batch, seq_len)
    diversity_penalty = -torch.mean(token_entropy)  # negative because we want to maximize entropy
    
    # KL divergence loss
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)
    
    # Add L2 regularization to prevent collapse
    l2_reg = 0.01 * (mu.pow(2).mean() + logvar.pow(2).mean())
    
    # Combine all losses
    total_loss = (
        recon_loss + 
        beta * kl_loss + 
        l2_reg + 
        0.1 * length_penalty +  # Penalize empty sequences
        0.1 * eos_penalty +    # Encourage proper EOS placement
        0.1 * diversity_penalty  # Encourage diverse outputs
    )
    
    return total_loss, recon_loss, kl_loss

class ImageEncoder(nn.Module):
    def __init__(self, latent_dim=64, backbone='resnet18', pretrained=True):
        super().__init__()
        # Use a ResNet backbone, remove the final classification layer
        if backbone == 'resnet18':
            resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            feature_dim = resnet.fc.in_features
            modules = list(resnet.children())[:-1]  # Remove the last FC layer
            self.cnn = nn.Sequential(*modules)
        else:
            raise NotImplementedError("Only resnet18 is implemented for now.")
        self.to_mu = nn.Linear(feature_dim, latent_dim)
        self.to_logvar = nn.Linear(feature_dim, latent_dim)

    def forward(self, x):
        # x: (batch, 3, H, W)
        features = self.cnn(x)  # (batch, feature_dim, 1, 1)
        features = features.view(features.size(0), -1)  # (batch, feature_dim)
        mu = self.to_mu(features)
        logvar = self.to_logvar(features)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
