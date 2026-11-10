import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x

class TimeSeriesMAE(nn.Module):
    def __init__(self, num_features, d_model=64, num_layers=3, nhead=4, dropout=0.1, mask_ratio=0.5):
        """
        Masked Autoencoder for Time Series.
        
        Args:
            num_features (int): Number of sensors/features in the time series (e.g., 114).
            d_model (int): Dimension of the latent embeddings.
            num_layers (int): Number of Transformer layers in encoder and decoder.
            nhead (int): Number of attention heads.
            dropout (float): Dropout probability.
            mask_ratio (float): Ratio of timesteps to mask (0 to 1).
        """
        super(TimeSeriesMAE, self).__init__()
        self.num_features = num_features
        self.mask_ratio = mask_ratio
        
        # Linear projection from features to d_model
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Learnable mask token (to replace masked timesteps)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Transformer Decoder (can be shallower in practice, but keeping symmetric here)
        decoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        
        # Final projection back to original features
        self.output_projection = nn.Linear(d_model, num_features)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)

    def random_masking(self, x):
        """
        Args:
            x: shape (batch_size, seq_len, d_model)
        Returns:
            x_masked: Sequence with masked tokens inserted.
            mask: Boolean mask indicating which tokens were masked.
        """
        batch_size, seq_len, _ = x.shape
        len_keep = int(seq_len * (1 - self.mask_ratio))
        
        noise = torch.rand(batch_size, seq_len, device=x.device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Generate the binary mask: 1 is masked, 0 is kept
        mask = torch.ones([batch_size, seq_len], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # Apply mask by replacing masked elements with mask_token
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x_masked = x * (1 - mask_expanded) + self.mask_token * mask_expanded
        
        return x_masked, mask

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, num_features)
        Returns:
            x_recon: (batch_size, seq_len, num_features) - reconstructed sequence
            mask: (batch_size, seq_len) - binary mask (1 = masked, 0 = unmasked)
        """
        # 1. Linear Projection & Positional Encoding
        x_emb = self.input_projection(x)
        x_emb = self.pos_encoder(x_emb)
        
        # 2. Masking (only during training to learn physics/representations)
        if self.training:
            x_masked, mask = self.random_masking(x_emb)
        else:
            x_masked = x_emb
            mask = torch.zeros(x.shape[0], x.shape[1], device=x.device)
            
        # 3. Encoding
        encoded = self.encoder(x_masked)
        
        # 4. Decoding
        decoded = self.decoder(encoded)
        
        # 5. Output Projection
        x_recon = self.output_projection(decoded)
        
        return x_recon, mask

    def compute_loss(self, x, x_recon, mask):
        """
        Computes MSE loss. By standard MAE design, loss is only computed on masked patches.
        """
        loss = (x_recon - x) ** 2
        
        if self.training:
            # Mean loss only on masked elements
            loss = (loss.mean(dim=-1) * mask).sum() / (mask.sum() + 1e-8)
        else:
            # During inference/validation, we compute overall reconstruction error
            loss = loss.mean()
            
        return loss
