import torch
import torch.nn as nn
import math

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_size)
        self.attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(embed_size)
        self.mlp = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask, padding_mask):
        # Self-attention with residual and layer norm
        norm_x = self.ln1(x)
        attn_output, _ = self.attn(
            norm_x, norm_x, norm_x,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
            is_causal=True
        )
        x = x + self.dropout1(attn_output)

        # Feedforward network with residual and layer norm
        norm_x = self.ln2(x)
        ff_output = self.mlp(norm_x)
        x = x + self.dropout2(ff_output)
        return x



class PositionalEncoding(nn.Module):
    """
    Positional encoding module: adds positional information to the input embeddings.
    """
    def __init__(self, embed_size, max_len):
        super().__init__()

        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        pe = torch.zeros(max_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)  # even positions
        pe[:, 1::2] = torch.cos(position * div_term)  # odd positions

        pe = pe.unsqueeze(0)  # shape: (1, max_len, embed_size)
        self.register_buffer("positional_encoding", pe)  # non-trainable

    def forward(self, x):
        # x shape: (batch_size, seq_len, embed_size)
        seq_len = x.size(1)
        pe = self.positional_encoding[:, :seq_len, :].to(x.device)
        return x + pe

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_size = config.embed_size
        self.num_layers = config.num_layers 
        self.vocab_size = config.vocab_size
        self.max_len = config.max_len
        self.dropout_p = config.dropout_p
        self.num_heads = config.num_heads
        self.device = config.device

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.pos_encoder = PositionalEncoding(self.embed_size, self.max_len)

        self.layers = nn.ModuleList([DecoderBlock(self.embed_size, self.num_heads, self.dropout_p) for _ in range(self.num_layers)])
        self.fc_out = nn.Linear(self.embed_size, self.vocab_size)

        # Precompute the causal mask and positional encoding
        self.register_buffer("causal_mask", self.generate_causal_mask(self.max_len))

    def forward(self, x, padding_mask=None):
        batch_size, seq_len = x.shape

        # Use the precomputed causal mask (trim to match seq_len)
        attn_mask = self.causal_mask[:seq_len, :seq_len]

        # Embed and add positional encoding
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attn_mask, padding_mask)

        return self.fc_out(x)

    def generate_causal_mask(self, seq_len):
        """
        Generates an upper triangular mask to prevent attending to future tokens.
        """
        return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()


if __name__ == "__main__":
    from tokenizers import Tokenizer
    from torch.nn.functional import cross_entropy

    from config import config
    from utils import get_num_params
    from dataset import QADataset

    model = TransformerModel(config)
    print(f"Number of parameters in the model: {get_num_params(model):,}")

    # Simple forward pass for sanity checking
    tokenizer = Tokenizer.from_file(config.tokenizer_filename)
    dataset = QADataset(config, tokenizer)
    source = dataset[0]["source_sequence"].unsqueeze(0)
    target = dataset[0]["target_sequence"].unsqueeze(0)
    padding_mask = dataset[0]["key_padding_mask"].unsqueeze(0)

    # Forward pass
    out = model(source, padding_mask)
    print("Output shape:", out.shape)
    print("Target shape:", target.shape)
    print("Loss mask shape:", padding_mask.shape)

    # Calculate loss
    loss = cross_entropy(out.transpose(1, 2), target)
    print("Loss:", loss.item())

