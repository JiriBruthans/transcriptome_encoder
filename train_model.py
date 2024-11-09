"""
Model class

"""

import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch

# Model configuration
TOKEN_DIM = 5120  # ESM-2 embedding dimension
D_MODEL = 1280    # Transformer embedding dimension (output and inner)
N_HEAD = 20       # Number of attention heads
D_HID = 5120     # Hidden dimension
N_LAYERS = 4      # Number of transformer layers
OUTPUT_DIM = 1080 # Output dimension
DROPOUT = 0.05    # Dropout rate

def full_block(in_features, out_features, p_drop=0.1):
    return nn.Sequential(
        nn.Linear(in_features, out_features, bias=True),
        nn.LayerNorm(out_features),
        nn.GELU(),
        nn.Dropout(p=p_drop),
    )

class TransformerModel(nn.Module):

    def __init__(self, token_dim: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, output_dim:int, dropout: float = 0.05):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model

        self.encoder = nn.Sequential(nn.Linear(token_dim, d_model),
                                     nn.GELU(),
                                     nn.LayerNorm(d_model))

        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.d_model = d_model
        self.dropout = dropout


        self.decoder = nn.Sequential(full_block(d_model, 1024, self.dropout),
                                     full_block(1024, output_dim, self.dropout),
                                     full_block(output_dim, output_dim, self.dropout),
                                     nn.Linear(output_dim, output_dim)
                                     )

        self.binary_decoder = nn.Sequential(
            full_block(output_dim + 1280, 2048, self.dropout),
            full_block(2048, 512, self.dropout),
            full_block(512, 128, self.dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.gene_embedding_layer = nn.Sequential(nn.Linear(token_dim, d_model),
                                                  nn.GELU(),
                                                  nn.LayerNorm(d_model))

        self.pe_embedding = None

    def forward(self, src: Tensor, mask: Tensor):
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
        Returns:
            output Tensor of shape [seq_len, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        output = self.transformer_encoder(src, src_key_padding_mask=( 1 -mask))
        embedding = gene_output[0, :, :] # select only the CLS token.
        embedding = nn.functional.normalize(embedding, dim=1) # Normalize.
        return embedding


    def predict(self, cell_embedding, gene_embeddings):
        gene_embeddings = self.gene_embedding_layer(gene_embeddings)
        dec = self.binary_decoder \
            (torch.hstack((cell_embedding, gene_embeddings)))
        return dec


model = TransformerModel(
    token_dim=TOKEN_DIM,
    d_model=D_MODEL,
    nhead=NHEAD,
    d_hid=D_HID,
    nlayers=NLAYERS,
    output_dim=OUTPUT_DIM,
    dropout=DROPOUT
)
