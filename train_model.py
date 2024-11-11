import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import cellxgene_census
import warnings
import scanpy as sc
warnings.filterwarnings("ignore")



################################################################################
########################## Model definition ####################################
################################################################################

# Model configuration
batch_size = 1
set_size = 1024

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

        self.pe_embedding = nn.Embedding.from_pretrained(torch.load('embedding_layer.pt'))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pe_embedding.requires_grad_(False)

    def forward(self, src: Tensor):
        """
        Args:
            src: Tensor, shape [set_size, batch_size, token_dim] -- torch.Size([32, 1023, 5120])
        Returns:
            output Tensor of shape [set_size, batch_size, ntoken]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        # Expand cls_token to match batch size
        batch_size = src.size(1)
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)
        # Concatenate cls token to the front of the sequence
        src = torch.cat((cls_tokens, src), dim=0)
        output = self.transformer_encoder(src)
        embedding = output[0, :, :] # select only the CLS token
        embedding = nn.functional.normalize(embedding, dim=1) # Normalize
        return embedding


    def predict(self, cell_embedding, gene_embeddings):
        gene_embeddings = self.gene_embedding_layer(gene_embeddings)
        dec = self.binary_decoder \
            (torch.hstack((cell_embedding, gene_embeddings)))
        return dec


################################################################################
########################## Model initialization ################################
################################################################################

model = TransformerModel(
    token_dim=TOKEN_DIM,
    d_model=D_MODEL,
    nhead=N_HEAD,
    d_hid=D_HID,
    nlayers=N_LAYERS,
    output_dim=OUTPUT_DIM,
    dropout=DROPOUT
)

model.eval()
model.to('cpu')

# ... after model initialization ...
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')



census = cellxgene_census.open_soma()

gene_names = torch.load('gene_names.pt')

adata = sc.read_h5ad('human_tongue.h5ad')

print(adata)

mask = ~adata.var["feature_name"].str.contains("ENSG")
adata = adata[:, mask]
print(adata)

mask = adata.var["feature_name"].isin(gene_names)
adata = adata[:, mask]
print(adata)


i = 0
data = torch.from_numpy(adata.X.toarray())
while i < data.shape[0]:
    if (i + batch_size) < data.shape[0]:
        batch = data[i:i+batch_size, :]
        batch = torch.log1p(batch)
        batch = batch / torch.sum(batch, dim=1, keepdim=True)
        batch = torch.multinomial(batch, 1023, replacement=True)
    break


batch = model.pe_embedding(batch)
print(batch.shape)

batch = batch.permute(1, 0, 2)

cell_emb = model(batch)
print(cell_emb.shape)
torch.save(cell_emb, 'cell_emb.pt')
