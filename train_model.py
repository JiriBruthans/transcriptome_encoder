import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
import cellxgene_census
import warnings
import scanpy as sc
import time
warnings.filterwarnings("ignore")



################################################################################
########################## Model definition ####################################
################################################################################

# Model configuration
batch_size = 64
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
model.to(device)
print(f'Using device: {device}')
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')



census = cellxgene_census.open_soma()

gene_names = torch.load('gene_names.pt')

adata = sc.read_h5ad('human_tongue.h5ad')


mask = ~adata.var["feature_name"].str.contains("ENSG")
adata = adata[:, mask]
mask = adata.var["feature_name"].isin(gene_names)
adata = adata[:, mask]

i = 0
data = torch.from_numpy(adata.X.toarray()).to(device)
cell_embs = []
total_time = 0
total_tokens = 0

with torch.no_grad():
    while i < data.shape[0]:
        start_time = time.time()
        
        if (i + batch_size) < data.shape[0]:
            batch = data[i:i+batch_size, :]
            current_batch_size = batch_size
        else:
            batch = data[i:, :]
            current_batch_size = data.shape[0] - i
            
        batch = torch.log1p(batch)
        batch = batch / torch.sum(batch, dim=1, keepdim=True)
        batch = torch.multinomial(batch, 1023, replacement=True)
        
        batch = model.pe_embedding(batch)
        batch = batch.permute(1, 0, 2)
        
        cell_emb = model(batch)
        cell_embs.append(cell_emb)
        
        end_time = time.time()
        batch_time = end_time - start_time
        total_time += batch_time
        tokens_processed = current_batch_size * 1024  # Including CLS token
        total_tokens += tokens_processed
        
        print(f'Processed {i}/{data.shape[0]} cells')
        print(f'Batch processing time: {batch_time:.3f}s')
        print(f'Tokens per second: {tokens_processed/batch_time:.2f}')
        
        i += batch_size

    cell_embs = torch.cat(cell_embs, dim=0)
    print(f'Final embeddings shape: {cell_embs.shape}')
    print(f'Total processing time: {total_time:.2f}s')
    print(f'Average tokens per second: {total_tokens/total_time:.2f}')
    torch.save(cell_embs.cpu(), 'cell_emb.pt')
