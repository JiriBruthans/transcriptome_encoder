import math
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch
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
n_genes = 19565
n_loss_total = 1024
n_loss = n_loss_total // 2

TOKEN_DIM = 5120  # ESM-2 embedding dimension
D_MODEL = 1280    # Transformer embedding dimension (output and inner)
N_HEAD = 20       # Number of attention heads
D_HID = 5120     # Hidden dimension
N_LAYERS = 4      # Number of transformer layers
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
                 nlayers: int, dropout: float = 0.05, n_loss: int = 512):
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


        self.token_dim = token_dim
        self.n_loss = n_loss
        self.pe_embedding = nn.Embedding.from_pretrained(torch.load('embedding_layer.pt'))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model))
        self.pe_embedding.requires_grad_(False)

        self.binary_decoder = nn.Sequential(
            full_block(2 * self.d_model, 2048, self.dropout),
            full_block(2048, 512, self.dropout),
            full_block(512, 128, self.dropout),
            nn.Linear(128, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, src: Tensor, g_plus: Tensor, g_minus: Tensor):
        """
        Args:
            src: Tensor, shape [set_size, batch_size, token_dim] -- torch.Size([32, 1023, 5120])
            counts: Tensor, shape [batch_size, n_genes]
        Returns:
            output Tensor of shape [batch_size, d_model]
        """
        
        src = self.encoder(src) * math.sqrt(self.d_model)
        # Expand cls_token to match batch size
        batch_size = src.size(1)
        cls_tokens = self.cls_token.expand(-1, batch_size, -1)
        # Concatenate cls token to the front of the sequence
        src = torch.cat((cls_tokens, src), dim=0)
        output = self.transformer_encoder(src)
        embedding = output[0, :, :] # select only the CLS token
        #import code; code.interact(local=locals())
        embedding = nn.functional.normalize(embedding, dim=1) # Normalize
        
        #loss calculation
        g_plus = g_plus.to(device)
        g_minus = g_minus.to(device)
        g_plus = self.encoder(g_plus) * math.sqrt(self.d_model)
        g_minus = self.encoder(g_minus) * math.sqrt(self.d_model)
        g_plus = torch.cat((g_plus, embedding.expand(self.n_loss, batch_size, self.d_model)), dim=2)
        g_minus = torch.cat((g_minus, embedding.expand(self.n_loss, batch_size, self.d_model)), dim=2)
        
        g_plus = self.binary_decoder(g_plus)
        g_minus = self.binary_decoder(g_minus)  

        loss = None
        bce_loss = torch.nn.BCEWithLogitsLoss()
        loss = bce_loss(g_plus, torch.ones(self.n_loss, batch_size, 1).cuda())
        loss += bce_loss(g_minus, torch.zeros(self.n_loss, batch_size, 1).cuda())
        loss = loss / 2

        return embedding, loss
################################################################################
########################## Model initialization ################################
################################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#use TensorFloat32 for faster computation
torch.set_float32_matmul_precision('high')

model = TransformerModel(
    token_dim=TOKEN_DIM,
    d_model=D_MODEL,
    nhead=N_HEAD,
    d_hid=D_HID,
    nlayers=N_LAYERS,
    dropout=DROPOUT,
    n_loss=n_loss
)


model.to(device)
print(f'Using device: {device}')

#compile the model for faster training
model = torch.compile(model)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f'Total parameters: {total_params:,}')
print(f'Trainable parameters: {trainable_params:,}')

def forward(batch, current_batch_size):
    #randomly select n_loss expressed and n_loss not eexpressed genes for loss calculation
    batch4loss = batch.clone()
    batch4loss[batch4loss != 0] = 1

    g_plus = torch.multinomial(batch4loss, n_loss, replacement=True)
    g_minus = torch.multinomial(1 - batch4loss, n_loss, replacement=True)
    g_plus = model.pe_embedding(g_plus)
    g_minus = model.pe_embedding(g_minus)
    g_plus = g_plus.permute(1, 0, 2)
    g_minus = g_minus.permute(1, 0, 2)

    batch = torch.log1p(batch)
    batch = batch / torch.sum(batch, dim=1, keepdim=True)

    batch = torch.multinomial(batch, (set_size -1), replacement=True)
    
    batch = model.pe_embedding(batch)
    batch = batch.permute(1, 0, 2)

    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        cell_emb, loss = model(batch, g_plus, g_minus)
    return cell_emb, loss


################################################################################
########################## Model training #####################################
################################################################################    
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

adata = sc.read_h5ad('human_tongue.h5ad')
data = torch.from_numpy(adata.X.toarray()).to(device)
cell_embs = []
total_time = 0
total_tokens = 0
n = 0


for epoch in range(1):
    i = 0  # Reset i at the start of each epoch
    #shuffle the data
    data = data[torch.randperm(data.size(0))]
    while i < data.shape[0]:
        start_time = time.time()
        if (i + batch_size) < data.shape[0]:
            batch = data[i:i+batch_size, :]
            current_batch_size = batch_size
        else:
            batch = data[i:, :]
            current_batch_size = data.shape[0] - i
        cell_emb, loss = forward(batch, current_batch_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n += 1
        torch.cuda.synchronize()
        end_time = time.time()
        batch_time = end_time - start_time
        cells_per_second = current_batch_size / batch_time
        
        # Print and log the loss
        print(f"loss: {loss.item()}, epoch {epoch}, step {n}, cells {i} to {i+batch_size} of {data.shape[0]}")
        print(f"Processing speed: {cells_per_second:.2f} cells/second")
        with open('logs/loss_log_1L_shuffled_epoch.csv', 'a') as f:
            f.write(f"{epoch},{n},{loss.item()},{cells_per_second}\n")

        n += 1        
        i += batch_size

#sample the whole dataset
data = torch.from_numpy(adata.X.toarray()).to(device)
with torch.no_grad():   
    i = 0
    while i < data.shape[0]:
        batch = data[i:i+batch_size, :]
        current_batch_size = batch_size
        cell_emb, loss = forward(batch, current_batch_size)
        cell_embs.append(cell_emb)
        i += batch_size

cell_embs = torch.cat(cell_embs, dim=0)
torch.save(cell_embs.cpu(), 'cell_emb_1L_shuffled_epoch.pt')
torch.save(model.state_dict(), 'model_state_dict_1L_shuffled_epoch.pt')
