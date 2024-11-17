#! /bin/bash

export TMPDIR=$SCRATCHDIR
cd $SCRATCHDIR

# Clone repository
git clone https://github.com/JiriBruthans/transcriptome_encoder.git
cd transcriptome_encoder
#copy all necessary files
cp /storage/praha1/home/jiribruthans/sota/transcriptome_encoder/embedding_layer.pt .
cp /storage/praha1/home/jiribruthans/sota/transcriptome_encoder/gene_names.pt .
cp /storage/praha1/home/jiribruthans/sota/transcriptome_encoder/human_tongue.h5ad .

# Download and install Miniforge non-interactively
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh 
mv Miniforge3-Linux-x86_64.sh ../
chmod +x ../Miniforge3-Linux-x86_64.sh
../Miniforge3-Linux-x86_64.sh -b -p $SCRATCHDIR/miniforge3

# Initialize shell for conda
eval "$($SCRATCHDIR/miniforge3/bin/conda shell.bash hook)"

# Create and activate environment
mamba env create -f environment.yml
conda activate txn-enc
