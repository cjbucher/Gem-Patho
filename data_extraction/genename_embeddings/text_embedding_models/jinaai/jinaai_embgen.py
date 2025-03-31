#!/usr/bin/env python
import json
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# -------------------------------
# 1. Set up device (GPU if available)
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------
# 2. Load gene descriptions from JSON
# -------------------------------
desc_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/genept/GenePT_embedding_v2/NCBI_UniProt_summary_of_genes.json"
with open(desc_path, "r", encoding="utf-8") as f:
    gene_descriptions = json.load(f)

# Filter out genes with empty/missing descriptions
filtered_descriptions = {gene: desc for gene, desc in gene_descriptions.items() if desc}
print(f"Loaded descriptions for {len(filtered_descriptions)} genes.")

# -------------------------------
# 3. Load the SentenceTransformer model
# -------------------------------
# This will load the model with the custom code from the remote repository.
model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
model = model.to(device)  # send model to GPU if available

# -------------------------------
# 4. Encode gene descriptions in batch
# -------------------------------
# We'll use the task "retrieval.query" as in your snippet.
task = "separation"

# Prepare lists of genes and their descriptions.
gene_names = list(filtered_descriptions.keys())
descriptions = [filtered_descriptions[gene] for gene in gene_names]

print("Encoding gene descriptions...")
# Encode in batches to speed things up and reduce GPU memory load.
embeddings = model.encode(
    descriptions,
    task=task,
    prompt_name=task,
    batch_size=64,          # adjust batch size if needed
    show_progress_bar=True  # display a progress bar
)

# Create a dictionary mapping gene names to embeddings.
gene_embeddings = dict(zip(gene_names, embeddings))
print(f"Computed embeddings for {len(gene_embeddings)} genes.")

# -------------------------------
# 5. Save embeddings to a Parquet file
# -------------------------------
# Convert the dictionary to a pandas DataFrame.
df_embs = pd.DataFrame.from_dict(gene_embeddings, orient="index")
print("Embedding DataFrame shape:", df_embs.shape)

# Define the output path (ensure the directory exists)
output_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/genept/cancer_geneset/jinaai/embeddings.parquet"
df_embs.to_parquet(output_path)
print("Saved embeddings to", output_path)
