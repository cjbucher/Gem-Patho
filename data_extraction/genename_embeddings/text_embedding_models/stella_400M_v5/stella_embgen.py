#!/usr/bin/env python
import json
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# -------------------------------
# 1. Set up the device (GPU if available)
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------
# 2. Load gene descriptions from JSON
# -------------------------------
# Path to your gene descriptions file
json_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/genept/GenePT_embedding_v2/NCBI_UniProt_summary_of_genes.json"
with open(json_path, "r", encoding="utf-8") as f:
    gene_descriptions = json.load(f)

# Filter out genes that have empty or missing descriptions
filtered_descriptions = {gene: desc for gene, desc in gene_descriptions.items() if desc}
print(f"Loaded descriptions for {len(filtered_descriptions)} genes.")

# -------------------------------
# 3. Load the Stella model via SentenceTransformer
# -------------------------------
# Here we use the Stella model "dunzhang/stella_en_400M_v5" with trust_remote_code enabled.
model_name = "dunzhang/stella_en_400M_v5"
model = SentenceTransformer(
    "dunzhang/stella_en_400M_v5",
    trust_remote_code=True,
    device="cuda",
    config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
)
print(f"Loaded model {model_name} on {device}.")

# -------------------------------
# 4. Encode gene descriptions
# -------------------------------
# Since gene descriptions are considered documents (as in the Stella example docs), we do not supply a prompt.
gene_names = list(filtered_descriptions.keys())
descriptions = [filtered_descriptions[gene] for gene in gene_names]

print("Encoding gene descriptions...")
# Adjust batch_size if needed based on your GPU memory.
embeddings = model.encode(
    descriptions,
    batch_size=64,
    show_progress_bar=True
)

# Create a dictionary mapping each gene to its embedding vector.
gene_embeddings = dict(zip(gene_names, embeddings))
print(f"Computed embeddings for {len(gene_embeddings)} genes.")

# -------------------------------
# 5. Save the embeddings to a Parquet file
# -------------------------------
# Convert the dictionary to a DataFrame where each row index is a gene and columns are the embedding dimensions.
df_embs = pd.DataFrame.from_dict(gene_embeddings, orient="index")
print("Embedding DataFrame shape:", df_embs.shape)

# Save the DataFrame as a Parquet file in your target directory.
output_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/genept/cancer_geneset/stella_400M_v5/embeddings.parquet"
df_embs.to_parquet(output_path)
print("Saved embeddings to", output_path)