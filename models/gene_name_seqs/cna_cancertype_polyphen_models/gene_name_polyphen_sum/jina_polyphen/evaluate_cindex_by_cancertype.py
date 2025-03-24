#!/usr/bin/env python
"""
Script to evaluate the trained transformer survival model and compute the 
concordance index (c-index) for each cancer type on the test set.

This script re-uses the PreprocessedSequenceDataset, collate function, and 
PreprocessedTransformerSurvivalModel definitions from your new training script. 
It additionally returns the original cancer type (acronym) for each sample so that 
the c-index can be computed for each cancer type separately.

Usage:
    $ python evaluate_cindex_by_cancer_type.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
import random

# -----------------------------
# Set Seeds for Reproducibility
# -----------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

##########################################
# Constants & Paths
##########################################
# Modify these as needed.
FOLD = 1
INPUT_DIR = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold"
MODEL_DIR = "/home/chb3333/yulab/chb3333/gem-patho/models/gene_name_seqs/cna_cancertype_polyphen_models/gene_name_polyphen_sum/jina_polyphen"
TEST_PATH = os.path.join(INPUT_DIR, f"fold_{FOLD}", "test.parquet")
MODEL_PATH = os.path.join(MODEL_DIR, f"fold_{FOLD}", f"best_model_fold_{FOLD}.pth")

# Path to your CSV mapping each cancer type (Study Abbreviation) to an index
CANCER_TYPE_MAPPING_CSV = (
    "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/"
    "cancertype_location_description/tcga_study_abbreviations.csv"
)

##########################################
# Cancer Type Mapping
##########################################
def int_to_binary_vector(x, width=6):
    """Convert integer x to a fixed-width binary vector of length `width`."""
    return [int(b) for b in format(x, f"0{width}b")]

df_ct = pd.read_csv(CANCER_TYPE_MAPPING_CSV)
unique_types = sorted(df_ct["Study Abbreviation"].unique())
cancer_type_mapping = {ct: int_to_binary_vector(i, 6) for i, ct in enumerate(unique_types)}
print("Cancer type mapping:", cancer_type_mapping)

##########################################
# Dataset for Preprocessed Sequences
# (same logic as in your new training code, but returns acronym too)
##########################################
class PreprocessedSequenceDataset(Dataset):
    def __init__(self, df, token_col="gene_embed_seq", cancer_type_mapping=None):
        """
        Expects a DataFrame with columns:
          - token_col: list of tokens (dicts with at least "embedding", "score", and "cna")
          - "OS.time": survival time
          - "OS": event indicator
          - "type": e.g., "BRCA", "LAML", etc.
        """
        self.df = df.reset_index(drop=True)
        self.token_col = token_col
        self.cancer_type_mapping = cancer_type_mapping if cancer_type_mapping is not None else {}
        
        # Infer gene embedding dimension from the first non-empty token list.
        self.genename_dim = None
        for idx in range(len(self.df)):
            tokens = self.df.iloc[idx][token_col]
            if isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            if tokens and len(tokens) > 0:
                self.genename_dim = len(tokens[0]["embedding"])
                break
        if self.genename_dim is None:
            raise ValueError("Could not determine gene embedding dimension from data.")

        # Count rows with empty token lists.
        self.default_token_count = 0
        for idx in range(len(self.df)):
            tokens = self.df.iloc[idx][token_col]
            if isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            if not tokens or len(tokens) == 0:
                self.default_token_count += 1
        total_samples = len(self.df)
        default_percentage = (self.default_token_count / total_samples) * 100
        print(f"Samples with default tokens: {self.default_token_count} "
              f"({default_percentage:.2f}% of {total_samples} samples)")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = row[self.token_col]
        
        # Convert any NumPy arrays for tokens into Python lists
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        
        # If token list is empty, supply a default token
        if not tokens or len(tokens) == 0:
            tokens = [
                {"gene": "", "embedding": [0.0]*self.genename_dim, "score": 0.0, "cna": 0.0}
            ]
        
        # Break out embeddings, polyphen scores, CNAs
        embeddings = [torch.tensor(token["embedding"], dtype=torch.float) for token in tokens]
        scores = [torch.tensor(token["score"], dtype=torch.float) for token in tokens]
        cnas = [torch.tensor(token.get("cna", 0.0), dtype=torch.float) for token in tokens]
        
        # Convert the cancer type acronym to the 6-d binary vector
        ct_acronym = row.get("type", None)
        if ct_acronym is None or ct_acronym not in self.cancer_type_mapping:
            ct_vector = [0]*6
        else:
            ct_vector = self.cancer_type_mapping[ct_acronym]
        cancer_type_tensor = torch.tensor(ct_vector, dtype=torch.float)
        
        time = torch.tensor(row["OS.time"], dtype=torch.float)
        event = torch.tensor(row["OS"], dtype=torch.float)
        
        # Return the raw acronym too (for grouping c-index by type)
        return embeddings, scores, cnas, cancer_type_tensor, time, event, ct_acronym

##########################################
# Collate Function for Padding
# (now returns the list of cancer acronyms)
##########################################
def collate_fn_preprocessed(batch):
    # batch is a list of tuples (embeddings, scores, cnas, cancer_type_tensor, time, event, ct_acronym)
    emb_list, score_list, cna_list, ct_tensors, times, events, ct_acronym_list = zip(*batch)
    
    # Pad gene embeddings
    padded_emb = torch.nn.utils.rnn.pad_sequence(
        [torch.stack(seq) for seq in emb_list],
        batch_first=True,
        padding_value=0.0
    )
    # Pad polyphen scores
    padded_scores = torch.nn.utils.rnn.pad_sequence(
        [torch.stack(seq) for seq in score_list],
        batch_first=True,
        padding_value=0.0
    )
    # Pad CNA values
    padded_cnas = torch.nn.utils.rnn.pad_sequence(
        [torch.stack(seq) for seq in cna_list],
        batch_first=True,
        padding_value=0.0
    )
    # Stack cancer type vectors
    cancer_types = torch.stack(ct_tensors)
    
    lengths = [len(seq) for seq in emb_list]
    B = len(batch)
    L_max = padded_emb.shape[1]
    mask = torch.zeros((B, L_max), dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l < L_max:
            mask[i, l:] = True  # Mark padded positions.
    
    times = torch.stack(times)
    events = torch.stack(events)
    
    # Return also the list of acronyms for grouping
    return (
        padded_emb,
        padded_scores,
        padded_cnas,
        cancer_types,
        times,
        events,
        mask,
        list(ct_acronym_list)
    )

##########################################
# Transformer Survival Model (same as training code)
##########################################
class PreprocessedTransformerSurvivalModel(nn.Module):
    def __init__(self, d_gene, d_model=256, polyphen_hidden_dim=128, nhead=4, dropout=0.1):
        super().__init__()
        self.gene_linear = nn.Linear(d_gene, d_model)
        self.polyphen_mlp = nn.Sequential(
            nn.Linear(1, polyphen_hidden_dim),
            nn.GELU(),
            nn.Linear(polyphen_hidden_dim, d_model)
        )
        self.cna_mlp = nn.Sequential(
            nn.Linear(1, polyphen_hidden_dim),
            nn.GELU(),
            nn.Linear(polyphen_hidden_dim, d_model)
        )
        self.cancer_type_mlp = nn.Sequential(
            nn.Linear(6, polyphen_hidden_dim),
            nn.GELU(),
            nn.Linear(polyphen_hidden_dim, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dropout=dropout, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.final_linear = nn.Linear(d_model, 1)
        
    def forward(self, emb, scores, cnas, cancer_type, src_key_padding_mask=None):
        """
        emb: (B, L, d_gene)
        scores: (B, L)
        cnas: (B, L)
        cancer_type: (B, 6)
        src_key_padding_mask: (B, L) of booleans (True = pad)
        """
        # Project gene embeddings
        gene_proj = self.gene_linear(emb)  # (B, L, d_model)
        # Project polyphen
        polyphen_proj = self.polyphen_mlp(scores.unsqueeze(-1))  # (B, L, d_model)
        # Project CNA
        cna_proj = self.cna_mlp(cnas.unsqueeze(-1))  # (B, L, d_model)
        # Project cancer type (B, 6) -> (B, d_model) -> unsqueeze to (B, 1, d_model)
        cancer_type_proj = self.cancer_type_mlp(cancer_type).unsqueeze(1)
        
        # Sum them up
        token_emb = gene_proj + polyphen_proj + cna_proj + cancer_type_proj
        
        # Transformer expects (L, B, d_model)
        token_emb = token_emb.transpose(0, 1)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(token_emb.device)
        
        transformer_out = self.transformer_encoder(token_emb, src_key_padding_mask=src_key_padding_mask)
        # Now (L, B, d_model) -> (B, L, d_model)
        transformer_out = transformer_out.transpose(0, 1)
        
        # Mean-pool over non-padded tokens
        if src_key_padding_mask is not None:
            valid_counts = (~src_key_padding_mask).sum(dim=1, keepdim=True).float()
            mask_expanded = (~src_key_padding_mask).unsqueeze(-1).float()
            pooled = (transformer_out * mask_expanded).sum(dim=1) / valid_counts
        else:
            pooled = transformer_out.mean(dim=1)
        
        risk = self.final_linear(pooled)  # (B, 1)
        return risk

##########################################
# Evaluate a Single DataLoader by Cancer Type
##########################################
def evaluate_by_cancer_type(model, dataloader, device):
    """
    Compute risk scores for each batch, then group by cancer type acronym
    to compute the c-index for each type separately.
    Returns a dict: {cancer_type: c_index, ...}.
    """
    model.eval()
    all_T = []
    all_E = []
    all_risk = []
    all_cancer_types = []
    
    with torch.no_grad():
        for (emb, scores, cnas, ct_tensor, T, E, mask, ct_list) in dataloader:
            emb = emb.to(device)
            scores = scores.to(device)
            cnas = cnas.to(device)
            ct_tensor = ct_tensor.to(device)
            T = T.to(device)
            E = E.to(device)
            mask = mask.to(device)
            
            # Forward pass
            risk = model(
                emb, 
                scores, 
                cnas, 
                ct_tensor, 
                src_key_padding_mask=mask
            )
            
            # Collect data (move CPU for easy handling)
            all_T.append(T.cpu().numpy())
            all_E.append(E.cpu().numpy())
            all_risk.append(risk.cpu().numpy())
            all_cancer_types.extend(ct_list)
    
    # Convert to flat numpy arrays
    all_T = np.concatenate(all_T).squeeze()
    all_E = np.concatenate(all_E).squeeze()
    all_risk = np.concatenate(all_risk).squeeze()
    
    # Compute c-index per cancer type
    results = {}
    unique_types = np.unique(all_cancer_types)
    for ct in unique_types:
        mask = np.array(all_cancer_types) == ct
        if mask.sum() < 2:
            results[ct] = np.nan
            continue
        T_ct = all_T[mask]
        E_ct = all_E[mask]
        risk_ct = all_risk[mask]
        # If no events, skip
        if E_ct.sum() == 0:
            results[ct] = np.nan
            continue
        try:
            # We often use -risk for c-index in survival tasks (higher risk => quicker failure)
            c_index = concordance_index(T_ct, -risk_ct, E_ct)
            results[ct] = c_index
        except ZeroDivisionError:
            results[ct] = np.nan
    return results

##########################################
# Single-Fold Evaluation Script
##########################################
def main():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load test set
    test_df = pd.read_parquet(TEST_PATH)
    print(f"Loaded test set: {test_df.shape[0]} samples")
    
    # Create dataset/loader
    test_dataset = PreprocessedSequenceDataset(
        df=test_df,
        token_col="gene_embed_seq",
        cancer_type_mapping=cancer_type_mapping
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=collate_fn_preprocessed
    )

    # Infer d_gene from a single batch
    sample_batch = next(iter(test_loader))
    d_gene = sample_batch[0].shape[-1]
    print(f"Inferred d_gene = {d_gene}")

    # Load model
    model = PreprocessedTransformerSurvivalModel(
        d_gene=d_gene,
        d_model=256,
        polyphen_hidden_dim=128,
        nhead=4,
        dropout=0.1
    ).to(device)
    
    # Load trained weights
    print(f"Loading model from {MODEL_PATH}")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Evaluate c-index by cancer type
    cindex_results = evaluate_by_cancer_type(model, test_loader, device)
    
    # Print results
    print("\nC-index by Cancer Type on the Test Set:")
    for ct, cidx in cindex_results.items():
        print(f"  {ct}: {cidx:.4f}" if not np.isnan(cidx) else f"  {ct}: NaN")

    # Optionally, save results
    results_df = pd.DataFrame({
        "Cancer_Type": list(cindex_results.keys()),
        "C_index": list(cindex_results.values())
    })
    out_csv = os.path.join(MODEL_DIR, f"cindex_by_cancertype_fold_{FOLD}.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nSaved results to {out_csv}")


if __name__ == "__main__":
    main()
