#!/usr/bin/env python
"""
Script to evaluate the trained transformer survival model and compute the 
concordance index (c-index) for each cancer type on the test set.

This script re-uses the PreprocessedSequenceDataset, collate function, and 
PreprocessedTransformerSurvivalModel definitions from the training script. 
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
from torch.utils.data import DataLoader
from lifelines.utils import concordance_index
import random

# Set seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

##########################################
# Constants & Paths
##########################################
# Modify these paths as needed.
FOLD = 1
INPUT_DIR = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold"
MODEL_DIR = "/home/chb3333/yulab/chb3333/gem-patho/models/learning_phases_scheduling/v2/exp_0_weight_2.0_warmup_10_transition_10"
TEST_PATH = os.path.join(INPUT_DIR, f"fold_{FOLD}", "test.parquet")
MODEL_PATH = os.path.join(MODEL_DIR, f"fold_{FOLD}", f"best_model_fold_{FOLD}.pth")
CANCER_TYPE_MAPPING_CSV = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/tcga_study_abbreviations.csv"

##########################################
# Cancer Type Mapping
##########################################
def int_to_binary_vector(x, width=6):
    return [int(b) for b in format(x, f"0{width}b")]

df_ct = pd.read_csv(CANCER_TYPE_MAPPING_CSV)
unique_types = sorted(df_ct["Study Abbreviation"].unique())
cancer_type_mapping = {ct: int_to_binary_vector(i, 6) for i, ct in enumerate(unique_types)}
print("Cancer type mapping:", cancer_type_mapping)

##########################################
# Dataset for Preprocessed Sequences (with cancer type acronym return)
##########################################
class PreprocessedSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, df, token_col="gene_embed_seq", cancer_type_mapping=None):
        """
        Expects a DataFrame with columns:
          - token_col: a list/array of tokens (each token is a dict with keys:
              "gene", "embedding", "score", "cna").
          - "OS.time": survival time.
          - "OS": event indicator.
          - "CANCER_TYPE_ACRONYM": cancer type abbreviation.
          Optionally, if present, "description_embeddings" will be used as a separate token.
        """
        self.df = df.reset_index(drop=True)
        self.token_col = token_col
        self.cancer_type_mapping = cancer_type_mapping if cancer_type_mapping is not None else {}
        self.has_description = "description_embeddings" in self.df.columns

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

        # Count rows with empty token list.
        self.default_token_count = 0
        for idx in range(len(self.df)):
            tokens = self.df.iloc[idx][token_col]
            if isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            if not tokens or (hasattr(tokens, '__len__') and len(tokens) == 0):
                self.default_token_count += 1
        total_samples = len(self.df)
        default_percentage = (self.default_token_count / total_samples) * 100
        print(f"Samples with default tokens: {self.default_token_count} ({default_percentage:.2f}% of {total_samples} samples)")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Get gene tokens.
        tokens = row[self.token_col]
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        if not tokens or (hasattr(tokens, '__len__') and len(tokens) == 0):
            tokens = [{"gene": "", "embedding": [0.0]*self.genename_dim, "score": 0.0, "cna": 0.0}]
        embeddings = [torch.tensor(token["embedding"], dtype=torch.float) for token in tokens]
        scores = [torch.tensor(token["score"], dtype=torch.float) for token in tokens]
        cnas = [torch.tensor(token.get("cna", 0.0), dtype=torch.float) for token in tokens]
        
        # Process cancer type.
        cancer_type_acronym = row.get("type", None)
        if cancer_type_acronym is None or cancer_type_acronym not in self.cancer_type_mapping:
            ct_vector = [0]*6
        else:
            ct_vector = self.cancer_type_mapping[cancer_type_acronym]
        cancer_type_tensor = torch.tensor(ct_vector, dtype=torch.float)
        
        # Process description embeddings if present.
        if self.has_description:
            description = torch.tensor(row["description_embeddings"], dtype=torch.float)
        else:
            description = torch.zeros(self.genename_dim, dtype=torch.float)
        
        time = torch.tensor(row["OS.time"], dtype=torch.float)
        event = torch.tensor(row["OS"], dtype=torch.float)
        # Also return the original cancer type acronym for grouping later.
        return embeddings, scores, cnas, cancer_type_tensor, description, time, event, cancer_type_acronym

##########################################
# Collate Function for Padding (returning cancer type acronyms)
##########################################
def collate_fn_preprocessed(batch):
    emb_list, score_list, cna_list, cancer_type_list, desc_list, times, events, cancer_acronym_list = zip(*batch)
    padded_emb = torch.nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in emb_list],
                                                  batch_first=True, padding_value=0.0)
    padded_scores = torch.nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in score_list],
                                                     batch_first=True, padding_value=0.0)
    padded_cnas = torch.nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in cna_list],
                                                  batch_first=True, padding_value=0.0)
    cancer_types = torch.stack(cancer_type_list)  # (B, 6)
    descriptions = torch.stack(desc_list)         # (B, desc_dim)
    lengths = torch.tensor([len(seq) for seq in emb_list], dtype=torch.long)
    B, L_max, _ = padded_emb.shape
    # Create padding mask for gene tokens.
    mask = torch.zeros((B, L_max), dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l < L_max:
            mask[i, l:] = True  # Mark padded positions.
    times = torch.stack(times)
    events = torch.stack(events)
    return padded_emb, padded_scores, padded_cnas, cancer_types, descriptions, times, events, mask, list(cancer_acronym_list)

##########################################
# Transformer Survival Model (same as training)
##########################################
class PreprocessedTransformerSurvivalModel(nn.Module):
    def __init__(self, d_gene, d_model=256, polyphen_hidden_dim=128, nhead=4, dropout=0.1, desc_dim=None):
        """
        d_gene: dimension of the precomputed gene embedding.
        d_model: token dimension.
        polyphen_hidden_dim: hidden dimension for the polyphen, CNA, and cancer type MLPs.
        desc_dim: dimension of description embeddings; if None, assumed equal to d_gene.
        """
        super(PreprocessedTransformerSurvivalModel, self).__init__()
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
        # Linear projection for description embeddings.
        if desc_dim is None:
            desc_dim = d_gene
        self.description_linear = nn.Linear(desc_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dropout=dropout, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        # Final linear to predict risk.
        self.final_linear = nn.Linear(d_model, 1)
        
    def forward(self, emb, scores, cnas, cancer_type, description, src_key_padding_mask=None):
        # Process gene tokens.
        gene_proj = self.gene_linear(emb)  # (B, L, d_model)
        polyphen_proj = self.polyphen_mlp(scores.unsqueeze(-1))  # (B, L, d_model)
        cna_proj = self.cna_mlp(cnas.unsqueeze(-1))              # (B, L, d_model)
        cancer_type_proj = self.cancer_type_mlp(cancer_type).unsqueeze(1)  # (B, 1, d_model)
        token_emb = gene_proj + polyphen_proj + cna_proj + cancer_type_proj  # (B, L, d_model)
        
        # Process description token.
        desc_proj = self.description_linear(description)  # (B, d_model)
        desc_proj = desc_proj.unsqueeze(1)  # (B, 1, d_model)
        
        # Prepend description token to the gene tokens.
        token_emb = torch.cat([desc_proj, token_emb], dim=1)  # (B, L+1, d_model)
        
        # Adjust padding mask: description token is always valid.
        if src_key_padding_mask is not None:
            new_mask = torch.cat([torch.zeros(src_key_padding_mask.size(0), 1, device=src_key_padding_mask.device, dtype=src_key_padding_mask.dtype),
                                  src_key_padding_mask], dim=1)
        else:
            new_mask = None
        
        # Transformer expects input of shape (seq_len, batch, d_model)
        token_emb = token_emb.transpose(0, 1)  # (L+1, B, d_model)
        transformer_out = self.transformer_encoder(token_emb, src_key_padding_mask=new_mask)
        transformer_out = transformer_out.transpose(0, 1)  # (B, L+1, d_model)
        # Use the output corresponding to the description token as the pooled representation.
        pooled = transformer_out[:, 0, :]  # (B, d_model)
        risk = self.final_linear(pooled)     # (B, 1)
        return risk
##########################################
# Evaluation: Calculate c-index for each Cancer Type
##########################################
def evaluate_by_cancer_type(model, dataloader, device):
    model.eval()
    all_T = []
    all_E = []
    all_risk = []
    cancer_acronyms = []
    with torch.no_grad():
        for batch in dataloader:
            emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, T_batch, E_batch, mask, cancer_acronym_list = batch
            emb_batch = emb_batch.to(device)
            score_batch = score_batch.to(device)
            cna_batch = cna_batch.to(device)
            cancer_type_batch = cancer_type_batch.to(device)
            desc_batch = desc_batch.to(device)
            T_batch = T_batch.to(device)
            E_batch = E_batch.to(device)
            mask = mask.to(device)
            risk = model(emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, src_key_padding_mask=mask)
            all_T.append(T_batch.cpu().numpy())
            all_E.append(E_batch.cpu().numpy())
            all_risk.append(risk.cpu().numpy())
            cancer_acronyms.extend(cancer_acronym_list)
    all_T = np.concatenate(all_T).squeeze()
    all_E = np.concatenate(all_E).squeeze()
    all_risk = np.concatenate(all_risk).squeeze()
    results = {}
    unique_types = np.unique(cancer_acronyms)
    for ct in unique_types:
        indices = np.array(cancer_acronyms) == ct
        sample_count = np.sum(indices)
        if sample_count < 2:
            results[ct] = np.nan
            continue
        T_ct = all_T[indices]
        E_ct = all_E[indices]
        risk_ct = all_risk[indices]
        event_count = np.sum(E_ct)
        if event_count == 0:
            results[ct] = np.nan
            continue
        try:
            c_index = concordance_index(T_ct, -risk_ct, E_ct)
            results[ct] = c_index
        except ZeroDivisionError:
            results[ct] = np.nan
    return results





def evaluate_all_folds(fold_dirs, model_dir, input_base_dir, cancer_type_mapping):
    all_T = []
    all_E = []
    all_risk = []
    all_cancer_types = []
    
    for fold in range(1, len(fold_dirs)+1):
        print(f"\nProcessing Fold {fold}")
        
        # Load fold-specific data
        test_path = os.path.join(input_base_dir, f"fold_{fold}", "test.parquet")
        model_path = os.path.join(model_dir, f"fold_{fold}", f"best_model_fold_{fold}.pth")
        
        # Load test data and model
        test_df = pd.read_parquet(test_path)
        test_dataset = PreprocessedSequenceDataset(test_df, ...)
        test_loader = DataLoader(...)
        
        model = load_model_for_fold(...)
        
        # Get predictions for this fold
        fold_T, fold_E, fold_risk, fold_ct = evaluate_single_fold(model, test_loader, device)
        
        # Aggregate across folds
        all_T.append(fold_T)
        all_E.append(fold_E)
        all_risk.append(fold_risk)
        all_cancer_types.extend(fold_ct)
    
    # Concatenate all data
    global_T = np.concatenate(all_T)
    global_E = np.concatenate(all_E)
    global_risk = np.concatenate(all_risk)
    
    # Compute global c-index
    global_cindex = concordance_index(global_T, -global_risk, global_E)
    
    # Compute per-cancer-type c-index using all data
    cancer_results = {}
    for ct in np.unique(all_cancer_types):
        mask = np.array(all_cancer_types) == ct
        if sum(mask) < 2: continue
        cancer_results[ct] = concordance_index(global_T[mask], -global_risk[mask], global_E[mask])
    
    return global_cindex, cancer_results


##########################################
# Main Function
##########################################
def main():
    # Determine number of folds from directory structure
    NUM_FOLDS = len([d for d in os.listdir(INPUT_DIR) if d.startswith("fold_")])
    print(f"Found {NUM_FOLDS} folds")

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Run cross-fold evaluation
    global_cindex, cancer_results = evaluate_all_folds(
        fold_dirs=[f"fold_{i}" for i in range(1, NUM_FOLDS+1)],
        model_dir=MODEL_DIR,
        input_base_dir=INPUT_DIR,
        cancer_type_mapping=cancer_type_mapping,
        device=device
    )

    # Print results
    print(f"\nGlobal C-index across all folds: {global_cindex:.4f}")
    print("\nC-index by Cancer Type (cross-fold):")
    for ct, cindex in cancer_results.items():
        print(f"  {ct}: {cindex:.4f}")

    # Save combined results
    results_df = pd.DataFrame({
        "Cancer_Type": list(cancer_results.keys()),
        "C_index": list(cancer_results.values())
    })
    output_csv = os.path.join(MODEL_DIR, "cross_fold_cindex_results.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved cross-fold results to {output_csv}")

##########################################
# Modified Evaluation Functions
##########################################
def evaluate_single_fold(model, dataloader, device):
    """Get predictions for a single fold without computing metrics"""
    model.eval()
    fold_T, fold_E, fold_risk, cancer_types = [], [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            emb, scores, cnas, cancer_type, desc, T, E, mask, ct_list = batch
            risk = model(
                emb.to(device),
                scores.to(device),
                cnas.to(device),
                cancer_type.to(device),
                desc.to(device),
                src_key_padding_mask=mask.to(device)
            )
            
            fold_T.append(T.cpu().numpy())
            fold_E.append(E.cpu().numpy())
            fold_risk.append(risk.cpu().numpy())
            cancer_types.extend(ct_list)
    
    return (
        np.concatenate(fold_T).squeeze(),
        np.concatenate(fold_E).squeeze(),
        np.concatenate(fold_risk).squeeze(),
        cancer_types
    )

def evaluate_all_folds(fold_dirs, model_dir, input_base_dir, cancer_type_mapping, device):
    all_T, all_E, all_risk, all_ct = [], [], [], []
    
    for fold_dir in fold_dirs:
        fold_num = fold_dir.split("_")[-1]
        print(f"\nProcessing {fold_dir}...")

        # Load data
        test_path = os.path.join(input_base_dir, fold_dir, "test.parquet")
        test_df = pd.read_parquet(test_path)
        test_dataset = PreprocessedSequenceDataset(
            test_df, 
            token_col="gene_embed_seq",
            cancer_type_mapping=cancer_type_mapping
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=32, 
            collate_fn=collate_fn_preprocessed
        )

        # Load model
        model_path = os.path.join(model_dir, fold_dir, f"best_model_fold_{fold_num}.pth")
        sample_batch = next(iter(test_loader))
        d_gene = sample_batch[0].shape[-1]
        
        model = PreprocessedTransformerSurvivalModel(
            d_gene=d_gene,
            d_model=256,
            polyphen_hidden_dim=128,
            nhead=4,
            dropout=0.1,
            desc_dim=sample_batch[4].shape[-1]
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Get predictions
        T, E, risk, ct = evaluate_single_fold(model, test_loader, device)
        all_T.append(T)
        all_E.append(E)
        all_risk.append(risk)
        all_ct.extend(ct)

    # Concatenate across folds
    global_T = np.concatenate(all_T)
    global_E = np.concatenate(all_E)
    global_risk = np.concatenate(all_risk)

    # Calculate metrics
    global_cindex = concordance_index(global_T, -global_risk, global_E)
    
    cancer_results = {}
    for ct in np.unique(all_ct):
        mask = np.array(all_ct) == ct
        if sum(mask) < 2: continue
        try:
            cancer_results[ct] = concordance_index(global_T[mask], -global_risk[mask], global_E[mask])
        except:
            cancer_results[ct] = np.nan
    
    return global_cindex, cancer_results



if __name__ == "__main__":
    main()