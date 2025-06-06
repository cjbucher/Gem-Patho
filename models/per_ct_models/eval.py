#!/usr/bin/env python
"""
Script to compute a global c-index across all per–cancer–type models.
For each fold in the k–fold test splits and for each cancer type (folder)
under /home/chb3333/yulab/chb3333/gem-patho/models/per_ct_models (excluding "slurm"),
this script:
  - Loads the test set for that fold.
  - Filters it by cancer type.
  - Loads the corresponding best model.
  - Computes risk scores on the test split.
Then the predictions are concatenated and a global c-index is computed.
"""

import os
import glob
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
from tqdm import tqdm

# Set seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

##########################################
# Constants & Paths
##########################################
# Input k–fold folder (assumes parquet files for each fold)
INPUT_DIR = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold_20val"
# Output directory for per–cancer–type models.
BASE_OUTPUT_DIR = "/home/chb3333/yulab/chb3333/gem-patho/models/per_ct_models"
# Number of folds to process.
NUM_FOLDS = 10

# Cancer type mapping CSV – maps study abbreviations (e.g., "LAML") to study names.
CANCER_TYPE_MAPPING_CSV = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/data_extraction/cancertype_location_description/tcga_study_abbreviations.csv"

def int_to_binary_vector(x, width=6):
    """Convert integer to a fixed-width binary vector (list of ints)."""
    return [int(b) for b in format(x, f"0{width}b")]

# Load cancer type mapping.
df_ct = pd.read_csv(CANCER_TYPE_MAPPING_CSV)
unique_types = sorted(df_ct["Study Abbreviation"].unique())
# This mapping is used to encode the "type" field.
cancer_type_mapping = {ct: int_to_binary_vector(i, 6) for i, ct in enumerate(unique_types)}
print("Cancer type mapping:", cancer_type_mapping)

# Build a mapping from cancer type abbreviation to a unique integer index.
type_to_index = {ct: i for i, ct in enumerate(unique_types)}
print("Cancer type to index mapping:", type_to_index)

##########################################
# Collate Function for Preprocessed Data
##########################################
def collate_fn_preprocessed(batch):
    """
    Collate function for per–cancer–type models.
    Each sample returns:
      (embeddings, scores, cnas, cancer_type_tensor, description, OS.time, OS, case_id)
    """
    emb_list, score_list, cna_list, cancer_type_list, desc_list, times, events, case_ids = zip(*batch)
    padded_emb = torch.nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in emb_list],
                                                  batch_first=True, padding_value=0.0)
    padded_scores = torch.nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in score_list],
                                                     batch_first=True, padding_value=0.0)
    padded_cnas = torch.nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in cna_list],
                                                  batch_first=True, padding_value=0.0)
    cancer_types = torch.stack(cancer_type_list)
    descriptions = torch.stack(desc_list)
    times = torch.stack(times)
    events = torch.stack(events)
    return padded_emb, padded_scores, padded_cnas, cancer_types, descriptions, times, events, case_ids

##########################################
# Dataset for Preprocessed Sequences (Per–CT Version)
##########################################
class PreprocessedSequenceDataset(Dataset):
    def __init__(self, df, token_col="gene_embed_seq", cancer_type_mapping=None):
        """
        Expects a DataFrame with columns:
          - "Case ID": patient identifier.
          - token_col: a list/array of tokens (each token is a dict with keys: "gene", "embedding", "score", "cna").
          - "OS.time": survival time.
          - "OS": event indicator.
          - "type": cancer type abbreviation.
          Optionally, if present, "description_embeddings" will be used.
        """
        self.df = df.reset_index(drop=True)
        self.token_col = token_col
        self.cancer_type_mapping = cancer_type_mapping if cancer_type_mapping is not None else {}
        self.has_description = "description_embeddings" in self.df.columns

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
        tokens = row[self.token_col]
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        if not tokens or (hasattr(tokens, '__len__') and len(tokens) == 0):
            tokens = [{"gene": "", "embedding": [0.0]*self.genename_dim, "score": 0.0, "cna": 0.0}]
        embeddings = [torch.tensor(token["embedding"], dtype=torch.float) for token in tokens]
        scores = [torch.tensor(token["score"], dtype=torch.float) for token in tokens]
        cnas = [torch.tensor(token.get("cna", 0.0), dtype=torch.float) for token in tokens]
        
        cancer_type_acronym = row.get("type", None)
        if cancer_type_acronym is None or cancer_type_acronym not in self.cancer_type_mapping:
            ct_vector = [0] * 6
        else:
            ct_vector = self.cancer_type_mapping[cancer_type_acronym]
        cancer_type_tensor = torch.tensor(ct_vector, dtype=torch.float)
        
        if self.has_description:
            description = torch.tensor(row["description_embeddings"], dtype=torch.float)
        else:
            print("No description embedding found for sample, using zero vector.")
            description = torch.zeros(self.genename_dim, dtype=torch.float)
        
        time = torch.tensor(row["OS.time"], dtype=torch.float)
        event = torch.tensor(row["OS"], dtype=torch.float)
        case_id = row.get("Case ID", "Unknown")
        return embeddings, scores, cnas, cancer_type_tensor, description, time, event, case_id

##########################################
# Unified Loss Function
##########################################
def compute_loss(risk, times, events, cancer_type_indices, model, lambda_reg=1e-4):
    """
    Computes the negative log-likelihood loss with sample weights normalized by cancer type frequency.
    """
    risk = torch.clamp(risk, min=-50, max=50)
    diff = times.unsqueeze(0) - times.unsqueeze(1)
    mat_A = (diff > 0).float()
    mat_B = (diff == 0).float().triu(diagonal=1)
    exp_risk = torch.exp(risk)
    R = torch.sum((mat_A + mat_B) * exp_risk.T, dim=1) + 1e-6
    unique_labels, inv, counts = torch.unique(cancer_type_indices, return_inverse=True, return_counts=True)
    scale = 1.0 / counts[inv].float()
    loss = -torch.mean(scale * events * (risk.squeeze() - torch.log(R)))
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2)
    loss += lambda_reg * reg_loss
    return loss

##########################################
# Transformer Survival Model (Per–CT Version)
##########################################
class PreprocessedTransformerSurvivalModel(nn.Module):
    def __init__(self, d_gene, d_model=256, polyphen_hidden_dim=128, nhead=4, dropout=0.1, desc_dim=None):
        """
        A survival transformer model for per–cancer–type training.
        d_gene: dimension of gene embeddings.
        d_model: token embedding dimension.
        desc_dim: dimension of description embeddings (if None, assumed equal to d_gene).
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
        if desc_dim is None:
            desc_dim = d_gene
        self.description_linear = nn.Linear(desc_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dropout=dropout, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.final_linear = nn.Linear(d_model, 1)
        
    def forward(self, emb, scores, cnas, cancer_type, description):
        gene_proj = self.gene_linear(emb)
        polyphen_proj = self.polyphen_mlp(scores.unsqueeze(-1))
        cna_proj = self.cna_mlp(cnas.unsqueeze(-1))
        cancer_type_proj = self.cancer_type_mlp(cancer_type).unsqueeze(1)
        token_emb = gene_proj + polyphen_proj + cna_proj + cancer_type_proj

        desc_proj = self.description_linear(description).unsqueeze(1)
        token_emb = torch.cat([desc_proj, token_emb], dim=1)
        # Transformer expects input shape (sequence length, batch, embedding dim)
        token_emb = token_emb.transpose(0, 1)
        transformer_out = self.transformer_encoder(token_emb)
        transformer_out = transformer_out.transpose(0, 1)
        pooled = transformer_out[:, 0, :]
        risk = self.final_linear(pooled)
        return risk

##########################################
# Evaluation Function
##########################################
def evaluate_model(model, dataloader, device):
    """
    Evaluate the model on a test split.
    Returns:
      avg_loss, global c-index, c-index by cancer type, normalized average c-index,
      arrays for times (all_T), events (all_E), risks (all_risk), cancer type indices (all_ct),
      list of case IDs, and list of missing cancer types (if any)
    """
    model.eval()
    all_T, all_E, all_risk, all_ct, all_case_ids = [], [], [], [], []
    losses = []
    missing_cancer_types = []
    with torch.no_grad():
        for emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, T_batch, E_batch, case_ids in dataloader:
            emb_batch = emb_batch.to(device)
            score_batch = score_batch.to(device)
            cna_batch = cna_batch.to(device)
            cancer_type_batch = cancer_type_batch.to(device)
            desc_batch = desc_batch.to(device)
            T_batch = T_batch.to(device)
            E_batch = E_batch.to(device)
            
            risk = model(emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch)
            
            # Convert binary cancer type vector back to an integer label.
            ct_idx_list = []
            for ct_tensor in cancer_type_batch:
                ct_list = ct_tensor.cpu().tolist()
                found = False
                for key, binary_vec in cancer_type_mapping.items():
                    if binary_vec == [int(x) for x in ct_list]:
                        ct_idx_list.append(type_to_index[key])
                        found = True
                        break
                if not found:
                    ct_idx_list.append(-1)
            ct_idx = torch.tensor(ct_idx_list, device=device)
            
            loss = compute_loss(risk, T_batch, E_batch, ct_idx, model)
            losses.append(loss.item())
            all_T.append(T_batch.cpu().numpy())
            all_E.append(E_batch.cpu().numpy())
            all_risk.append(risk.cpu().numpy())
            all_ct.append(ct_idx.cpu().numpy())
            all_case_ids.extend(case_ids)
    avg_loss = np.mean(losses)
    all_T = np.concatenate(all_T).squeeze()
    all_E = np.concatenate(all_E).squeeze()
    all_risk = np.concatenate(all_risk).squeeze()
    all_ct = np.concatenate(all_ct).squeeze()
    global_cindex = concordance_index(all_T, -all_risk, all_E)
    
    # Compute per-cancer–type c–indices.
    cindex_by_type = {}
    normalized_values = []
    unique_cts = np.unique(all_ct)
    for ct in unique_cts:
        mask_ct = (all_ct == ct)
        n = np.sum(mask_ct)
        if n < 2:
            c_idx = np.nan
            missing_cancer_types.append(int(ct))
        else:
            # Count admissible pairs.
            admissible_pairs = 0
            T_subset = all_T[mask_ct]
            E_subset = all_E[mask_ct]
            for i in range(n):
                for j in range(i+1, n):
                    if T_subset[i] != T_subset[j] and (E_subset[i] == 1 or E_subset[j] == 1):
                        admissible_pairs += 1
            if admissible_pairs == 0:
                c_idx = np.nan
                missing_cancer_types.append(int(ct))
            else:
                try:
                    c_idx = concordance_index(all_T[mask_ct], -all_risk[mask_ct], all_E[mask_ct])
                except ZeroDivisionError:
                    c_idx = np.nan
                    missing_cancer_types.append(int(ct))
        cindex_by_type[int(ct)] = c_idx
        if not np.isnan(c_idx):
            normalized_values.append(c_idx)
    normalized_avg_cindex = np.mean(normalized_values) if normalized_values else np.nan
    
    return avg_loss, global_cindex, cindex_by_type, normalized_avg_cindex, all_T, all_E, all_risk, all_ct, all_case_ids, missing_cancer_types

##########################################
# Main Function: Global Evaluation Over Per–CT Models
##########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # These lists will store results across all cancer types and folds.
    global_T = []
    global_E = []
    global_risk = []
    global_ct = []
    global_case_ids = []
    
    # Loop over each fold.
    for fold in range(1, NUM_FOLDS+1):
        print(f"\nProcessing Fold {fold} ...")
        # Path to the global test parquet file for the current fold.
        test_path = os.path.join(INPUT_DIR, f"fold_{fold}", "test.parquet")
        if not os.path.exists(test_path):
            print(f"Test file not found: {test_path}. Skipping fold {fold}.")
            continue
        
        # Load the test data.
        cols = ["Case ID", "gene_embed_seq", "OS.time", "OS", "type", "description_embeddings"]
        try:
            test_df = pd.read_parquet(test_path, engine="pyarrow")[cols]
        except Exception as e:
            print(f"Error loading test file {test_path}: {e}. Skipping fold {fold}.")
            continue
        
        # List cancer type folders (skip if folder is "slurm" or non–directory).
        cancer_type_folders = [ct for ct in os.listdir(BASE_OUTPUT_DIR)
                               if os.path.isdir(os.path.join(BASE_OUTPUT_DIR, ct)) and ct.lower() != "slurm"]
        if not cancer_type_folders:
            print("No cancer type folders found under", BASE_OUTPUT_DIR)
            return
        
        # Loop over each cancer type.
        for ct in sorted(cancer_type_folders):
            # Filter the test DataFrame to the current cancer type.
            test_df_ct = test_df[test_df["type"] == ct].copy()
            if test_df_ct.empty:
                print(f"No test samples for cancer type {ct} in fold {fold}. Skipping.")
                continue
            
            # Expect the best model to be saved in:
            # BASE_OUTPUT_DIR/{ct}/fold_{fold}/best_model_fold_{fold}_ct_{ct}.pth
            model_folder = os.path.join(BASE_OUTPUT_DIR, ct, f"fold_{fold}")
            if not os.path.exists(model_folder):
                print(f"Model folder not found for cancer type {ct} in fold {fold}: {model_folder}. Skipping.")
                continue
            model_file = os.path.join(model_folder, f"best_model_fold_{fold}_ct_{ct}.pth")
            if not os.path.exists(model_file):
                print(f"Best model file not found for cancer type {ct}, fold {fold}: {model_file}. Skipping.")
                continue
            
            print(f"Evaluating cancer type {ct}, fold {fold} ...")
            # Create dataset and dataloader.
            test_dataset = PreprocessedSequenceDataset(test_df_ct, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)
            batch_size = 64
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed)
            
            # Determine embedding dimensions from a sample batch.
            try:
                sample_batch = next(iter(test_loader))
            except StopIteration:
                print(f"No samples found in DataLoader for cancer type {ct}, fold {fold}. Skipping.")
                continue
            
            sample_emb = sample_batch[0]
            sample_desc = sample_batch[4]
            d_gene = sample_emb.shape[-1]
            
            # Instantiate the per–CT model.
            model = PreprocessedTransformerSurvivalModel(d_gene=d_gene, d_model=256,
                                                           polyphen_hidden_dim=128, nhead=4, dropout=0.1,
                                                           desc_dim=sample_desc.shape[-1])
            model.to(device)
            try:
                model.load_state_dict(torch.load(model_file, map_location=device))
            except Exception as e:
                print(f"Error loading model state from {model_file}: {e}. Skipping cancer type {ct}, fold {fold}.")
                continue
            model.eval()
            
            try:
                # Evaluate the model on the filtered test set.
                avg_loss, cindex, cindex_by_type, norm_avg_cindex, all_T, all_E, all_risk, all_ct, case_ids, missing = evaluate_model(model, test_loader, device)
            except Exception as e:
                print(f"Error during evaluation for cancer type {ct}, fold {fold}: {e}. Skipping.")
                continue
            
            print(f"Cancer type {ct}, Fold {fold}: Test loss = {avg_loss:.4f}, c-index = {cindex:.4f}")
            
            # Append results to the global lists.
            global_T.append(all_T)
            global_E.append(all_E)
            global_risk.append(all_risk)
            global_ct.append(all_ct)
            global_case_ids.extend(case_ids)
    
    # If no predictions were collected, stop.
    if not global_T:
        print("No test predictions processed across folds and cancer types.")
        return
    
    # Concatenate results across folds and cancer types.
    global_T = np.concatenate(global_T).squeeze()
    global_E = np.concatenate(global_E).squeeze()
    global_risk = np.concatenate(global_risk).squeeze()
    global_ct = np.concatenate(global_ct).squeeze()
    
    # Compute global c-index.
    global_cindex = concordance_index(global_T, -global_risk, global_E)
    print(f"\nGlobal test c-index (all folds, all cancer types): {global_cindex:.4f}")
    
    # Compute per–cancer–type c–indices.
    ct_cindices = {}
    for ct in np.unique(global_ct):
        subset_mask = (global_ct == ct)
        subset = {
            "OS.time": global_T[subset_mask],
            "OS": global_E[subset_mask],
            "risk": global_risk[subset_mask]
        }
        if len(subset["OS.time"]) < 2:
            ct_cindices[ct] = np.nan
            print(f"Cancer type {ct} has too few samples for c-index calculation.")
        else:
            try:
                ct_cindex = concordance_index(subset["OS.time"], -subset["risk"], subset["OS"])
            except ZeroDivisionError:
                ct_cindex = np.nan
            ct_cindices[ct] = ct_cindex
            print(f"Cancer type {ct}: c-index = {ct_cindex:.4f}")
    
    # Save global test risk scores and c-index summary.
    global_df = pd.DataFrame({
        "Case ID": global_case_ids,
        "OS.time": global_T,
        "OS": global_E,
        "risk": global_risk,
        "cancer_type": global_ct,
    })
    global_save_path = os.path.join(BASE_OUTPUT_DIR, "global_test_risk_scores.csv")
    global_df.to_csv(global_save_path, index=False)
    print(f"Saved global test risk scores to {global_save_path}")
    
    # Save global test c-index.
    global_cindex_file = os.path.join(BASE_OUTPUT_DIR, "global_test_cindex.txt")
    with open(global_cindex_file, "w") as f:
        f.write(str(global_cindex))
    print(f"Saved global test c-index to {global_cindex_file}")
    
    # Create an inverse mapping: integer index -> cancer type abbreviation.
    index_to_type = {v: k for k, v in type_to_index.items()}
    
    # Convert per-cancer-type c-indices using abbreviations.
    ct_df = pd.DataFrame(
        [(index_to_type.get(key, key), value) for key, value in ct_cindices.items()],
        columns=["cancer_type", "c_index"]
    )
    ct_save_path = os.path.join(BASE_OUTPUT_DIR, "global_test_cindices_by_cancer.csv")
    ct_df.to_csv(ct_save_path, index=False)
    print(f"Saved per-cancer-type test c-indices to {ct_save_path}")

if __name__ == "__main__":
    main()
