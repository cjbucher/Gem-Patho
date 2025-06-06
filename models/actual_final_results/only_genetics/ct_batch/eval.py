#!/usr/bin/env python
"""
Script to compute test risk scores from scratch using the best genetic-only model
for each fold, normalize the risk scores within each fold, and then compute global 
and per–cancer-type c-indices.

If a fold’s test risk file already exists, the risk scores are loaded (and the best 
model for that fold is also loaded if available) and used for global aggregation.
Otherwise, risk scores are computed from scratch.
 
Input directory: /n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold_20val
Base output directory: /home/chb3333/yulab/chb3333/gem-patho/models/actual_final_results/only_genetics/ct_batch
There are folders fold_1, fold_2, …, fold_10.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import argparse
import json
from lifelines.utils import concordance_index
from torch.utils.data import Dataset, DataLoader

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

##########################################
# Constants & Paths
##########################################
INPUT_DIR = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold_20val"
BASE_OUTPUT_DIR = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/models/actual_final_results/only_genetics/ct_batch"
NUM_FOLDS = 10
print("Base output directory:", BASE_OUTPUT_DIR)

# Cancer type mapping CSV – maps study abbreviations to study names.
CANCER_TYPE_MAPPING_CSV = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/data_extraction/cancertype_location_description/tcga_study_abbreviations.csv"

def int_to_binary_vector(x, width=6):
    """Convert integer to a fixed-width binary vector (list of ints)."""
    return [int(b) for b in format(x, f"0{width}b")]

df_ct = pd.read_csv(CANCER_TYPE_MAPPING_CSV)
unique_types = sorted(df_ct["Study Abbreviation"].unique())
cancer_type_mapping = {ct: int_to_binary_vector(i, 6) for i, ct in enumerate(unique_types)}
print("Cancer type mapping:", cancer_type_mapping)

# Mapping from cancer type abbreviation to a unique integer index.
type_to_index = {ct: i for i, ct in enumerate(unique_types)}
print("Cancer type to index mapping:", type_to_index)

##########################################
# Dataset and Collate for Simplified Genetic Data
##########################################
class SimpleSurvivalDataset(Dataset):
    def __init__(self, df):
        """
        Expects a DataFrame that includes:
          - "polyphen_score": mutation vector (nonzero entries become 1)
          - "gene_embed_seq": list of tokens (used for CNA values)
          - "OS.time": survival time
          - "OS": event indicator
          - "Case ID": patient identifier
          - "type": cancer type (e.g. "LAML")
        """
        self.df = df.reset_index(drop=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Process polyphen vector.
        polyphen = row["polyphen_score"]
        if isinstance(polyphen, list):
            polyphen = np.array(polyphen)
        elif not isinstance(polyphen, np.ndarray):
            polyphen = np.array([polyphen])
        polyphen = torch.tensor(polyphen, dtype=torch.float)
        polyphen = (polyphen != 0).float()  # one-hot: nonzero -> 1
        
        # Construct CNA vector.
        global gene_list
        cna_vector = [0.0] * len(gene_list)
        tokens = row["gene_embed_seq"]
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        if tokens is None or len(tokens) == 0:
            tokens = []
        for token in tokens:
            gene = token.get("gene", "")
            cna_value = token.get("cna", 0.0)
            if gene in gene_list:
                idx_gene = gene_list.index(gene)
                cna_vector[idx_gene] = cna_value
        cna_vector = torch.tensor(cna_vector, dtype=torch.float)
        
        time = torch.tensor(row["OS.time"], dtype=torch.float)
        event = torch.tensor(row["OS"], dtype=torch.float)
        case_id = row.get("Case ID", "Unknown")
        cancer_type = row.get("type", "Unknown")
        
        return polyphen, cna_vector, time, event, case_id, cancer_type

def collate_fn_simple(batch):
    polyphen_batch, cna_batch, times, events, case_ids, cancer_types = zip(*batch)
    polyphen_batch = torch.stack(polyphen_batch)
    cna_batch = torch.stack(cna_batch)
    times = torch.stack(times)
    events = torch.stack(events)
    return polyphen_batch, cna_batch, times, events, list(case_ids), list(cancer_types)

##########################################
# Simplified Survival Model (as used during training)
##########################################
class SimpleSurvivalModel(nn.Module):
    def __init__(self, polyphen_dim, cna_dim, hidden_dim=256):
        """
        Args:
            polyphen_dim (int): Dimension of the polyphen vector.
            cna_dim (int): Dimension of the CNA vector.
            hidden_dim (int): Hidden layer dimension.
        """
        super(SimpleSurvivalModel, self).__init__()
        input_dim = polyphen_dim + cna_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.final_linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, polyphen, cna):
        combined = torch.cat([polyphen, cna], dim=1)
        features = self.mlp(combined)
        risk = self.final_linear(features)
        return risk

##########################################
# Evaluation Function
##########################################
def evaluate_model(model, dataloader, device):
    model.eval()
    all_T, all_E, all_risk = [], [], []
    all_case_ids, all_types = [], []
    with torch.no_grad():
        for polyphen_batch, cna_batch, times, events, case_ids, cancer_types in dataloader:
            polyphen_batch = polyphen_batch.to(device)
            cna_batch = cna_batch.to(device)
            times = times.to(device)
            events = events.to(device)
            risk = model(polyphen_batch, cna_batch)
            all_T.extend(times.cpu().numpy())
            all_E.extend(events.cpu().numpy())
            all_risk.extend(risk.cpu().numpy())
            all_case_ids.extend(case_ids)
            all_types.extend(cancer_types)
    global_cindex = concordance_index(np.array(all_T), -np.array(all_risk).squeeze(), np.array(all_E))
    return global_cindex, np.array(all_T), np.array(all_risk).squeeze(), np.array(all_E), all_case_ids, all_types

##########################################
# Main Function: Compute Test Risk Scores and c-indices
##########################################
def main():
    parser = argparse.ArgumentParser(description="Compute test risk scores from the best genetic-only model and calculate c-indices.")
    parser.add_argument("--exp_folder", type=str, default="cancertypebatching_exp_0_warmup_0",
                        help="Experiment folder name (e.g., cancertypebatching_exp_0_warmup_0)")
    args = parser.parse_args()
    EXPERIMENT = args.exp_folder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the fixed gene list (used during training)
    gene_list_path = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/data_extraction/cancer_gene_list_selection/combined_genelist.csv"
    gene_df = pd.read_csv(gene_list_path)
    global gene_list
    if "gene" in gene_df.columns:
        gene_list = gene_df["gene"].tolist()
    else:
        gene_list = gene_df.iloc[:, 0].tolist()
    print("Loaded gene list of length:", len(gene_list))
    
    all_test_dfs = []
    batch_size = 64

    # Process each fold: if a test risk file exists, load it and also load the best model.
    for fold in range(1, NUM_FOLDS+1):
        print(f"\nProcessing Fold {fold} ...")
        fold_save_dir = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT, f"fold_{fold}")
        test_risk_file = os.path.join(fold_save_dir, "test_risk_scores.csv")
        if os.path.exists(test_risk_file):
            print(f"Test risk scores already computed for fold {fold} at {test_risk_file}. Loading risk file.")
            df_test_risk = pd.read_csv(test_risk_file)
            # Also load the best model for completeness if needed.
            best_model_path = os.path.join(fold_save_dir, "best_model.pth")
            if os.path.exists(best_model_path):
                print(f"Best model found for fold {fold} at {best_model_path}.")
            else:
                print(f"Best model not found for fold {fold}.")
            all_test_dfs.append(df_test_risk)
            continue
        
        fold_folder = os.path.join(INPUT_DIR, f"fold_{fold}")
        test_path = os.path.join(fold_folder, "test.parquet")
        if not os.path.exists(test_path):
            print(f"Test file not found: {test_path}. Skipping fold {fold}.")
            continue

        # Assume the test file contains at least: Case ID, polyphen_score, gene_embed_seq, OS.time, OS, type.
        cols = ["Case ID", "polyphen_score", "gene_embed_seq", "OS.time", "OS", "type"]
        test_df = pd.read_parquet(test_path, engine="pyarrow")[cols]
        
        # Create test dataset and dataloader.
        test_dataset = SimpleSurvivalDataset(test_df)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_simple)
        
        # Load the best model for this fold.
        model_path = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT, f"fold_{fold}", "best_model.pth")
        if not os.path.exists(model_path):
            print(f"Best model not found for fold {fold} at {model_path}. Skipping.")
            continue
        
        # Determine input dimensions from a sample.
        sample_polyphen, sample_cna, _, _, _, _ = test_dataset[0]
        polyphen_dim = sample_polyphen.shape[0]
        cna_dim = sample_cna.shape[0]
        
        model = SimpleSurvivalModel(polyphen_dim, cna_dim)
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        try:
            global_cindex, all_T, all_risk, all_E, all_case_ids, all_types = evaluate_model(model, test_loader, device)
        except Exception as e:
            print(f"Error during evaluation for fold {fold}: {e}. Skipping fold {fold}.")
            continue
        
        # Normalize risk scores (z-score normalization within the fold)
        all_risk = np.array(all_risk)
        mean_risk = np.mean(all_risk)
        std_risk = np.std(all_risk)
        if std_risk > 0:
            normalized_risk = (all_risk - mean_risk) / std_risk
        else:
            normalized_risk = all_risk
        
        df_test_risk = pd.DataFrame({
            "Case ID": all_case_ids,
            "OS.time": all_T,
            "OS": all_E,
            "risk": all_risk,
            "normalized_risk": normalized_risk,
            "cancer_type": all_types,
            "fold": fold
        })
        os.makedirs(fold_save_dir, exist_ok=True)
        save_path = os.path.join(fold_save_dir, "test_risk_scores.csv")
        df_test_risk.to_csv(save_path, index=False)
        print(f"Saved test risk scores for fold {fold} to {save_path}")
        
        all_test_dfs.append(df_test_risk)
    
    # After processing all folds, concatenate results and compute global and per-cancer-type c-indices.
    if all_test_dfs:
        all_test_df = pd.concat(all_test_dfs, ignore_index=True)
        global_cindex = concordance_index(all_test_df["OS.time"], -all_test_df["normalized_risk"], all_test_df["OS"])
        print(f"\nGlobal test c-index (all folds): {global_cindex:.4f}") 
        # Save the global test c-index in an extra file.
        global_cindex_file = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT, "global_test_cindex.txt")
        with open(global_cindex_file, "w") as f:
            f.write(str(global_cindex))
        print(f"Saved global test c-index to {global_cindex_file}")
        
        # Compute per-cancer-type c-indices.
        ct_cindices = {}
        for ct in all_test_df["cancer_type"].unique():
            subset = all_test_df[all_test_df["cancer_type"] == ct]
            if subset.shape[0] < 2:
                ct_cindices[ct] = np.nan
                print(f"Cancer type {ct} has too few samples for c-index calculation.")
            else:
                try:
                    ct_cindex = concordance_index(subset["OS.time"], -subset["normalized_risk"], subset["OS"])
                except ZeroDivisionError:
                    ct_cindex = np.nan
                ct_cindices[ct] = ct_cindex
                print(f"Cancer type {ct}: c-index = {ct_cindex:.4f}")
        
        global_save_path = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT, "global_test_risk_scores.csv")
        all_test_df.to_csv(global_save_path, index=False)
        print(f"Saved global test risk scores to {global_save_path}")
        
        ct_summary = pd.DataFrame([{"cancer_type": ct, "c_index": c_idx} for ct, c_idx in ct_cindices.items()])
        ct_summary_path = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT, "global_test_cindices.csv")
        ct_summary.to_csv(ct_summary_path, index=False)
        print(f"Saved global test c-indices summary to {ct_summary_path}")
    else:
        print("No test data processed across folds.")

if __name__ == "__main__":
    main()
