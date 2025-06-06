#!/usr/bin/env python
"""
Transformer-based survival model using preprocessed Jina embeddings,
polyphen scores, CNA values, cancer type information, and description embeddings.

Data files are assumed to be in:
  Input k-folds:
    /n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold_20val/
Model outputs (per-cancer-type models) will be saved in:
  /n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/models/per_ct_models
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index  # for c-index
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import argparse

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
# New output directory for per-cancer-type models
BASE_OUTPUT_DIR = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho//models/per_ct_models"
NUM_FOLDS = 10
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
print("Base output directory:", BASE_OUTPUT_DIR)

# Cancer type mapping CSV – maps study abbreviations (e.g., "LAML") to study names.
CANCER_TYPE_MAPPING_CSV = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/data_extraction/cancertype_location_description/tcga_study_abbreviations.csv"

def int_to_binary_vector(x, width=6):
    """Convert integer to a fixed-width binary vector (list of ints)."""
    return [int(b) for b in format(x, f"0{width}b")]

# Load cancer type mapping.
df_ct = pd.read_csv(CANCER_TYPE_MAPPING_CSV)
unique_types = sorted(df_ct["Study Abbreviation"].unique())
cancer_type_mapping = {ct: int_to_binary_vector(i, 6) for i, ct in enumerate(unique_types)}
print("Cancer type mapping:", cancer_type_mapping)

# Build a mapping from cancer type abbreviation to a unique integer index.
type_to_index = {ct: i for i, ct in enumerate(unique_types)}
print("Cancer type to index mapping:", type_to_index)

##########################################
# Collate Function for Padding (mask removed)
##########################################
def collate_fn_preprocessed(batch):
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
# Unified Loss Function (normalized by cancer type)
##########################################
def compute_loss(risk, times, events, cancer_type_indices, model, lambda_reg=1e-4):
    """
    Computes the negative log-likelihood loss for the survival model.
    Each sample is weighted by 1/(# samples in its cancer type).
    No masking of pairs is applied.
    """
    risk = torch.clamp(risk, min=-50, max=50)
    
    # Compute pairwise differences.
    diff = times.unsqueeze(0) - times.unsqueeze(1)
    mat_A = (diff > 0).float()
    mat_B = (diff == 0).float().triu(diagonal=1)
    
    exp_risk = torch.exp(risk)
    R = torch.sum((mat_A + mat_B) * exp_risk.T, dim=1) + 1e-6
    
    # Normalize loss by cancer type frequency.
    unique_labels, inv, counts = torch.unique(cancer_type_indices, return_inverse=True, return_counts=True)
    scale = 1.0 / counts[inv].float()
    
    loss = -torch.mean(scale * events * (risk.squeeze() - torch.log(R)))
    
    # L2 regularization.
    reg_loss = 0.0
    for param in model.parameters():
        reg_loss += torch.sum(param ** 2)
    
    loss += lambda_reg * reg_loss
    return loss

##########################################
# Dataset for Preprocessed Sequences (Jina)
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
          Optionally, if present, "description_embeddings" will be used as a separate token.
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
            ct_vector = [0]*6
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
# Transformer Survival Model (with Description Token)
##########################################
class PreprocessedTransformerSurvivalModel(nn.Module):
    def __init__(self, d_gene, d_model=256, polyphen_hidden_dim=128, nhead=4, dropout=0.1, desc_dim=None):
        """
        d_gene: dimension of the gene embedding.
        d_model: token dimension.
        polyphen_hidden_dim: hidden dimension for polyphen, CNA, and cancer type MLPs.
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
    model.eval()
    all_T, all_E, all_risk, all_ct, all_case_ids = [], [], [], [], []
    losses = []
    missing_cancer_types = []  # To record cancer types with insufficient pairs for c-index computation
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
    # Compute global c-index with error handling.
    try:
        global_cindex = concordance_index(all_T, -all_risk, all_E)
    except ZeroDivisionError:
        print("No admissible pairs in global evaluation; setting global c-index to NaN.")
        global_cindex = np.nan
    
    # Compute per-cancer type c-index.
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
                c_idx = concordance_index(all_T[mask_ct], -all_risk[mask_ct], all_E[mask_ct])
        cindex_by_type[int(ct)] = c_idx
        if not np.isnan(c_idx):
            normalized_values.append(c_idx)
    normalized_avg_cindex = np.mean(normalized_values) if normalized_values else np.nan
    
    return avg_loss, global_cindex, cindex_by_type, normalized_avg_cindex, all_T, all_E, all_risk, all_ct, all_case_ids, missing_cancer_types


##########################################
# Training Function (Phase 0 only; phase2_epochs set to 0)
##########################################
def train_model_fn(train_loader, val_loader, model, device,
                   max_phase0_epochs=100, phase2_epochs=0, patience=20, batch_size=64):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    history = []
    best_val_loss = float('inf')
    best_cindex = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = copy.deepcopy(model.state_dict())
    phase0_epochs_run = 0

    print("Starting Phase 0 training (up to {} epochs) ...".format(max_phase0_epochs))
    for epoch in range(1, max_phase0_epochs+1):
        model.train()
        train_losses = []
        for emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, T_batch, E_batch, _ in train_loader:
            emb_batch = emb_batch.to(device)
            score_batch = score_batch.to(device)
            cna_batch = cna_batch.to(device)
            cancer_type_batch = cancer_type_batch.to(device)
            desc_batch = desc_batch.to(device)
            T_batch = T_batch.to(device)
            E_batch = E_batch.to(device)
            
            # Convert binary cancer type vector to integer indices.
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
            
            optimizer.zero_grad()
            risk = model(emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch)
            loss = compute_loss(risk, T_batch, E_batch, ct_idx, model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss_epoch = np.mean(train_losses)

        val_loss, val_cindex_global, val_cindex_by_type, normalized_avg_cindex, _, _, _, _, _, _ = evaluate_model(
            model, val_loader, device)
        history.append({
            "phase": 0,
            "epoch": epoch,
            "train_loss": train_loss_epoch,
            "val_loss": val_loss,
            "val_cindex_global": val_cindex_global,
            "normalized_avg_cindex": normalized_avg_cindex,
            "val_cindex_by_type": val_cindex_by_type
        })
        print(f"Phase 0 Epoch {epoch:02d}: Train Loss = {train_loss_epoch:.4f}, Val Loss = {val_loss:.4f}, Global Val C-index = {val_cindex_global:.4f}, Normalized Val C-index = {normalized_avg_cindex:.4f}")
        
        if val_loss < best_val_loss or val_cindex_global > best_cindex:
            best_val_loss = min(best_val_loss, val_loss)
            best_cindex = max(best_cindex, val_cindex_global)
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        phase0_epochs_run = epoch

        if epochs_no_improve >= patience:
            print("Early stopping in Phase 0 at epoch", epoch)
            break

    model.load_state_dict(best_model_state)
    return model, best_epoch, best_val_loss, history, phase0_epochs_run

##########################################
# Plotting Function
##########################################
def plot_history(history, save_dir, fold, phase0_end_epoch=None):
    h_df = pd.DataFrame(history)
    
    plt.figure(figsize=(10,5))
    plt.plot(h_df['epoch'], h_df['val_cindex_global'], label='Global Val C-index')
    plt.xlabel('Epoch')
    plt.ylabel('C-index')
    plt.title(f'Global Val C-index - Fold {fold}')
    if phase0_end_epoch is not None:
        plt.axvline(x=phase0_end_epoch, color='red', linestyle='--', label='Phase 0 End')
    plt.legend()
    global_cindex_path = os.path.join(save_dir, f'global_cindex_fold_{fold}.png')
    plt.savefig(global_cindex_path)
    plt.close()
    
    all_types = set()
    for d in h_df['val_cindex_by_type']:
        all_types.update(d.keys())
    all_types = sorted(list(all_types))
    
    plt.figure(figsize=(10,5))
    for ct in all_types:
        values = [d.get(ct, np.nan) for d in h_df['val_cindex_by_type']]
        plt.plot(h_df['epoch'], values, label=f'Cancer type {ct}')
    plt.plot(h_df['epoch'], h_df['normalized_avg_cindex'], label='Normalized Avg', linewidth=2, linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('C-index')
    plt.title(f'Per-Cancer-Type Val C-index - Fold {fold}')
    if phase0_end_epoch is not None:
        plt.axvline(x=phase0_end_epoch, color='red', linestyle='--', label='Phase 0 End')
    plt.legend()
    per_type_cindex_path = os.path.join(save_dir, f'per_type_cindex_fold_{fold}.png')
    plt.savefig(per_type_cindex_path)
    plt.close()
    
    history_save_path = os.path.join(save_dir, "history_data.csv")
    h_df.to_csv(history_save_path, index=False)
    print(f"Saved plots and history for fold {fold} in {save_dir}")

##########################################
# Main Training Loop Over Folds and Cancer Types
##########################################
all_experiment_results = []

def main():
    parser = argparse.ArgumentParser(description="Train separate models for each cancer type in one fold.")
    parser.add_argument("--job_idx", type=int, required=True,
                        help="Global job index (0-indexed) used to select the fold.")
    args = parser.parse_args()

    # Use job_idx to select the fold (we assume one experiment with phase2_epochs=0)
    fold = (args.job_idx % NUM_FOLDS) + 1

    print(f"Global job index: {args.job_idx}")
    print(f"Selected fold: {fold}")

    if not torch.cuda.is_available():
        print("No GPU available.")

    cols = ["Case ID", "gene_embed_seq", "OS.time", "OS", "type", "description_embeddings"]

    fold_folder = os.path.join(INPUT_DIR, f"fold_{fold}")
    train_path = os.path.join(fold_folder, "train.parquet")
    val_path = os.path.join(fold_folder, "val.parquet")
    test_path = os.path.join(fold_folder, "test.parquet")
    train_df = pd.read_parquet(train_path, engine="pyarrow")[cols]
    val_df = pd.read_parquet(val_path, engine="pyarrow")[cols]
    test_df = pd.read_parquet(test_path, engine="pyarrow")[cols]

    # Get unique cancer types from the training data.
    unique_cancer_types = train_df['type'].unique()
    print("Unique cancer types found:", unique_cancer_types)

    batch_size = 64

    # Loop over each cancer type and train a separate model.
    for ct in unique_cancer_types:
        print(f"\n====================\nTraining model for cancer type: {ct}\n====================")
        # Filter the DataFrames for the current cancer type.
        train_df_ct = train_df[train_df['type'] == ct].copy()
        val_df_ct = val_df[val_df['type'] == ct].copy()
        test_df_ct = test_df[test_df['type'] == ct].copy()

        # Skip if not enough training samples.
        if len(train_df_ct) < 10:
            print(f"Skipping cancer type {ct} due to insufficient training samples ({len(train_df_ct)} samples).")
            continue

        # Create datasets for this cancer type.
        train_dataset = PreprocessedSequenceDataset(train_df_ct, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)
        val_dataset = PreprocessedSequenceDataset(val_df_ct, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)
        test_dataset = PreprocessedSequenceDataset(test_df_ct, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)

        # Create DataLoaders with normal batching.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_preprocessed)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed)

        # Get sample embedding dimension.
        sample_emb, sample_scores, sample_cnas, sample_cancer_type, sample_desc, sample_time, sample_event, _ = next(iter(train_loader))
        d_gene = sample_emb.shape[-1]

        # Instantiate the model.
        model = PreprocessedTransformerSurvivalModel(d_gene=d_gene, d_model=256,
                                                       polyphen_hidden_dim=128, nhead=4, dropout=0.1,
                                                       desc_dim=sample_desc.shape[-1])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        model.to(device)
        print(f"Training model for cancer type {ct}, Fold {fold}")
        # Train the model (phase2_epochs is 0).
        model, best_epoch, best_val_loss, history, phase0_epochs_run = train_model_fn(
            train_loader, val_loader, model, device, max_phase0_epochs=100,
            phase2_epochs=0, patience=20, batch_size=batch_size
        )

        # Create output subfolder for this cancer type and fold.
        ct_folder = os.path.join(BASE_OUTPUT_DIR, ct, f"fold_{fold}")
        os.makedirs(ct_folder, exist_ok=True)

        model_save_path = os.path.join(ct_folder, f"best_model_fold_{fold}_ct_{ct}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved best model for cancer type {ct}, Fold {fold} to {model_save_path}")

        # Save predictions for train and val splits.
        for split, loader, prefix in zip(["train", "val"], [train_loader, val_loader], ["train", "val"]):
            avg_loss, global_cindex, cindex_by_type, normalized_avg_cindex, all_T, all_E, all_risk, all_ct, all_case_ids, missing_ct = evaluate_model(model, loader, device)

            df_out = pd.DataFrame({
                "Case ID": all_case_ids,
                "OS.time": all_T,
                "OS": all_E,
                "risk": all_risk,
                "cancer_type": all_ct,
                "split": prefix
            })
            out_path = os.path.join(ct_folder, f"{prefix}_risk_scores_ct_{ct}.csv")
            df_out.to_csv(out_path, index=False)
            print(f"Saved {prefix} risk scores for cancer type {ct} to {out_path}")
            missing_df = pd.DataFrame({"Missing Cancer Types": missing_ct})
            missing_out_path = os.path.join(ct_folder, f"{prefix}_missing_cancer_types_ct_{ct}.csv")
            missing_df.to_csv(missing_out_path, index=False)
            print(f"Saved missing cancer types for {prefix} split to {missing_out_path}")

        # Evaluate on the test split.
        test_loss, test_cindex, _, _, test_T, test_E, test_risk, test_ct, test_case_ids, _ = evaluate_model(model, test_loader, device)

        risk_mean = np.mean(test_risk)
        risk_std = np.std(test_risk)
        try:
            corr, _ = pearsonr(test_T, test_risk)
        except Exception as e:
            print("Pearson correlation error:", e)
            corr = np.nan

        fold_metrics = {
            "fold": fold,
            "cancer_type": ct,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_loss": evaluate_model(model, train_loader, device)[0],
            "train_cindex": evaluate_model(model, train_loader, device)[1],
            "val_loss": evaluate_model(model, val_loader, device)[0],
            "val_cindex_global": evaluate_model(model, val_loader, device)[1],
            "normalized_val_cindex": evaluate_model(model, val_loader, device)[3],
            "test_loss": test_loss,
            "test_cindex": test_cindex,
            "test_risk_mean": risk_mean,
            "test_risk_std": risk_std,
            "test_risk_OS_corr": corr
        }
        metrics_save_path = os.path.join(ct_folder, "fold_metrics.csv")
        pd.DataFrame([fold_metrics]).to_csv(metrics_save_path, index=False)
        print(f"Saved fold metrics to {metrics_save_path}")

        # Save history and plots.
        history_df = pd.DataFrame(history)
        history_save_path = os.path.join(ct_folder, "history_fold.csv")
        history_df.to_csv(history_save_path, index=False)
        plot_history(history, ct_folder, fold, phase0_end_epoch=phase0_epochs_run)

        all_experiment_results.append(fold_metrics)

        del model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
        torch.cuda.empty_cache()

    # Save overall experiment summary.
    summary_df = pd.DataFrame(all_experiment_results)
    summary_csv_path = os.path.join(BASE_OUTPUT_DIR, "all_experiments_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved experiment summary to {summary_csv_path}")



if __name__ == "__main__":
    main()
