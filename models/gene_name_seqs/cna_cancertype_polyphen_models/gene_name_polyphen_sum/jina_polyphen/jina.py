#!/usr/bin/env python
"""
Transformer-based survival model using preprocessed Jina embeddings,
polyphen scores, CNA values, and cancer type information.

Data files are now assumed to be in:
  Input k-folds:
    /home/chb3333/yulab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold/fold_1
  Model outputs will be saved in:
    /home/chb3333/yulab/chb3333/gem-patho/models/gene_name_seqs/cna_cancertype_polyphen_models/gene_name_polyphen_sum/jina_polyphen

Cancer type information is provided via the "CANCER_TYPE_ACRONYM" column.
It is converted to a 6-dimensional binary vector (based on an index mapping from the file:
  /home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/tcga_study_abbreviations.csv)
and fed through an MLP. Its output is added (broadcast over the token dimension) along with the other projections.
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

##########################################
# Constants & Paths
##########################################
INPUT_DIR = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold"
OUTPUT_DIR = "/home/chb3333/yulab/chb3333/gem-patho/models/gene_name_seqs/cna_cancertype_polyphen_models/gene_name_polyphen_sum/jina_polyphen"
NUM_FOLDS = 10

# Cancer type mapping CSV – this file maps study abbreviations (e.g., "LAML") to study names.
CANCER_TYPE_MAPPING_CSV = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/tcga_study_abbreviations.csv"

def int_to_binary_vector(x, width=6):
    """Convert integer to a fixed-width binary vector (list of ints)."""
    return [int(b) for b in format(x, f"0{width}b")]

# Load the cancer type mapping, assign each unique study abbreviation an index,
# and then map it to its 6-dimensional binary vector.
df_ct = pd.read_csv(CANCER_TYPE_MAPPING_CSV)
unique_types = sorted(df_ct["Study Abbreviation"].unique())
cancer_type_mapping = {ct: int_to_binary_vector(i, 6) for i, ct in enumerate(unique_types)}
print("Cancer type mapping:", cancer_type_mapping)

##########################################
# Loss Function (Partial Ranking Loss)
##########################################
def partial_ranking_loss(risk, Y_label_T, Y_label_E, eps=1e-6):
    B, L = risk.shape
    assert L == 1, "Risk shape should be (B, 1)"
    risk = torch.clamp(risk, min=-50, max=50)
    one_vector = torch.ones_like(Y_label_T)
    mat_A = ((torch.matmul(one_vector, Y_label_T.T) - torch.matmul(Y_label_T, one_vector.T)) > 0).float()
    mat_B = ((torch.matmul(one_vector, Y_label_T.T) - torch.matmul(Y_label_T, one_vector.T)) == 0).float()
    for i in range(B):
        mat_B[i, i+1:] = 0
    exp_H = torch.exp(risk)
    mat_C = ((mat_A + mat_B) > 0).float()
    R = torch.sum(mat_C * (exp_H.T), dim=-1, keepdim=True) + eps
    loss = -torch.mean(Y_label_E * (risk - torch.log(R)))
    return loss

##########################################
# Dataset for Preprocessed Sequences (Jina)
##########################################
class PreprocessedSequenceDataset(Dataset):
    def __init__(self, df, token_col="gene_embed_seq", cancer_type_mapping=None):
        """
        Expects a DataFrame with columns:
          - token_col: each row is a list/array of tokens (each token is a dict with keys:
              "gene", "embedding", "score", "cna").
          - "OS.time": survival time.
          - "OS": event indicator.
          - "CANCER_TYPE_ACRONYM": cancer type abbreviation.
        If a token list is empty, a default token is created.
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
        print(f"Samples with default tokens: {self.default_token_count} "
              f"({default_percentage:.2f}% of {total_samples} samples)")
        
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
        
        # Process cancer type: retrieve from the row and map to binary vector.
        cancer_type_acronym = row.get("CANCER_TYPE_ACRONYM", None)
        if cancer_type_acronym is None or cancer_type_acronym not in self.cancer_type_mapping:
            ct_vector = [0]*6
        else:
            ct_vector = self.cancer_type_mapping[cancer_type_acronym]
        cancer_type_tensor = torch.tensor(ct_vector, dtype=torch.float)
        
        time = torch.tensor(row["OS.time"], dtype=torch.float)
        event = torch.tensor(row["OS"], dtype=torch.float)
        return embeddings, scores, cnas, cancer_type_tensor, time, event

##########################################
# Collate Function for Padding
##########################################
def collate_fn_preprocessed(batch):
    emb_list, score_list, cna_list, cancer_type_list, times, events = zip(*batch)
    padded_emb = torch.nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in emb_list],
                                                  batch_first=True, padding_value=0.0)
    padded_scores = torch.nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in score_list],
                                                     batch_first=True, padding_value=0.0)
    padded_cnas = torch.nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in cna_list],
                                                  batch_first=True, padding_value=0.0)
    # Cancer type: each is a vector of shape (6,), so stack directly.
    cancer_types = torch.stack(cancer_type_list)
    
    lengths = torch.tensor([len(seq) for seq in emb_list], dtype=torch.long)
    B, L_max, _ = padded_emb.shape
    mask = torch.zeros((B, L_max), dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l < L_max:
            mask[i, l:] = True  # Mark padded positions.
    times = torch.stack(times)
    events = torch.stack(events)
    return padded_emb, padded_scores, padded_cnas, cancer_types, times, events, mask

##########################################
# Transformer Survival Model for Preprocessed Input (Jina)
# with polyphen, CNA, and cancer type MLPs.
##########################################
class PreprocessedTransformerSurvivalModel(nn.Module):
    def __init__(self, d_gene, d_model=256, polyphen_hidden_dim=128, nhead=4, dropout=0.1):
        """
        d_gene: dimension of the precomputed gene embedding.
        d_model: token dimension.
        polyphen_hidden_dim: hidden dimension for the polyphen, CNA, and cancer type MLPs.
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
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dropout=dropout, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.final_linear = nn.Linear(d_model, 1)
        
    def forward(self, emb, scores, cnas, cancer_type, src_key_padding_mask=None):
        # Project gene embeddings: (B, L, d_gene) -> (B, L, d_model)
        gene_proj = self.gene_linear(emb)
        # Project polyphen scores: (B, L, 1) -> (B, L, d_model)
        polyphen_proj = self.polyphen_mlp(scores.unsqueeze(-1))
        # Project CNA values: (B, L, 1) -> (B, L, d_model)
        cna_proj = self.cna_mlp(cnas.unsqueeze(-1))
        # Project cancer type: (B, 6) -> (B, d_model) then unsqueeze to (B, 1, d_model)
        cancer_type_proj = self.cancer_type_mlp(cancer_type).unsqueeze(1)
        # Sum projections; note that cancer_type_proj is broadcast across the token dimension.
        token_emb = gene_proj + polyphen_proj + cna_proj + cancer_type_proj
        # Prepare for transformer encoder: transpose to (L, B, d_model)
        token_emb = token_emb.transpose(0, 1)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(token_emb.device)
        transformer_out = self.transformer_encoder(token_emb, src_key_padding_mask=src_key_padding_mask)
        transformer_out = transformer_out.transpose(0, 1)  # Back to (B, L, d_model)
        # Mean-pool over non-padded tokens.
        if src_key_padding_mask is not None:
            valid_counts = (~src_key_padding_mask).sum(dim=1, keepdim=True).float()  # (B, 1)
            mask_expanded = (~src_key_padding_mask).unsqueeze(-1).float()             # (B, L, 1)
            pooled = (transformer_out * mask_expanded).sum(dim=1) / valid_counts
        else:
            pooled = transformer_out.mean(dim=1)
        risk = self.final_linear(pooled)
        return risk

##########################################
# Evaluation Function
##########################################
def evaluate_model(model, dataloader, device):
    model.eval()
    all_T, all_E, all_risk = [], [], []
    losses = []
    with torch.no_grad():
        for emb_batch, score_batch, cna_batch, cancer_type_batch, T_batch, E_batch, mask in dataloader:
            emb_batch = emb_batch.to(device)
            score_batch = score_batch.to(device)
            cna_batch = cna_batch.to(device)
            cancer_type_batch = cancer_type_batch.to(device)
            T_batch = T_batch.to(device)
            E_batch = E_batch.to(device)
            mask = mask.to(device)
            risk = model(emb_batch, score_batch, cna_batch, cancer_type_batch, src_key_padding_mask=mask)
            loss = partial_ranking_loss(risk, T_batch.unsqueeze(1), E_batch.unsqueeze(1))
            losses.append(loss.item())
            all_T.append(T_batch.cpu().numpy())
            all_E.append(E_batch.cpu().numpy())
            all_risk.append(risk.cpu().numpy())
    avg_loss = np.mean(losses)
    all_T = np.concatenate(all_T).squeeze()
    all_E = np.concatenate(all_E).squeeze()
    all_risk = np.concatenate(all_risk).squeeze()
    c_index = concordance_index(all_T, -all_risk, all_E)
    return avg_loss, c_index, all_T, all_risk

##########################################
# Training Function
##########################################
def train_model_fn(train_loader, val_loader, model, device, max_epochs=100, patience=20):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_cindex = 0.0
    best_epoch = 0
    best_model_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0
    history = []
    for epoch in range(1, max_epochs+1):
        model.train()
        train_losses = []
        for emb_batch, score_batch, cna_batch, cancer_type_batch, T_batch, E_batch, mask in train_loader:
            emb_batch = emb_batch.to(device)
            score_batch = score_batch.to(device)
            cna_batch = cna_batch.to(device)
            cancer_type_batch = cancer_type_batch.to(device)
            T_batch = T_batch.to(device)
            E_batch = E_batch.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            risk = model(emb_batch, score_batch, cna_batch, cancer_type_batch, src_key_padding_mask=mask)
            loss = partial_ranking_loss(risk, T_batch.unsqueeze(1), E_batch.unsqueeze(1))
            if torch.isnan(loss):
                print("NaN loss at epoch", epoch)
                break
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        if not train_losses:
            break
        train_loss_epoch = np.mean(train_losses)
        val_loss, val_cindex, _, _ = evaluate_model(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss_epoch, "val_loss": val_loss, "val_cindex": val_cindex})
        print(f"Epoch {epoch:02d}: Train Loss = {train_loss_epoch:.4f}, Val Loss = {val_loss:.4f}, Val C-index = {val_cindex:.4f}")
        if val_loss < best_val_loss or val_cindex > best_cindex:
            best_val_loss = min(best_val_loss, val_loss)
            best_cindex = max(best_cindex, val_cindex)
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping at epoch", epoch)
            break
    model.load_state_dict(best_model_state)
    return model, best_epoch, best_val_loss, history

##########################################
# Plotting Function
##########################################
def plot_history(history, save_dir, fold):
    h_df = pd.DataFrame(history)
    plt.figure(figsize=(10,5))
    plt.plot(h_df['epoch'], h_df['train_loss'], label='Train Loss')
    plt.plot(h_df['epoch'], h_df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss Curve - Fold {fold}')
    plt.legend()
    loss_path = os.path.join(save_dir, f'loss_curve_fold_{fold}.png')
    plt.savefig(loss_path)
    plt.close()

    plt.figure(figsize=(10,5))
    plt.plot(h_df['epoch'], h_df['val_cindex'], label='Val C-index', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('C-index')
    plt.title(f'Val C-index Curve - Fold {fold}')
    plt.legend()
    cindex_path = os.path.join(save_dir, f'cindex_curve_fold_{fold}.png')
    plt.savefig(cindex_path)
    plt.close()
    print(f"Saved plots for fold {fold} in {save_dir}")

##########################################
# Main Training Loop Over Folds
##########################################
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    folds_metrics = []

    for fold in range(1, NUM_FOLDS+1):
        print(f"\n=== Processing Fold {fold} ===")
        fold_save_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
        os.makedirs(fold_save_dir, exist_ok=True)

        # Load train, validation, test parquet files from the fold folder.
        fold_dir = os.path.join(INPUT_DIR, f"fold_{fold}")
        train_path = os.path.join(fold_dir, "train.parquet")
        val_path   = os.path.join(fold_dir, "val.parquet")
        test_path  = os.path.join(fold_dir, "test.parquet")
        train_df = pd.read_parquet(train_path, engine="pyarrow")
        val_df   = pd.read_parquet(val_path, engine="pyarrow")
        test_df  = pd.read_parquet(test_path, engine="pyarrow")
        # Keep only required columns (including cancer type)
        cols = ["gene_embed_seq", "OS.time", "OS", "CANCER_TYPE_ACRONYM"]
        train_df = train_df[cols]
        val_df   = val_df[cols]
        test_df  = test_df[cols]

        # Create dataset objects with the cancer type mapping.
        train_dataset = PreprocessedSequenceDataset(train_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)
        val_dataset   = PreprocessedSequenceDataset(val_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)
        test_dataset  = PreprocessedSequenceDataset(test_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_preprocessed)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed)

        # Unpack a sample batch to infer embedding dimension.
        sample_emb, sample_scores, sample_cnas, sample_cancer_type, sample_time, sample_event, sample_mask = next(iter(train_loader))
        print("Sample batch shape:", sample_emb.shape)
        d_gene = sample_emb.shape[-1]

        # Initialize model.
        model = PreprocessedTransformerSurvivalModel(d_gene=d_gene, d_model=256,
                                                       polyphen_hidden_dim=128, nhead=4, dropout=0.1)
        model.to(device)
        print(f"Training transformer model for fold {fold}")
        model, best_epoch, best_val_loss, history = train_model_fn(train_loader, val_loader, model, device,
                                                                   max_epochs=100, patience=20)
        model_save_path = os.path.join(fold_save_dir, f"best_model_fold_{fold}.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved best model for fold {fold} to {model_save_path}")

        train_loss, train_cindex, _, _ = evaluate_model(model, train_loader, device)
        val_loss, val_cindex, _, _ = evaluate_model(model, val_loader, device)
        test_loss, test_cindex, test_T, test_risk = evaluate_model(model, test_loader, device)
        risk_mean = np.mean(test_risk)
        risk_std = np.std(test_risk)
        try:
            corr, _ = pearsonr(test_T, test_risk)
        except Exception as e:
            print("Pearson correlation error:", e)
            corr = np.nan

        fold_metrics = {
            "fold": fold,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "train_loss": train_loss,
            "train_cindex": train_cindex,
            "val_loss": val_loss,
            "val_cindex": val_cindex,
            "test_loss": test_loss,
            "test_cindex": test_cindex,
            "test_risk_mean": risk_mean,
            "test_risk_std": risk_std,
            "test_risk_OS_corr": corr
        }
        folds_metrics.append(fold_metrics)
        print(f"Fold {fold} metrics:")
        for key, value in fold_metrics.items():
            print(f"  {key}: {value}")

        risk_scores_path = os.path.join(fold_save_dir, f"test_risk_scores_fold_{fold}.csv")
        pd.DataFrame({"OS.time": test_T, "risk": test_risk}).to_csv(risk_scores_path, index=False)
        history_df = pd.DataFrame(history)
        history_save_path = os.path.join(fold_save_dir, f"history_fold_{fold}.csv")
        history_df.to_csv(history_save_path, index=False)
        plot_history(history, fold_save_dir, fold)

        del train_loader, val_loader, test_loader

    metrics_df = pd.DataFrame(folds_metrics)
    metrics_csv_path = os.path.join(OUTPUT_DIR, "folds_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved folds metrics to {metrics_csv_path}")

if __name__ == "__main__":
    main()
