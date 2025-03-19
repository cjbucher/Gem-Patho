#!/usr/bin/env python
"""
Transformer-based survival model using preprocessed Jina embeddings, polyphen scores, and CNA values.
Assumes that your preprocessed parquet files (saved in, e.g.,
/home/chb3333/yulab/chb3333/data_extraction/data_prep/kfolds_CNA/genename_polyphen_summed/jinaai)
contain a column ("jinaai_embds") where each row is a list of token dictionaries.
Each token dictionary must have the keys:
  - "gene": gene name (string)
  - "embedding": gene embedding (list of floats)
  - "score": polyphen score (a scalar)
  - "cna": copy number alteration value (a scalar)

The model will:
  - Project the gene embedding via a linear layer.
  - Project the polyphen score via a 2-layer MLP (with GELU activation) into a vector.
  - Project the CNA value via a 2-layer MLP (identical architecture as the polyphen branch) into a vector.
  - Sum all three projections to form the token representation.
  - Process the (padded) sequence with 2 transformer encoder layers (using GELU).
  - Mean-pool over non-padded tokens and predict a risk score.
    
Usage example:
  python jina_polyphen.py \
      --input_dir /home/chb3333/yulab/chb3333/data_extraction/data_prep/kfolds_CNA/genename_polyphen/jinaai \
      --output_dir /home/chb3333/yulab/chb3333/models_cna/gene_name_polyphen_sum/jina_polyphen \
      --num_folds 10
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import copy
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index  # for c-index
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    def __init__(self, df, token_col="jinaai_embds"):
        """
        Expects a DataFrame with columns:
          - token_col: each row is a numpy array of tokens, where each token is a dict with keys:
              "gene", "embedding", "score", and "cna".
          - "OS.time": survival time.
          - "OS": event indicator.
        The embedding dimension is inferred from the first non-empty token list.
        Uses a default CNA value of 0.0 if missing.
        """
        self.df = df.reset_index(drop=True)  # Ensure sequential indexing.
        self.token_col = token_col

        # Infer the gene embedding dimension from the first non-empty token list.
        for idx in range(len(self.df)):
            tokens = self.df.iloc[idx][token_col]
            if len(tokens) > 0:
                self.genename_dim = len(tokens[0]["embedding"])
                break

        # Count rows that have an empty token list.
        self.default_token_count = 0
        for idx in range(len(self.df)):
            tokens = self.df.iloc[idx][token_col]
            if isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            if tokens is None or (hasattr(tokens, '__len__') and len(tokens) == 0):
                self.default_token_count += 1

        total_samples = len(self.df)
        default_percentage = (self.default_token_count / total_samples) * 100
        print(f"Number of samples with default tokens: {self.default_token_count} "
              f"({default_percentage:.2f}% of {total_samples} samples)")
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = row[self.token_col]
        # Convert tokens to list if needed.
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        if tokens is None or (hasattr(tokens, '__len__') and len(tokens) == 0):
            tokens = [{"gene": "", "embedding": [0.0]*self.genename_dim, "score": 0.0, "cna": 0.0}]
            
        embeddings = [torch.tensor(token["embedding"], dtype=torch.float) for token in tokens]
        scores = [torch.tensor(token["score"], dtype=torch.float) for token in tokens]
        # Use default CNA value of 0.0 if missing.
        cnas = [torch.tensor(token.get("cna", 0.0), dtype=torch.float) for token in tokens]
        time = torch.tensor(row["OS.time"], dtype=torch.float)
        event = torch.tensor(row["OS"], dtype=torch.float)
        return embeddings, scores, cnas, time, event

##########################################
# Collate Function for Padding (Preprocessed)
##########################################
def collate_fn_preprocessed(batch):
    emb_list, score_list, cna_list, times, events = zip(*batch)
    padded_emb = torch.nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in emb_list],
                                                  batch_first=True, padding_value=0.0)
    padded_scores = torch.nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in score_list],
                                                     batch_first=True, padding_value=0.0)
    padded_cnas = torch.nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in cna_list],
                                                  batch_first=True, padding_value=0.0)
    lengths = torch.tensor([len(seq) for seq in emb_list], dtype=torch.long)
    B, L_max, _ = padded_emb.shape
    mask = torch.zeros((B, L_max), dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l < L_max:
            mask[i, l:] = True  # Padded positions marked as True.
    times = torch.stack(times)
    events = torch.stack(events)
    return padded_emb, padded_scores, padded_cnas, times, events, mask

##########################################
# Transformer Survival Model for Preprocessed Input (Jina)
# with polyphen and CNA MLPs and summing of projections.
##########################################
class PreprocessedTransformerSurvivalModel(nn.Module):
    def __init__(self, d_gene, d_model=256, polyphen_hidden_dim=128, nhead=4, dropout=0.1):
        """
        d_gene: dimension of the precomputed gene embedding.
        d_model: token dimension.
        polyphen_hidden_dim: hidden dimension for both polyphen and CNA MLPs.
        """
        super(PreprocessedTransformerSurvivalModel, self).__init__()
        self.gene_linear = nn.Linear(d_gene, d_model)

        # Polyphen Projection
        self.polyphen_mlp = nn.Sequential(
            nn.Linear(1, polyphen_hidden_dim),
            nn.GELU(),
            nn.Linear(polyphen_hidden_dim, d_model)
        )
        # CNA Projection 
        self.cna_mlp = nn.Sequential(
            nn.Linear(1, polyphen_hidden_dim),
            nn.GELU(),
            nn.Linear(polyphen_hidden_dim, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.final_linear = nn.Linear(d_model, 1)

    def forward(self, emb, scores, cnas, src_key_padding_mask=None):
        # Project gene embeddings: transforms (B, L, d_gene) -> (B, L, d_model)
        gene_proj = self.gene_linear(emb)
        # Project polyphen scores: unsqueeze to add a feature dimension, then transform -> (B, L, d_model)
        polyphen_proj = self.polyphen_mlp(scores.unsqueeze(-1))
        # Project CNA values: unsqueeze and transform -> (B, L, d_model)
        cna_proj = self.cna_mlp(cnas.unsqueeze(-1))
        # Sum the projections to form the token representation.
        token_emb = gene_proj + polyphen_proj + cna_proj  # (B, L, d_model)
        # Transpose for transformer encoder input: (L, B, d_model)
        token_emb = token_emb.transpose(0, 1)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(token_emb.device)
        transformer_out = self.transformer_encoder(token_emb, src_key_padding_mask=src_key_padding_mask)
        transformer_out = transformer_out.transpose(0, 1)  # Back to (B, L, d_model)
        # Pooling over valid (non-padded) tokens.
        if src_key_padding_mask is not None:
            valid_counts = (~src_key_padding_mask).sum(dim=1, keepdim=True).float()  # (B, 1)
            mask_expanded = (~src_key_padding_mask).unsqueeze(-1).float()             # (B, L, 1)
            pooled = (transformer_out * mask_expanded).sum(dim=1) / valid_counts
        else:
            pooled = transformer_out.mean(dim=1)
        risk = self.final_linear(pooled)  # (B, 1)
        return risk

##########################################
# Evaluation Function
##########################################
def evaluate_model(model, dataloader, device):
    model.eval()
    all_T, all_E, all_risk = [], [], []
    losses = []
    with torch.no_grad():
        for emb_batch, score_batch, cna_batch, T_batch, E_batch, mask in dataloader:
            emb_batch = emb_batch.to(device)
            score_batch = score_batch.to(device)
            cna_batch = cna_batch.to(device)
            T_batch = T_batch.to(device)
            E_batch = E_batch.to(device)
            mask = mask.to(device)
            risk = model(emb_batch, score_batch, cna_batch, src_key_padding_mask=mask)
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
# Main Training Loop Over Folds
##########################################
if __name__ == "__main__":
    parser_main = argparse.ArgumentParser(
        description="Train transformer survival model on preprocessed Jina patient data with CNA integration."
    )
    parser_main.add_argument(
        "--input_dir",
        type=str,
        default="/home/chb3333/yulab/chb3333/data_extraction/data_prep/kfolds_CNA/genename_polyphen/jinaai",
        help="Directory containing preprocessed parquet files (Jina embeddings)."
    )
    parser_main.add_argument(
        "--output_dir",
        type=str,
        default="/home/chb3333/yulab/chb3333/models_cna/gene_name_polyphen_sum/jina_polyphen",
        help="Directory to save models, risk scores, and training history."
    )
    parser_main.add_argument(
        "--num_folds",
        type=int,
        default=10,
        help="Number of k-folds to process."
    )
    args_main = parser_main.parse_args()

    os.makedirs(args_main.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    folds_metrics = []

    for fold in tqdm(range(args_main.num_folds), desc="Processing folds"):
        print(f"\n=== Processing Fold {fold} ===")
        fold_save_dir = os.path.join(args_main.output_dir, f"fold_{fold}")
        os.makedirs(fold_save_dir, exist_ok=True)

        # Load parquet files for train, val, test.
        train_path = os.path.join(args_main.input_dir, f"train_fold_{fold}_jinaai_embds.parquet")
        val_path   = os.path.join(args_main.input_dir, f"val_fold_{fold}_jinaai_embds.parquet")
        test_path  = os.path.join(args_main.input_dir, f"test_fold_{fold}_jinaai_embds.parquet")
        train_df = pd.read_parquet(train_path, engine="pyarrow")
        val_df   = pd.read_parquet(val_path, engine="pyarrow")
        test_df  = pd.read_parquet(test_path, engine="pyarrow")
        # Keep only required columns.
        train_df = train_df[["jinaai_embds", "OS.time", "OS"]]
        val_df   = val_df[["jinaai_embds", "OS.time", "OS"]]
        test_df  = test_df[["jinaai_embds", "OS.time", "OS"]]

        # Create Dataset objects.
        train_dataset = PreprocessedSequenceDataset(train_df, token_col="jinaai_embds")
        val_dataset   = PreprocessedSequenceDataset(val_df, token_col="jinaai_embds")
        test_dataset  = PreprocessedSequenceDataset(test_df, token_col="jinaai_embds")

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_preprocessed)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed)

        # Unpack a sample batch.
        sample_emb, sample_scores, sample_cnas, sample_time, sample_event, sample_mask = next(iter(train_loader))
        print("Sample batch shape:", sample_emb.shape)  # (B, L, d_emb)
        d_gene = sample_emb.shape[-1]

        model = PreprocessedTransformerSurvivalModel(d_gene=d_gene, d_model=256,
                                                       polyphen_hidden_dim=128, nhead=4, dropout=0.1)
        model.to(device)

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
                for emb_batch, score_batch, cna_batch, T_batch, E_batch, mask in train_loader:
                    emb_batch = emb_batch.to(device)
                    score_batch = score_batch.to(device)
                    cna_batch = cna_batch.to(device)
                    T_batch = T_batch.to(device)
                    E_batch = E_batch.to(device)
                    mask = mask.to(device)
                    optimizer.zero_grad()
                    risk = model(emb_batch, score_batch, cna_batch, src_key_padding_mask=mask)
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

        print("Training transformer model for fold", fold)
        model, best_epoch, best_val_loss, history = train_model_fn(train_loader, val_loader, model, device, max_epochs=100, patience=20)
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
        plot_history(history, fold_save_dir, fold)

        del train_loader, val_loader, test_loader

    metrics_df = pd.DataFrame(folds_metrics)
    metrics_csv_path = os.path.join(args_main.output_dir, "folds_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"\nSaved folds metrics to {metrics_csv_path}")
