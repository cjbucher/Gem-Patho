#!/usr/bin/env python
"""
Transformer-based survival model using preprocessed Jina embeddings,
polyphen scores, CNA values, cancer type information, and description embeddings.

Data files are assumed to be in:
  Input k-folds:
    /home/chb3333/yulab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold/
Model outputs (for multiple hyperparameter experiments) will be saved in:
  /home/chb3333/yulab/chb3333/gem-patho/models/learning_phases_scheduling/v4_extreme
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
INPUT_DIR = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold"
BASE_OUTPUT_DIR = "/home/chb3333/yulab/chb3333/gem-patho/models/learning_phases_scheduling/v4_extreme"
NUM_FOLDS = 10
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
print("Base output directory:", BASE_OUTPUT_DIR)

# Cancer type mapping CSV – maps study abbreviations (e.g., "LAML") to study names.
CANCER_TYPE_MAPPING_CSV = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/tcga_study_abbreviations.csv"

def int_to_binary_vector(x, width=6):
    """Convert integer to a fixed-width binary vector (list of ints)."""
    return [int(b) for b in format(x, f"0{width}b")]

# Load cancer type mapping.
df_ct = pd.read_csv(CANCER_TYPE_MAPPING_CSV)
unique_types = sorted(df_ct["Study Abbreviation"].unique())
cancer_type_mapping = {ct: int_to_binary_vector(i, 6) for i, ct in enumerate(unique_types)}
print("Cancer type mapping:", cancer_type_mapping)

##########################################
# Phase-Aware Loss Function
##########################################
class PhaseAwareLoss:
    @staticmethod
    def compute(risk, times, events, cancer_type_indices, phase=0,
                transition_percentage=0.9, final_percentage=0.8):
        B = risk.shape[0]
        risk = torch.clamp(risk, min=-50, max=50)
        
        # Compute differences between times.
        diff = times.unsqueeze(0) - times.unsqueeze(1)
        # mat_A: pairs where later time > earlier time.
        mat_A = (diff > 0).float()
        if phase == 0:
            mat_B = (diff == 0).float()
            for i in range(B):
                mat_B[i, i+1:] = 0
            pair_mask = torch.ones((B, B), device=risk.device)
        else:
            mat_B = (diff == 0).float().triu(diagonal=1)
            valid_mask = (cancer_type_indices != -1).float()
            same_type = (cancer_type_indices.unsqueeze(1) == cancer_type_indices.unsqueeze(0)).float()
            if phase == 1:
                # Use transition_percentage for phase 1.
                rand_mask = (torch.rand_like(same_type) < transition_percentage).float()
                pair_mask = (same_type * 1 + (1 - same_type) * (1 - rand_mask)) * valid_mask
            else:  # phase 2
                # Use final_percentage for phase 2.
                rand_mask = (torch.rand_like(same_type) < final_percentage).float()
                pair_mask = (same_type * 1 + (1 - same_type) * (1 - rand_mask)) * valid_mask
        
        mat_A *= pair_mask
        mat_B *= pair_mask
        
        exp_risk = torch.exp(risk)
        R = torch.sum((mat_A + mat_B) * exp_risk.T, dim=1) + 1e-6
        loss = -torch.mean(events * (risk.squeeze() - torch.log(R)))
        return loss

##########################################
# Dataset for Preprocessed Sequences (Jina)
##########################################
class PreprocessedSequenceDataset(Dataset):
    def __init__(self, df, token_col="gene_embed_seq", cancer_type_mapping=None):
        """
        Expects a DataFrame with columns:
          - token_col: a list/array of tokens (each token is a dict with keys:
              "gene", "embedding", "score", "cna").
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
        return embeddings, scores, cnas, cancer_type_tensor, description, time, event

##########################################
# Collate Function for Padding
##########################################
def collate_fn_preprocessed(batch):
    emb_list, score_list, cna_list, cancer_type_list, desc_list, times, events = zip(*batch)
    padded_emb = torch.nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in emb_list],
                                                  batch_first=True, padding_value=0.0)
    padded_scores = torch.nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in score_list],
                                                     batch_first=True, padding_value=0.0)
    padded_cnas = torch.nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in cna_list],
                                                  batch_first=True, padding_value=0.0)
    cancer_types = torch.stack(cancer_type_list)
    descriptions = torch.stack(desc_list)
    
    lengths = torch.tensor([len(seq) for seq in emb_list], dtype=torch.long)
    B, L_max, _ = padded_emb.shape
    mask = torch.zeros((B, L_max), dtype=torch.bool)
    for i, l in enumerate(lengths):
        if l < L_max:
            mask[i, l:] = True
    times = torch.stack(times)
    events = torch.stack(events)
    return padded_emb, padded_scores, padded_cnas, cancer_types, descriptions, times, events, mask

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
        
    def forward(self, emb, scores, cnas, cancer_type, description, src_key_padding_mask=None):
        gene_proj = self.gene_linear(emb)
        polyphen_proj = self.polyphen_mlp(scores.unsqueeze(-1))
        cna_proj = self.cna_mlp(cnas.unsqueeze(-1))
        cancer_type_proj = self.cancer_type_mlp(cancer_type).unsqueeze(1)
        token_emb = gene_proj + polyphen_proj + cna_proj + cancer_type_proj
        
        desc_proj = self.description_linear(description).unsqueeze(1)
        token_emb = torch.cat([desc_proj, token_emb], dim=1)
        
        if src_key_padding_mask is not None:
            new_mask = torch.cat([torch.zeros(src_key_padding_mask.size(0), 1, device=src_key_padding_mask.device,
                                               dtype=src_key_padding_mask.dtype),
                                  src_key_padding_mask], dim=1)
        else:
            new_mask = None
        
        token_emb = token_emb.transpose(0, 1)
        transformer_out = self.transformer_encoder(token_emb, src_key_padding_mask=new_mask)
        transformer_out = transformer_out.transpose(0, 1)
        pooled = transformer_out[:, 0, :]
        risk = self.final_linear(pooled)
        return risk

##########################################
# Evaluation Function
##########################################
def evaluate_model(model, dataloader, device, final_percentage=0.8):
    model.eval()
    all_T, all_E, all_risk = [], [], []
    losses = []
    with torch.no_grad():
        for emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, T_batch, E_batch, mask in dataloader:
            emb_batch = emb_batch.to(device)
            score_batch = score_batch.to(device)
            cna_batch = cna_batch.to(device)
            cancer_type_batch = cancer_type_batch.to(device)
            desc_batch = desc_batch.to(device)
            T_batch = T_batch.to(device)
            E_batch = E_batch.to(device)
            mask = mask.to(device)
            risk = model(emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, src_key_padding_mask=mask)
            ct_idx = torch.argmax(cancer_type_batch, dim=1)
            loss = PhaseAwareLoss.compute(risk, T_batch, E_batch, ct_idx, phase=2,
                                          final_percentage=final_percentage)
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
# Phase Scheduler
##########################################
class PhaseScheduler:
    def __init__(self, warmup=10, transition=10):
        self.warmup = warmup
        self.transition = transition
        
    def get_phase(self, epoch):
        if epoch < self.warmup:
            return 0
        if epoch < self.warmup + self.transition:
            return 1
        return 2

##########################################
# Training Function (with hyperparameters)
##########################################
def train_model_fn(train_loader, val_loader, model, device, max_epochs=100, patience=20,
                   warmup=10, transition=10, transition_percentage=0.9, final_percentage=0.8):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = PhaseScheduler(warmup=warmup, transition=transition)

    history = []
    best_val_loss = float('inf')
    best_cindex = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    best_model_state = copy.deepcopy(model.state_dict())
    old_phase = -1 

    for epoch in range(1, max_epochs+1):
        model.train()
        phase = scheduler.get_phase(epoch)
        train_losses = []
        
        for emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, T_batch, E_batch, mask in train_loader:
            emb_batch = emb_batch.to(device)
            score_batch = score_batch.to(device)
            cna_batch = cna_batch.to(device)
            cancer_type_batch = cancer_type_batch.to(device)
            desc_batch = desc_batch.to(device)
            T_batch = T_batch.to(device)
            E_batch = E_batch.to(device)
            mask = mask.to(device)
            
            ct_idx = torch.argmax(cancer_type_batch, dim=1)
            
            optimizer.zero_grad()
            risk = model(emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch,
                         src_key_padding_mask=mask)
            loss = PhaseAwareLoss.compute(risk, T_batch, E_batch, ct_idx, phase=phase,
                                          transition_percentage=transition_percentage,
                                          final_percentage=final_percentage)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        if not train_losses:
            break
        train_loss_epoch = np.mean(train_losses)
        val_loss, val_cindex, _, _ = evaluate_model(model, val_loader, device,
                                                    final_percentage=final_percentage)
        history.append({"epoch": epoch, "train_loss": train_loss_epoch,
                        "val_loss": val_loss, "val_cindex": val_cindex})

        if old_phase != phase:
            print("Phase:", phase)
            old_phase = phase

        print(f"Epoch {epoch:02d}: Train Loss = {train_loss_epoch:.4f}, "
              f"Val Loss = {val_loss:.4f}, Val C-index = {val_cindex:.4f}")
        
        if val_loss < best_val_loss or val_cindex > best_cindex:
            best_val_loss = min(best_val_loss, val_loss)
            best_cindex = max(best_cindex, val_cindex)
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        if phase == 0 or phase == 1:
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
# Main Training Loop Over Experiments and Folds
##########################################
NUM_FOLDS = 10

all_experiment_results = []

def main():
    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Run one experiment and one k-fold job.")
    parser.add_argument("--job_idx", type=int, required=True,
                        help="Global job index (0-indexed) for experiment-fold combination.")
    args = parser.parse_args()

    # Compute experiment and fold indices from the global job index.
    experiment_idx = args.job_idx // NUM_FOLDS
    fold = (args.job_idx % NUM_FOLDS) + 1  # Folds numbered 1 through 10

    print(f"Global job index: {args.job_idx}")
    print(f"Selected experiment index: {experiment_idx}")
    print(f"Selected fold: {fold}")

    if not torch.cuda.is_available():
        print("No GPU available.")
        # If you want to continue on CPU, you may remove the exit or leave it commented out.
        # import sys
        # sys.exit(1)

    # Define hyperparameter experiments.
    experiments = [
        {"transition_percentage": 1, "final_percentage": 1, "warmup": 0, "transition": 0},
        
        {"transition_percentage": 0.5, "final_percentage": 0.5, "warmup": 10, "transition": 10},

        {"transition_percentage": 0.8, "final_percentage": 1, "warmup": 10, "transition": 10},

   
    ]

    # Check that the experiment index is valid.
    if experiment_idx < 0 or experiment_idx >= len(experiments):
        raise ValueError(f"Invalid experiment index: {experiment_idx}. Must be between 0 and {len(experiments)-1}.")

    # Select the experiment corresponding to the computed experiment index.
    exp = experiments[experiment_idx]
    print(f"Running experiment {experiment_idx}: {exp}")

    # Create output directory for this experiment.
    exp_dir = os.path.join(BASE_OUTPUT_DIR,
        f"exp_{experiment_idx}_tp_{exp['transition_percentage']}_fp_{exp['final_percentage']}_warmup_{exp['warmup']}_transition_{exp['transition']}")
    os.makedirs(exp_dir, exist_ok=True)

    # Process only the selected fold.
    print(f"\n--- Processing Fold {fold} for Experiment {experiment_idx} ---")
    fold_dir = os.path.join(exp_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)

    # Load data for the specific fold.
    fold_folder = os.path.join(INPUT_DIR, f"fold_{fold}")
    train_path = os.path.join(fold_folder, "train.parquet")
    val_path   = os.path.join(fold_folder, "val.parquet")
    test_path  = os.path.join(fold_folder, "test.parquet")
    train_df = pd.read_parquet(train_path, engine="pyarrow")
    val_df   = pd.read_parquet(val_path, engine="pyarrow")
    test_df  = pd.read_parquet(test_path, engine="pyarrow")
    cols = ["gene_embed_seq", "OS.time", "OS", "type", "description_embeddings"]
    train_df = train_df[cols]
    val_df   = val_df[cols]
    test_df  = test_df[cols]

    train_dataset = PreprocessedSequenceDataset(train_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)
    val_dataset   = PreprocessedSequenceDataset(val_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)
    test_dataset  = PreprocessedSequenceDataset(test_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)

    batch_size = 524
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_preprocessed)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed)

    sample_emb, sample_scores, sample_cnas, sample_cancer_type, sample_desc, sample_time, sample_event, sample_mask = next(iter(train_loader))
    d_gene = sample_emb.shape[-1]

    model = PreprocessedTransformerSurvivalModel(d_gene=d_gene, d_model=256,
                                                   polyphen_hidden_dim=128, nhead=4, dropout=0.1,
                                                   desc_dim=sample_desc.shape[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    print(f"Training model for Experiment {experiment_idx}, Fold {fold}")
    model, best_epoch, best_val_loss, history = train_model_fn(
        train_loader, val_loader, model, device, max_epochs=100, patience=20,
        transition_percentage=exp["transition_percentage"],
        final_percentage=exp["final_percentage"],
        warmup=exp["warmup"],
        transition=exp["transition"]
    )
    model_save_path = os.path.join(fold_dir, f"best_model_fold_{fold}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved best model for Fold {fold} to {model_save_path}")

    train_loss, train_cindex, _, _ = evaluate_model(model, train_loader, device,
                                                    final_percentage=exp["final_percentage"])
    val_loss, val_cindex, _, _ = evaluate_model(model, val_loader, device,
                                                final_percentage=exp["final_percentage"])
    test_loss, test_cindex, test_T, test_risk = evaluate_model(model, test_loader, device,
                                                                final_percentage=exp["final_percentage"])
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
    # Save the metrics for this fold.
    pd.DataFrame([fold_metrics]).to_csv(os.path.join(fold_dir, "fold_metrics.csv"), index=False)
    history_df = pd.DataFrame(history)
    history_save_path = os.path.join(fold_dir, "history_fold.csv")
    history_df.to_csv(history_save_path, index=False)
    plot_history(history, fold_dir, fold)

    # Save experiment-level summary for later collection.
    exp_metrics_df = pd.DataFrame([fold_metrics])
    exp_metrics_path = os.path.join(exp_dir, "experiment_fold_metrics.csv")
    exp_metrics_df.to_csv(exp_metrics_path, index=False)
    exp_summary = exp_metrics_df.mean(numeric_only=True).to_dict()
    exp_summary.update(exp)
    all_experiment_results.append(exp_summary)

    # Save summary of all experiments processed so far.
    summary_df = pd.DataFrame(all_experiment_results)
    summary_csv_path = os.path.join(BASE_OUTPUT_DIR, "all_experiments_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved experiment summary to {summary_csv_path}")

if __name__ == "__main__":
    main()
