#!/usr/bin/env python
"""
Transformer-based survival model using only image data from FS files,
with the same Phase loss and experiment structure as the multi-modal script.

Data files are assumed to be in:
  Input k-folds:
    /home/chb3333/yulab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold_20val
Model outputs (for multiple hyperparameter experiments) will be saved in:
  /home/chb3333/yulab/chb3333/gem-patho/models/final_results/only_image_data
"""

import os
import glob
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
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
INPUT_DIR = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold_20val"
BASE_OUTPUT_DIR = "/home/chb3333/yulab/chb3333/gem-patho/models/final_results/only_image_data"
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

# Build a mapping from cancer type abbreviation to a unique integer index.
type_to_index = {ct: i for i, ct in enumerate(unique_types)}
print("Cancer type to index mapping:", type_to_index)

##########################################
# Image Feature Extractor (Inspired by ABMIL)
##########################################
class ImageFeatureExtractor(nn.Module):
    def __init__(self, D_feat, D_inner, D_out, droprate=0.0):
        """
        D_feat: Dimension of raw image features (e.g., 1536 for giga).
        D_inner: Internal dimension for the intermediate representation.
        D_out: Output dimension.
        """
        super(ImageFeatureExtractor, self).__init__()
        self.dimreduction = nn.Sequential(
            nn.Linear(D_feat, D_inner, bias=False),
            nn.ReLU(inplace=True)
        )
        self.attention = nn.Sequential(
            nn.Linear(D_inner, D_inner),
            nn.Tanh(),
            nn.Linear(D_inner, 1)
        )
        self.classifier = nn.Linear(D_inner, D_out)
        self.dropout = nn.Dropout(p=droprate) if droprate > 0 else None

    def forward(self, x):
        # x: Tensor of shape (N, D_feat) for one sample.
        med_feat = self.dimreduction(x)  # (N, D_inner)
        attn_weights = self.attention(med_feat)  # (N, 1)
        attn_weights = torch.softmax(attn_weights, dim=0)  # softmax over N patches
        # Weighted sum over the patch features.
        afeat = torch.sum(attn_weights * med_feat, dim=0)  # (D_inner,)
        if self.dropout is not None:
            afeat = self.dropout(afeat)
        out = self.classifier(afeat)  # (D_out,)
        return out

##########################################
# Dataset for Only Image Data
##########################################
class OnlyImageDataset(Dataset):
    def __init__(self, df):
        """
        Expects a DataFrame with columns:
          - "Case ID", "Project ID": for locating FS image files.
          - "OS.time": survival time.
          - "OS": event indicator.
          - "type": cancer type abbreviation.
        """
        self.df = df.reset_index(drop=True)
        self.num_samples = len(self.df)
        print(f"Loaded {self.num_samples} samples for only image data.")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        time = torch.tensor(row["OS.time"], dtype=torch.float)
        event = torch.tensor(row["OS"], dtype=torch.float)
        ct = row.get("type", "Unknown")
        ct_idx = type_to_index.get(ct, -1)
        ct_idx = torch.tensor(ct_idx, dtype=torch.long)
        
        # --- Load image features from FS file ---
        project_id = row.get("Project ID", None)
        case_id = row.get("Case ID", None)
        if project_id is None or case_id is None:
            raise ValueError("Missing Project ID or Case ID for image data")
        fs_folder = f"/n/data2/hms/dbmi/kyu/lab/NCKU/foundation_model_features/WSI_features/{project_id}-FS/GIGAPATH/20X/pt_files(stain_norm)"
        pattern = os.path.join(fs_folder, f"{case_id}*.pt")
        fs_files = glob.glob(pattern)
        if fs_files and os.path.getsize(fs_files[0]) > 0:
            img_features = torch.load(fs_files[0])  # expected shape: (N, 1536)
        else:
            img_features = torch.zeros((1, 1536))
        # ------------------------------------------------
        
        return time, event, ct_idx, row.get("Case ID", "Unknown"), img_features

##########################################
# Collate Function for Only Image Data
##########################################
def collate_fn_only_image(batch):
    times, events, ct_list, case_ids, img_list = zip(*batch)
    times = torch.stack(times)
    events = torch.stack(events)
    ct_tensor = torch.stack(ct_list)
    return times, events, ct_tensor, list(case_ids), list(img_list)

##########################################
# Simple Image Survival Model
##########################################
class ImageSurvivalModel(nn.Module):
    def __init__(self, d_model=256):
        """
        d_model: Output dimension for the image feature extractor.
        """
        super(ImageSurvivalModel, self).__init__()
        self.image_extractor = ImageFeatureExtractor(D_feat=1536, D_inner=512, D_out=d_model, droprate=0.1)
        self.final_linear = nn.Linear(d_model, 1)
        
    def forward(self, image_features):
        # image_features: list of tensors, each of shape (N, 1536)
        img_feats = []
        for img in image_features:
            feat = self.image_extractor(img)  # (d_model,)
            img_feats.append(feat)
        img_feats = torch.stack(img_feats, dim=0)  # (B, d_model)
        risk = self.final_linear(img_feats)  # (B, 1)
        return risk

##########################################
# Phase-Aware Loss Function (Same as Provided)
##########################################
class PhaseAwareLoss:
    @staticmethod
    def compute(risk, times, events, cancer_type_indices, model, phase=0,
                transition_percentage=0.9, final_percentage=0.8, lambda_reg=1e-4):
        """
        Compute the phase-aware loss with an added L2 regularization term over model parameters.
        """
        B = risk.shape[0]
        risk = torch.clamp(risk, min=-50, max=50)
        diff = times.unsqueeze(0) - times.unsqueeze(1)
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
                rand_mask = (torch.rand_like(same_type) < transition_percentage).float()
                pair_mask = (same_type + (1 - same_type) * (1 - rand_mask)) * valid_mask
            else:
                rand_mask = (torch.rand_like(same_type) < final_percentage).float()
                pair_mask = (same_type + (1 - same_type) * (1 - rand_mask)) * valid_mask
        
        mat_A *= pair_mask
        mat_B *= pair_mask
        
        exp_risk = torch.exp(risk)
        R = torch.sum((mat_A + mat_B) * exp_risk.T, dim=1) + 1e-6
        
        if phase > 0:
            unique_labels, inv, counts = torch.unique(cancer_type_indices, return_inverse=True, return_counts=True)
            scale = 1.0 / counts[inv].float()
            loss = -torch.mean(scale * events * (risk.squeeze() - torch.log(R)))
        else:
            loss = -torch.mean(events * (risk.squeeze() - torch.log(R)))
        
        reg_loss = 0.0
        for param in model.parameters():
            reg_loss += torch.sum(param ** 2)
        loss += lambda_reg * reg_loss
        
        return loss

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
        
        for T_batch, E_batch, ct_batch, _ , img_list in train_loader:
            T_batch = T_batch.to(device, non_blocking=True)
            E_batch = E_batch.to(device, non_blocking=True)
            ct_batch = ct_batch.to(device, non_blocking=True)
            img_list = [img.to(device, non_blocking=True) for img in img_list]
            
            optimizer.zero_grad()
            risk = model(img_list)
            loss = PhaseAwareLoss.compute(risk, T_batch, E_batch, ct_batch, model, phase=phase,
                                          transition_percentage=transition_percentage,
                                          final_percentage=final_percentage)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        if not train_losses:
            break
        train_loss_epoch = np.mean(train_losses)
        val_loss, val_cindex_global, _, _, _, _ = evaluate_model(model, val_loader, device, final_percentage=final_percentage)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss_epoch,
            "val_loss": val_loss,
            "val_cindex_global": val_cindex_global
        })

        if old_phase != phase:
            print("Phase:", phase)
            old_phase = phase

        print(f"Epoch {epoch:02d}: Train Loss = {train_loss_epoch:.4f}, Val Loss = {val_loss:.4f}, Global Val C-index = {val_cindex_global:.4f}")
        
        if val_loss < best_val_loss or val_cindex_global > best_cindex:
            best_val_loss = min(best_val_loss, val_loss)
            best_cindex = max(best_cindex, val_cindex_global)
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
def plot_history(history, save_dir, fold, warmup=None, transition=None):
    h_df = pd.DataFrame(history)
    
    # Global c-index plot.
    plt.figure(figsize=(10,5))
    plt.plot(h_df['epoch'], h_df['val_cindex_global'], label='Global Val C-index')
    plt.xlabel('Epoch')
    plt.ylabel('C-index')
    plt.title(f'Global Val C-index - Fold {fold}')
    if warmup is not None and transition is not None:
        plt.axvline(x=warmup, color='red', linestyle='--', label='Phase 0→1')
        plt.axvline(x=warmup+transition, color='blue', linestyle='--', label='Phase 1→2')
    plt.legend()
    global_cindex_path = os.path.join(save_dir, f'global_cindex_fold_{fold}.png')
    plt.savefig(global_cindex_path)
    plt.close()
    
    history_save_path = os.path.join(save_dir, "history_data.csv")
    h_df.to_csv(history_save_path, index=False)
    print(f"Saved plots and history for fold {fold} in {save_dir}")

##########################################
# Main Training Loop Over Experiments and Folds
##########################################
all_experiment_results = []

def main():
    parser = argparse.ArgumentParser(description="Run one experiment and one k-fold job for only image data with Phase loss.")
    parser.add_argument("--job_idx", type=int, required=True,
                        help="Global job index (0-indexed) for experiment-fold combination.")
    args = parser.parse_args()

    experiment_idx = args.job_idx // NUM_FOLDS
    fold = (args.job_idx % NUM_FOLDS) + 1

    print(f"Global job index: {args.job_idx}")
    print(f"Selected experiment index: {experiment_idx}")
    print(f"Selected fold: {fold}")

    if not torch.cuda.is_available():
        print("No GPU available.")

    experiments = [
        {"transition_percentage": 1, "final_percentage": 1, "warmup": 0, "transition": 0},
        {"transition_percentage": 0.0, "final_percentage": 0.0, "warmup": 10, "transition": 10},
        {"transition_percentage": 0.8, "final_percentage": 1, "warmup": 10, "transition": 10},
        {"transition_percentage": 0.8, "final_percentage": 0.85, "warmup": 10, "transition": 10},
    ]

    if experiment_idx < 0 or experiment_idx >= len(experiments):
        raise ValueError(f"Invalid experiment index: {experiment_idx}.")
    
    exp = experiments[experiment_idx]
    print(f"Running experiment {experiment_idx}: {exp}")

    # For only image processing, we only need these columns.
    cols = ["Case ID", "Project ID", "OS.time", "OS", "type"]

    fold_folder = os.path.join(INPUT_DIR, f"fold_{fold}")
    train_path = os.path.join(fold_folder, "train.parquet")
    val_path = os.path.join(fold_folder, "val.parquet")
    test_path = os.path.join(fold_folder, "test.parquet")
    train_df = pd.read_parquet(train_path, engine="pyarrow")[cols]
    val_df = pd.read_parquet(val_path, engine="pyarrow")[cols]
    test_df = pd.read_parquet(test_path, engine="pyarrow")[cols]

    train_dataset = OnlyImageDataset(train_df)
    val_dataset = OnlyImageDataset(val_df)
    test_dataset = OnlyImageDataset(test_df)

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_only_image)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_only_image)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_only_image)

    model = ImageSurvivalModel(d_model=256)
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

    exp_dir = os.path.join(BASE_OUTPUT_DIR, f"only_image_exp_{experiment_idx}_tp_{exp['transition_percentage']}_fp_{exp['final_percentage']}_warmup_{exp['warmup']}_transition_{exp['transition']}")
    os.makedirs(exp_dir, exist_ok=True)
    fold_dir = os.path.join(exp_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    model_save_path = os.path.join(fold_dir, f"best_model_fold_{fold}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved best model for Fold {fold} to {model_save_path}")

    # Evaluate and save risk predictions.
    for split, loader, prefix in zip(["train", "val"], [train_loader, val_loader], ["train", "val"]):
        avg_loss, global_cindex, all_T, all_risk, all_ct, all_case_ids = evaluate_model(model, loader, device, final_percentage=exp["final_percentage"])
        df_out = pd.DataFrame({
            "Case ID": all_case_ids,
            "OS.time": all_T,
            "risk": all_risk,
            "cancer_type": all_ct,
            "split": prefix
        })
        out_path = os.path.join(fold_dir, f"{prefix}_risk_scores.csv")
        df_out.to_csv(out_path, index=False)
        print(f"Saved {prefix} risk scores to {out_path}")

    train_loss, train_cindex, _, _, _, _ = evaluate_model(model, train_loader, device, final_percentage=exp["final_percentage"])
    val_loss, val_cindex_global, _, _, _, _ = evaluate_model(model, val_loader, device, final_percentage=exp["final_percentage"])
    test_loss, test_cindex, test_T, test_risk, _, _ = evaluate_model(model, test_loader, device, final_percentage=exp["final_percentage"])
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
        "val_cindex_global": val_cindex_global,
        "test_loss": test_loss,
        "test_cindex": test_cindex,
        "test_risk_mean": risk_mean,
        "test_risk_std": risk_std,
        "test_risk_OS_corr": corr
    }
    pd.DataFrame([fold_metrics]).to_csv(os.path.join(fold_dir, "fold_metrics.csv"), index=False)
    history_df = pd.DataFrame(history)
    history_save_path = os.path.join(fold_dir, "history_fold.csv")
    history_df.to_csv(history_save_path, index=False)
    plot_history(history, fold_dir, fold, warmup=exp["warmup"], transition=exp["transition"])

    exp_metrics_df = pd.DataFrame([fold_metrics])
    exp_metrics_path = os.path.join(exp_dir, "experiment_fold_metrics.csv")
    exp_metrics_df.to_csv(exp_metrics_path, index=False)
    exp_summary = exp_metrics_df.mean(numeric_only=True).to_dict()
    exp_summary.update(exp)
    all_experiment_results.append(exp_summary)

    summary_df = pd.DataFrame(all_experiment_results)
    summary_csv_path = os.path.join(BASE_OUTPUT_DIR, "all_experiments_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved experiment summary to {summary_csv_path}")

if __name__ == "__main__":
    main()