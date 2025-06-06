#!/usr/bin/env python
"""
Simplified survival model using only polyphen scores and CNA values.

For each patient, the model uses:
  - A polyphen score vector (each nonzero entry is converted to 1)
  - A CNA vector, constructed by mapping per‐gene CNA values to a fixed gene list.
    If a gene has no CNA value in the sample, its value is set to 0.

Input directory: /home/chb3333/yulab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold_20val  
Output directory: /home/chb3333/yulab/chb3333/gem-patho/models/clean_baseline_results/pure_genetics/one_hot_encoded  
There are folders fold_1, fold_2, …, fold_10.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
import random
import json

# For reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

##########################################
# Define Input and Output Directories
##########################################
INPUT_DIR = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold_20val"
OUTPUT_DIR = "/home/chb3333/yulab/chb3333/gem-patho/models/clean_baseline_results/pure_genetics/one_hot_encoded"

##########################################
# Load Gene List from CSV
##########################################
gene_list_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancer_gene_list_selection/combined_genelist.csv"
gene_df = pd.read_csv(gene_list_path)
if "gene" in gene_df.columns:
    gene_list = gene_df["gene"].tolist()
else:
    gene_list = gene_df.iloc[:, 0].tolist()
print("Loaded gene list of length:", len(gene_list))

##########################################
# Dataset: Only Polyphen and CNA vectors (with cancer type)
##########################################
class SimpleSurvivalDataset(Dataset):
    def __init__(self, df, gene_list):
        """
        Args:
            df (pd.DataFrame): Must include:
                - "polyphen_score": mutation vector (nonzero entries will be converted to 1)
                - "gene_embed_seq": a list of tokens (each with keys "gene" and "cna")
                - "OS.time": survival time
                - "OS": event indicator
                - "Case ID": patient identifier (optional)
                - "type": cancer type (e.g. "LAML")
            gene_list (list): Fixed gene list used for mapping CNA values.
        """
        self.df = df.reset_index(drop=True)
        self.gene_list = gene_list

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Process polyphen vector: ensure it's a 1D array.
        polyphen = row["polyphen_score"]
        if isinstance(polyphen, list):
            polyphen = np.array(polyphen)
        elif not isinstance(polyphen, np.ndarray):
            polyphen = np.array([polyphen])
        polyphen = torch.tensor(polyphen, dtype=torch.float)
        polyphen = (polyphen != 0).float()
        
        # Construct CNA vector based on the fixed gene list.
        cna_vector = [0.0] * len(self.gene_list)
        tokens = row["gene_embed_seq"]
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        if tokens is None or len(tokens) == 0:
            tokens = []
        for token in tokens:
            gene = token.get("gene", "")
            cna_value = token.get("cna", 0.0)
            if gene in self.gene_list:
                idx_gene = self.gene_list.index(gene)
                cna_vector[idx_gene] = cna_value
        cna_vector = torch.tensor(cna_vector, dtype=torch.float)
        
        time = torch.tensor(row["OS.time"], dtype=torch.float)
        event = torch.tensor(row["OS"], dtype=torch.float)
        case_id = row.get("Case ID", "Unknown")
        cancer_type = row.get("type", "Unknown")
        
        return polyphen, cna_vector, time, event, case_id, cancer_type

##########################################
# Collate Function
##########################################
def collate_fn_simple(batch):
    polyphen_batch, cna_batch, times, events, case_ids, cancer_types = zip(*batch)
    polyphen_batch = torch.stack(polyphen_batch)
    cna_batch = torch.stack(cna_batch)
    times = torch.stack(times)
    events = torch.stack(events)
    return polyphen_batch, cna_batch, times, events, list(case_ids), list(cancer_types)

##########################################
# Simple Survival Model
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
        # Concatenate polyphen and CNA vectors.
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
# Loss Function (Cox Partial Likelihood)
##########################################
def compute_loss(risk, times, events, model, lambda_reg=1e-4):
    risk = torch.clamp(risk, min=-50, max=50)
    diff = times.unsqueeze(0) - times.unsqueeze(1)
    mat = (diff > 0).float()
    exp_risk = torch.exp(risk)
    R = torch.sum(mat * exp_risk.T, dim=1) + 1e-6
    loss = -torch.mean(events * (risk.squeeze() - torch.log(R)))
    reg_loss = sum(torch.sum(param ** 2) for param in model.parameters())
    loss += lambda_reg * reg_loss
    return loss

##########################################
# Training Function
##########################################
def train_model(train_loader, val_loader, model, device, max_epochs=100, patience=20, lr=1e-4):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    history = []
    epochs_no_improve = 0
    
    for epoch in range(1, max_epochs+1):
        model.train()
        train_losses = []
        # Note: we ignore cancer type info during training.
        for polyphen_batch, cna_batch, times, events, case_ids, cancer_types in train_loader:
            polyphen_batch = polyphen_batch.to(device)
            cna_batch = cna_batch.to(device)
            times = times.to(device)
            events = events.to(device)
            
            optimizer.zero_grad()
            risk = model(polyphen_batch, cna_batch)
            loss = compute_loss(risk, times, events, model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss_epoch = np.mean(train_losses)
        model.eval()
        val_losses = []
        all_times = []
        all_risks = []
        all_events = []
        with torch.no_grad():
            for polyphen_batch, cna_batch, times, events, case_ids, cancer_types in val_loader:
                polyphen_batch = polyphen_batch.to(device)
                cna_batch = cna_batch.to(device)
                times = times.to(device)
                events = events.to(device)
                risk = model(polyphen_batch, cna_batch)
                loss = compute_loss(risk, times, events, model)
                val_losses.append(loss.item())
                all_times.extend(times.cpu().numpy())
                all_risks.extend(risk.cpu().numpy())
                all_events.extend(events.cpu().numpy())
        val_loss = np.mean(val_losses)
        c_index = concordance_index(all_times, -np.array(all_risks).squeeze(), all_events)
        history.append((epoch, train_loss_epoch, val_loss, c_index))
        print(f"Fold: Epoch {epoch}: Train Loss = {train_loss_epoch:.4f}, Val Loss = {val_loss:.4f}, C-index = {c_index:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print("Early stopping")
            break
    
    model.load_state_dict(best_model_state)
    return model, best_epoch, best_val_loss, history

##########################################
# Main Function: Loop Over All Folds
##########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_fold_metrics = []

    # Loop over folds 1 to 10.
    for fold in range(1, 11):
        print(f"\nProcessing Fold {fold} ...")
        fold_dir = os.path.join(INPUT_DIR, f"fold_{fold}")
        train_path = os.path.join(fold_dir, "train.parquet")
        val_path = os.path.join(fold_dir, "val.parquet")
        test_path = os.path.join(fold_dir, "test.parquet")
        
        # Load data for this fold.
        train_df = pd.read_parquet(train_path, engine="pyarrow")
        val_df = pd.read_parquet(val_path, engine="pyarrow")
        test_df = pd.read_parquet(test_path, engine="pyarrow")
        
        # Create datasets (including cancer type).
        train_dataset = SimpleSurvivalDataset(train_df, gene_list)
        val_dataset = SimpleSurvivalDataset(val_df, gene_list)
        test_dataset = SimpleSurvivalDataset(test_df, gene_list)
        
        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_simple)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_simple)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_simple)
        
        # Determine input dimensions.
        sample_polyphen, sample_cna, _, _, _, _ = train_dataset[0]
        polyphen_dim = sample_polyphen.shape[0]
        cna_dim = sample_cna.shape[0]
        
        model = SimpleSurvivalModel(polyphen_dim, cna_dim)
        model.to(device)
        
        # Train model for this fold.
        model, best_epoch, best_val_loss, history = train_model(train_loader, val_loader, model, device,
                                                                max_epochs=100, patience=20, lr=1e-4)
        print(f"Fold {fold}: Best model obtained at epoch {best_epoch} with validation loss {best_val_loss:.4f}")
        
        # Evaluate on test set.
        model.eval()
        all_times = []
        all_risks = []
        all_events = []
        all_case_ids = []
        all_types = []
        with torch.no_grad():
            for polyphen_batch, cna_batch, times, events, case_ids, cancer_types in test_loader:
                polyphen_batch = polyphen_batch.to(device)
                cna_batch = cna_batch.to(device)
                times = times.to(device)
                events = events.to(device)
                risk = model(polyphen_batch, cna_batch)
                all_times.extend(times.cpu().numpy())
                all_risks.extend(risk.cpu().numpy())
                all_events.extend(events.cpu().numpy())
                all_case_ids.extend(case_ids)
                all_types.extend(cancer_types)
        test_c_index = concordance_index(all_times, -np.array(all_risks).squeeze(), all_events)
        print(f"Fold {fold}: Test C-index: {test_c_index:.4f}")
        
        # Save fold outputs.
        fold_output_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        
        model_save_path = os.path.join(fold_output_dir, "best_model.pth")
        torch.save(model.state_dict(), model_save_path)
        print(f"Fold {fold}: Saved best model to {model_save_path}")
        
        history_df = pd.DataFrame(history, columns=["epoch", "train_loss", "val_loss", "c_index"])
        history_csv_path = os.path.join(fold_output_dir, "training_history.csv")
        history_df.to_csv(history_csv_path, index=False)
        print(f"Fold {fold}: Saved training history to {history_csv_path}")
        
        test_results = pd.DataFrame({
            "Case ID": all_case_ids,
            "OS.time": all_times,
            "risk": np.array(all_risks).squeeze(),
            "OS": all_events,
            "cancer_type": all_types
        })
        test_results_csv_path = os.path.join(fold_output_dir, "test_predictions.csv")
        test_results.to_csv(test_results_csv_path, index=False)
        print(f"Fold {fold}: Saved test predictions to {test_results_csv_path}")
        
        metrics = {"Fold": fold, "Test C-index": test_c_index, "Best Epoch": best_epoch, "Best Val Loss": best_val_loss}
        metrics_path = os.path.join(fold_output_dir, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f)
        print(f"Fold {fold}: Saved evaluation metrics to {metrics_path}")
        
        all_fold_metrics.append(metrics)
    
    # Aggregate results across folds.
    summary_df = pd.DataFrame(all_fold_metrics)
    summary_csv_path = os.path.join(OUTPUT_DIR, "all_folds_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"\nSaved summary of all folds to {summary_csv_path}")

if __name__ == "__main__":
    main()
