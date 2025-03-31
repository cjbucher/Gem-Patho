#!/usr/bin/env python
"""
Transformer-based survival model using preprocessed Jina embeddings,
polyphen scores, CNA values, cancer type information, description embeddings,
and image modality integrated via an ABMIL-inspired image extractor.

Data files are assumed to be in:
  Input k-folds:
    /n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold_20val/
Model outputs (for multiple hyperparameter experiments) will be saved in:
 /n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/models/actual_final_results/only_image/all_batch
"""

import os
import copy
import glob
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
INPUT_DIR = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold_20val"
BASE_OUTPUT_DIR = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/models/actual_final_results/only_image/all_batch"
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
# Stratified Batch Sampler
##########################################
class StratifiedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, shuffle=True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.unique_labels = np.unique(self.labels)
        self.indices_per_label = {label: np.where(self.labels == label)[0].tolist()
                                  for label in self.unique_labels}
        self.num_samples = len(self.labels)
        self.num_batches = int(np.ceil(self.num_samples / self.batch_size))
    
    def __iter__(self):
        indices_per_label = {label: idxs.copy() for label, idxs in self.indices_per_label.items()}
        if self.shuffle:
            for label in self.unique_labels:
                np.random.shuffle(indices_per_label[label])
        for _ in range(self.num_batches):
            batch = []
            for label in self.unique_labels:
                prop = len(self.indices_per_label[label]) / self.num_samples
                n_samples = max(1, int(prop * self.batch_size))
                available = indices_per_label[label]
                if len(available) < n_samples:
                    chosen = np.random.choice(self.indices_per_label[label], n_samples, replace=True).tolist()
                else:
                    chosen = available[:n_samples]
                    indices_per_label[label] = available[n_samples:]
                batch.extend(chosen)
            if len(batch) > self.batch_size:
                batch = np.random.choice(batch, self.batch_size, replace=False).tolist()
            yield batch

    def __len__(self):
        return self.num_batches

##########################################
# Cancer Type–Specific Batch Sampler (for Phase 2)
##########################################
class CancerTypeSpecificSampler(Sampler):
    def __init__(self, labels, batch_size, shuffle=True):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.unique_labels = np.unique(self.labels)
        self.indices_per_label = {label: np.where(self.labels == label)[0].tolist() for label in self.unique_labels}
        self.batches = self._create_batches()
    
    def _create_batches(self):
        batches = []
        for label, indices in self.indices_per_label.items():
            if self.shuffle:
                np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i+self.batch_size]
                batches.append(batch)
        if self.shuffle:
            np.random.shuffle(batches)
        return batches

    def __iter__(self):
        batches = self._create_batches()
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.batches)

##########################################
# Collate Function for Padding (mask removed)
##########################################
def collate_fn_preprocessed(batch):
    # Unpack list of tuples (now including image features).
    emb_list, score_list, cna_list, cancer_type_list, desc_list, times, events, case_ids, img_list = zip(*batch)
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
    return padded_emb, padded_scores, padded_cnas, cancer_types, descriptions, times, events, case_ids, img_list

##########################################
# Unified Loss Function (normalized by cancer type)
##########################################
def compute_loss(risk, times, events, cancer_type_indices, model, lambda_reg=1e-4):
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
# Image Feature Extractor (Inspired by ABMIL)
##########################################
class ImageFeatureExtractor(nn.Module):
    def __init__(self, D_feat, D_inner, D_out, droprate=0.0):
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
        med_feat = self.dimreduction(x)
        attn_weights = self.attention(med_feat)
        attn_weights = torch.softmax(attn_weights, dim=0)
        afeat = torch.sum(attn_weights * med_feat, dim=0)
        if self.dropout is not None:
            afeat = self.dropout(afeat)
        out = self.classifier(afeat)
        return out

##########################################
# Dataset for Preprocessed Sequences (Jina) with Image Modality
##########################################
class PreprocessedSequenceDataset(Dataset):
    def __init__(self, df, token_col="gene_embed_seq", cancer_type_mapping=None):
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
        # --- IMAGE MODALITY INTEGRATION ---
        # Here we load image features from an FS file using "Project ID" and "Case ID".
        project_id = row.get("Project ID", None)
        if project_id is not None:
            fs_folder = f"/n/data2/hms/dbmi/kyu/lab/NCKU/foundation_model_features/WSI_features/{project_id}-FS/GIGAPATH/20X/pt_files(stain_norm)"
            pattern = os.path.join(fs_folder, f"{case_id}*.pt")
            fs_files = glob.glob(pattern)
            if fs_files:
                img_features = torch.load(fs_files[0])  # Expected shape: (N, D_feat) e.g., (N, 1536)
            else:
                img_features = torch.zeros((1, 1536), dtype=torch.float)
        else:
            img_features = torch.zeros((1, 1536), dtype=torch.float)
        # ---------------------------------
        return embeddings, scores, cnas, cancer_type_tensor, description, time, event, case_id, img_features

##########################################
# Transformer Survival Model (with Description Token and Image Modality)
##########################################
class PreprocessedTransformerSurvivalModel(nn.Module):
    def __init__(self, d_gene, d_model=256, polyphen_hidden_dim=128, nhead=4, dropout=0.1, desc_dim=None):
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
        # --- IMAGE MODALITY INTEGRATION ---
        # Image feature extractor (assumes raw image features of dimension 1536).
        self.image_extractor = ImageFeatureExtractor(D_feat=1536, D_inner=512, D_out=d_model, droprate=0.1)
        # Fusion layer: fuse pooled text representation and image representation.
        self.fusion_linear = nn.Linear(d_model, 1)
        # ---------------------------------
        
    def forward(self, emb, scores, cnas, cancer_type, description, image_features):
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
        
        # --- IMAGE MODALITY INTEGRATION ---
        img_feats = []
        for img in image_features:
            img_feat = self.image_extractor(img)
            img_feats.append(img_feat)
        img_feats = torch.stack(img_feats, dim=0)
        #fusion = torch.cat([pooled, img_feats], dim=1)
        risk = self.fusion_linear(img_feats)
        # ---------------------------------
        return risk

##########################################
# Evaluation Function
##########################################
def evaluate_model(model, dataloader, device):
    model.eval()
    all_T, all_E, all_risk, all_ct, all_case_ids = [], [], [], [], []
    losses = []
    missing_cancer_types = []
    with torch.no_grad():
        for emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, T_batch, E_batch, case_ids, img_list in dataloader:
            emb_batch = emb_batch.to(device)
            score_batch = score_batch.to(device)
            cna_batch = cna_batch.to(device)
            cancer_type_batch = cancer_type_batch.to(device)
            desc_batch = desc_batch.to(device)
            T_batch = T_batch.to(device)
            E_batch = E_batch.to(device)
            # Move each image feature tensor to device.
            img_list = [img.to(device) for img in img_list]
            
            risk = model(emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, img_list)
            
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
    
    return avg_loss, global_cindex, cindex_by_type, normalized_avg_cindex, all_T, all_risk, all_ct, all_case_ids, missing_cancer_types

##########################################
# Training Function (Two-Phase Training)
##########################################
def train_model_fn(train_loader, val_loader, model, device,
                   max_phase0_epochs=100, phase2_epochs=10, patience=20, batch_size=64):
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
        for emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, T_batch, E_batch, _ , img_list in train_loader:
            emb_batch = emb_batch.to(device)
            score_batch = score_batch.to(device)
            cna_batch = cna_batch.to(device)
            cancer_type_batch = cancer_type_batch.to(device)
            desc_batch = desc_batch.to(device)
            T_batch = T_batch.to(device)
            E_batch = E_batch.to(device)
            img_list = [img.to(device) for img in img_list]
            
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
            risk = model(emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, img_list)
            loss = compute_loss(risk, T_batch, E_batch, ct_idx, model)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        train_loss_epoch = np.mean(train_losses)
        val_loss, val_cindex_global, val_cindex_by_type, normalized_avg_cindex, _, _, _, _, _ = evaluate_model(
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
    phase0_best_epoch = best_epoch

    print("Entering Phase 2 training for additional {} epochs ...".format(phase2_epochs))
    if phase2_epochs > 0:
        labels = train_loader.dataset.df["type"].tolist()
        cancer_sampler = CancerTypeSpecificSampler(labels, batch_size, shuffle=True)
        phase2_loader = DataLoader(train_loader.dataset, batch_sampler=cancer_sampler, collate_fn=collate_fn_preprocessed)
        for epoch in range(1, phase2_epochs+1):
            model.train()
            phase2_losses = []
            for emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, T_batch, E_batch, _ , img_list in phase2_loader:
                emb_batch = emb_batch.to(device)
                score_batch = score_batch.to(device)
                cna_batch = cna_batch.to(device)
                cancer_type_batch = cancer_type_batch.to(device)
                desc_batch = desc_batch.to(device)
                T_batch = T_batch.to(device)
                E_batch = E_batch.to(device)
                img_list = [img.to(device) for img in img_list]
                
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
                risk = model(emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, img_list)
                loss = compute_loss(risk, T_batch, E_batch, ct_idx, model)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                phase2_losses.append(loss.item())
            
            phase2_loss_epoch = np.mean(phase2_losses)
            val_loss, val_cindex_global, val_cindex_by_type, normalized_avg_cindex, _, _, _, _, _ = evaluate_model(
                model, val_loader, device)
            total_epoch = phase0_epochs_run + epoch
            history.append({
                "phase": 2,
                "epoch": total_epoch,
                "train_loss": phase2_loss_epoch,
                "val_loss": val_loss,
                "val_cindex_global": val_cindex_global,
                "normalized_avg_cindex": normalized_avg_cindex,
                "val_cindex_by_type": val_cindex_by_type
            })
            print(f"Phase 2 Epoch {epoch:02d} (Total Epoch {total_epoch:02d}): Train Loss = {phase2_loss_epoch:.4f}, Val Loss = {val_loss:.4f}, Global Val C-index = {val_cindex_global:.4f}, Normalized Val C-index = {normalized_avg_cindex:.4f}")
            if val_loss < best_val_loss or val_cindex_global > best_cindex:
                best_val_loss = min(best_val_loss, val_loss)
                best_cindex = max(best_cindex, val_cindex_global)
                best_epoch = total_epoch
                best_model_state = copy.deepcopy(model.state_dict())
    else:
        print("Phase 2 training set to 0 epochs; skipping extra training.")

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
# Main Training Loop Over Experiments and Folds
##########################################
all_experiment_results = []

def main():
    parser = argparse.ArgumentParser(description="Run one experiment and one k-fold job.")
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
        {"phase2_epochs": 10},
        {"phase2_epochs": 0},
    ]

    if experiment_idx < 0 or experiment_idx >= len(experiments):
        raise ValueError(f"Invalid experiment index: {experiment_idx}.")
    
    exp = experiments[experiment_idx]
    print(f"Running experiment {experiment_idx}: {exp}")

    cols = ["Case ID", "gene_embed_seq", "OS.time", "OS", "type", "description_embeddings", "Project ID"]

    fold_folder = os.path.join(INPUT_DIR, f"fold_{fold}")
    train_path = os.path.join(fold_folder, "train.parquet")
    val_path = os.path.join(fold_folder, "val.parquet")
    test_path = os.path.join(fold_folder, "test.parquet")
    train_df = pd.read_parquet(train_path, engine="pyarrow")[cols]
    val_df = pd.read_parquet(val_path, engine="pyarrow")[cols]
    test_df = pd.read_parquet(test_path, engine="pyarrow")[cols]

    train_dataset = PreprocessedSequenceDataset(train_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)
    val_dataset = PreprocessedSequenceDataset(val_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)
    test_dataset = PreprocessedSequenceDataset(test_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)

    batch_size = 64
    train_labels = train_df['type'].tolist()
    strat_sampler = StratifiedBatchSampler(train_labels, batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_sampler=strat_sampler, collate_fn=collate_fn_preprocessed)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed)

    sample_emb, sample_scores, sample_cnas, sample_cancer_type, sample_desc, sample_time, sample_event, _, sample_img = next(iter(train_loader))
    d_gene = sample_emb.shape[-1]

    model = PreprocessedTransformerSurvivalModel(d_gene=d_gene, d_model=256,
                                                   polyphen_hidden_dim=128, nhead=4, dropout=0.1,
                                                   desc_dim=sample_desc.shape[-1])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    print(f"Training model for Experiment {experiment_idx}, Fold {fold}")
    model, best_epoch, best_val_loss, history, phase0_epochs_run = train_model_fn(
        train_loader, val_loader, model, device, max_phase0_epochs=100,
        phase2_epochs=exp["phase2_epochs"], patience=20, batch_size=batch_size
    )

    # Save epoch c-indices into a CSV.
    cindices_records = []
    for record in history:
        epoch = record['epoch']
        cindices_records.append({
            "epoch": epoch,
            "cancer_type": "global",
            "c_index": record['val_cindex_global']
        })
        cindices_records.append({
            "epoch": epoch,
            "cancer_type": "normalized_avg",
            "c_index": record['normalized_avg_cindex']
        })
        for ct, c_idx in record['val_cindex_by_type'].items():
            cindices_records.append({
                "epoch": epoch,
                "cancer_type": ct,
                "c_index": c_idx
            })
    df_cindices = pd.DataFrame(cindices_records)
    cindices_csv_path = os.path.join(fold_folder, f"fold_{fold}", "epoch_cindices.csv")
    os.makedirs(os.path.join(fold_folder, f"fold_{fold}"), exist_ok=True)
    df_cindices.to_csv(cindices_csv_path, index=False)
    print(f"Saved epoch c-indices to {cindices_csv_path}")

    exp_dir = os.path.join(BASE_OUTPUT_DIR, f"cancertypebatching_exp_{experiment_idx}_phase2_{exp['phase2_epochs']}")
    os.makedirs(exp_dir, exist_ok=True)
    fold_dir = os.path.join(exp_dir, f"fold_{fold}")
    os.makedirs(fold_dir, exist_ok=True)
    model_save_path = os.path.join(fold_dir, f"best_model_fold_{fold}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Saved best model for Fold {fold} to {model_save_path}")

    for split, loader, prefix in zip(["train", "val"], [train_loader, val_loader], ["train", "val"]):
        avg_loss, global_cindex, cindex_by_type, normalized_avg_cindex, all_T, all_risk, all_ct, all_case_ids, missing_ct = evaluate_model(model, loader, device)
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
        missing_df = pd.DataFrame({"Missing Cancer Types": missing_ct})
        missing_out_path = os.path.join(fold_dir, f"{prefix}_missing_cancer_types.csv")
        missing_df.to_csv(missing_out_path, index=False)
        print(f"Saved missing cancer types for {prefix} split to {missing_out_path}")

    train_loss, train_cindex, _, _, _, _, _, _, _ = evaluate_model(model, train_loader, device)
    val_loss, val_cindex_global, _, normalized_avg_cindex, _, _, _, _, _ = evaluate_model(model, val_loader, device)
    test_loss, test_cindex, test_T, test_risk, _, _, _, all_case_ids, _ = evaluate_model(model, test_loader, device)
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
        "normalized_val_cindex": normalized_avg_cindex,
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
    plot_history(history, fold_dir, fold, phase0_end_epoch=phase0_epochs_run)

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
