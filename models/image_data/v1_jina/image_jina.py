#!/usr/bin/env python
"""
Transformer-based survival model with image data fusion using FS files.

Data files are now assumed to be in:
  Input k-folds:
    /home/chb3333/yulab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold/fold_1
Model outputs will be saved in:
  /home/chb3333/yulab/chb3333/gem-patho/models/image_data/v1_jina

Cancer type information is provided via the "type" column.
It is converted to a 6-dimensional binary vector (using an index mapping loaded from:
  /home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/tcga_study_abbreviations.csv)
and fed through an MLP; its output is summed (broadcast over the token dimension) along with the
gene embedding projections, polyphen, and CNA projections.
Additionally, if available, the description embedding is fed as a separate token (prepended)
and its output (projected via its own linear layer) is used as the pooled representation.
Finally, image features (from FS files) are processed via an image extractor (inspired by ABMIL)
and late-fused with the pooled transformer output.
"""

import os
import copy
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index  # for c-index
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

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
OUTPUT_DIR = "/home/chb3333/yulab/chb3333/gem-patho/models/image_data/v1_jina"
print(OUTPUT_DIR)
NUM_FOLDS = 10

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
    def compute(risk, times, events, cancer_type_indices, phase=0, same_type_weight=2.0):
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
            if phase == 1:
                same_type = (cancer_type_indices.unsqueeze(1) == cancer_type_indices.unsqueeze(0)).float()
                rand_mask = (torch.rand_like(same_type) < 0.75).float()
                pair_mask = (same_type * rand_mask + (1 - rand_mask)) * valid_mask
            else:  # phase 2
                same_type = (cancer_type_indices.unsqueeze(1) == cancer_type_indices.unsqueeze(0)).float()
                pair_mask = (1 + same_type * (same_type_weight - 1)) * valid_mask
        
        mat_A *= pair_mask
        mat_B *= pair_mask
        
        exp_risk = torch.exp(risk)
        R = torch.sum((mat_A + mat_B) * exp_risk.T, dim=1) + 1e-6
        loss = -torch.mean(events * (risk.squeeze() - torch.log(R)))
        return loss

##########################################
# Image Feature Extractor (Inspired by ABMIL)
##########################################
class ImageFeatureExtractor(nn.Module):
    def __init__(self, D_feat, D_inner, D_out, droprate=0.0):
        """
        D_feat: Dimension of raw image features (e.g., 1536 for giga).
        D_inner: Internal dimension for the intermediate representation.
        D_out: Output dimension (should match transformer token dimension).
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
# Dataset for Preprocessed Sequences (Jina) with Image Data
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
          - "description_embeddings": description embeddings.
          - "Project ID" and "Case ID": for locating FS image files.
        Only rows with available and non-empty FS image data will be kept.
        """
        self.df = df.reset_index(drop=True)
        self.token_col = token_col
        self.cancer_type_mapping = cancer_type_mapping if cancer_type_mapping is not None else {}
        self.has_description = "description_embeddings" in self.df.columns
        
        # Filter rows: keep only samples that have an FS file available and non-empty.
        filtered_rows = []
        for idx, row in self.df.iterrows():
            project_id = row.get("Project ID", None)
            case_id = row.get("Case ID", None)
            if project_id is None or case_id is None:
                continue
            fs_folder = f"/n/data2/hms/dbmi/kyu/lab/NCKU/foundation_model_features/WSI_features/{project_id}-FS/GIGAPATH/20X/pt_files(stain_norm)"
            pattern = os.path.join(fs_folder, f"{case_id}*.pt")
            fs_files = glob.glob(pattern)
            # Check file existence and that the file is non-empty (by size).
            if fs_files and os.path.getsize(fs_files[0]) > 0:
                filtered_rows.append(idx)
        self.df = self.df.loc[filtered_rows].reset_index(drop=True)
        print(f"Dataset filtered: {len(self.df)} samples with valid FS image available.")

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
        # Process gene tokens.
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
        
        # Process description embeddings.
        if self.has_description:
            description = torch.tensor(row["description_embeddings"], dtype=torch.float)
        else:
            print("No description embedding found for sample, using zero vector.")
            description = torch.zeros(self.genename_dim, dtype=torch.float)
        
        time = torch.tensor(row["OS.time"], dtype=torch.float)
        event = torch.tensor(row["OS"], dtype=torch.float)
        
        # Load image features from FS file.
        project_id = row["Project ID"]
        case_id = row["Case ID"]
        fs_folder = f"/n/data2/hms/dbmi/kyu/lab/NCKU/foundation_model_features/WSI_features/{project_id}-FS/GIGAPATH/20X/pt_files(stain_norm)"
        pattern = os.path.join(fs_folder, f"{case_id}*.pt")
        fs_files = glob.glob(pattern)
        img_features = torch.load(fs_files[0])  # expected shape: (N, D_feat), e.g., (N, 1536)
        
        return embeddings, scores, cnas, cancer_type_tensor, description, time, event, img_features

##########################################
# Collate Function for Padding
##########################################
def collate_fn_preprocessed(batch):
    # Unpack list of tuples.
    emb_list, score_list, cna_list, cancer_type_list, desc_list, times, events, img_list = zip(*batch)
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
    return padded_emb, padded_scores, padded_cnas, cancer_types, descriptions, times, events, mask, img_list

##########################################
# Transformer Survival Model with Image Fusion
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
        if desc_dim is None:
            desc_dim = d_gene
        self.description_linear = nn.Linear(desc_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                    dropout=dropout, activation="gelu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Image feature extractor: assumes raw image features of dim 1536 (for giga), internal dim 512.
        self.image_extractor = ImageFeatureExtractor(D_feat=1536, D_inner=512, D_out=d_model, droprate=0.1)
        # Fusion layer: concatenated [transformer pooled; image vector] -> risk.
        self.fusion_linear = nn.Linear(2*d_model, 1)
        
    def forward(self, emb, scores, cnas, cancer_type, description, image_features, src_key_padding_mask=None):
        # Project gene embeddings.
        gene_proj = self.gene_linear(emb)  # (B, L, d_model)
        polyphen_proj = self.polyphen_mlp(scores.unsqueeze(-1))  # (B, L, d_model)
        cna_proj = self.cna_mlp(cnas.unsqueeze(-1))              # (B, L, d_model)
        cancer_type_proj = self.cancer_type_mlp(cancer_type).unsqueeze(1)  # (B, 1, d_model)
        token_emb = gene_proj + polyphen_proj + cna_proj + cancer_type_proj  # (B, L, d_model)
        
        # Process description token.
        desc_proj = self.description_linear(description)  # (B, d_model)
        desc_proj = desc_proj.unsqueeze(1)  # (B, 1, d_model)
        
        # Prepend description token.
        token_emb = torch.cat([desc_proj, token_emb], dim=1)  # (B, L+1, d_model)
        
        # Adjust padding mask: description token is always valid.
        if src_key_padding_mask is not None:
            new_mask = torch.cat([torch.zeros(src_key_padding_mask.size(0), 1, device=src_key_padding_mask.device, dtype=src_key_padding_mask.dtype),
                                  src_key_padding_mask], dim=1)
        else:
            new_mask = None
        
        # Transformer expects input shape: (seq_len, batch, d_model)
        token_emb = token_emb.transpose(0, 1)  # (L+1, B, d_model)
        transformer_out = self.transformer_encoder(token_emb, src_key_padding_mask=new_mask)
        transformer_out = transformer_out.transpose(0, 1)  # (B, L+1, d_model)
        pooled = transformer_out[:, 0, :]  # (B, d_model)
        
        # Process image features for each sample in the batch.
        img_feats = []
        for img in image_features:  # each img: (N, 1536)
            img_feat = self.image_extractor(img)  # (d_model,)
            img_feats.append(img_feat)
        img_feats = torch.stack(img_feats, dim=0)  # (B, d_model)
        
        # Late fusion: concatenate pooled transformer representation and image vector.
        fusion = torch.cat([pooled, img_feats], dim=1)  # (B, 2*d_model)
        risk = self.fusion_linear(fusion)  # (B, 1)
        return risk

##########################################
# Evaluation Function
##########################################
def evaluate_model(model, dataloader, device):
    model.eval()
    all_T, all_E, all_risk = [], [], []
    losses = []
    with torch.no_grad():
        for emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, T_batch, E_batch, mask, img_list in dataloader:
            emb_batch = emb_batch.to(device, non_blocking=True)
            score_batch = score_batch.to(device, non_blocking=True)
            cna_batch = cna_batch.to(device, non_blocking=True)
            cancer_type_batch = cancer_type_batch.to(device, non_blocking=True)
            desc_batch = desc_batch.to(device, non_blocking=True)
            T_batch = T_batch.to(device, non_blocking=True)
            E_batch = E_batch.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            img_list = [img.to(device, non_blocking=True) for img in img_list]
            
            risk = model(emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, img_list, src_key_padding_mask=mask)
            ct_idx = torch.argmax(cancer_type_batch, dim=1)
            loss = PhaseAwareLoss.compute(risk, T_batch, E_batch, ct_idx, phase=2)
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
# Training Function with Phase Scheduling
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

def train_model_fn(train_loader, val_loader, model, device, max_epochs=100, patience=20):
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = PhaseScheduler(warmup=10, transition=10)

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
        
        for emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, T_batch, E_batch, mask, img_list in train_loader:
            emb_batch = emb_batch.to(device, non_blocking=True)
            score_batch = score_batch.to(device, non_blocking=True)
            cna_batch = cna_batch.to(device, non_blocking=True)
            cancer_type_batch = cancer_type_batch.to(device, non_blocking=True)
            desc_batch = desc_batch.to(device, non_blocking=True)
            T_batch = T_batch.to(device, non_blocking=True)
            E_batch = E_batch.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            img_list = [img.to(device, non_blocking=True) for img in img_list]
            
            ct_idx = torch.argmax(cancer_type_batch, dim=1)
            optimizer.zero_grad()
            risk = model(emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, img_list, src_key_padding_mask=mask)
            loss = PhaseAwareLoss.compute(risk, T_batch, E_batch, ct_idx, phase=phase)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        if not train_losses:
            break
        train_loss_epoch = np.mean(train_losses)
        val_loss, val_cindex, _, _ = evaluate_model(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": train_loss_epoch, "val_loss": val_loss, "val_cindex": val_cindex})

        if old_phase != phase:
            print("Phase:", phase)
            old_phase = phase

        print(f"Epoch {epoch:02d}: Train Loss = {train_loss_epoch:.4f}, Val Loss = {val_loss:.4f}, Val C-index = {val_cindex:.4f}")
        
        if val_loss < best_val_loss or val_cindex > best_cindex:
            best_val_loss = min(best_val_loss, val_loss)
            best_cindex = max(best_cindex, val_cindex)
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        
        if phase in [0, 1]:
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

        # Load train, validation, and test parquet files from the fold folder.
        fold_dir = os.path.join(INPUT_DIR, f"fold_{fold}")
        train_path = os.path.join(fold_dir, "train.parquet")
        val_path   = os.path.join(fold_dir, "val.parquet")
        test_path  = os.path.join(fold_dir, "test.parquet")
        train_df = pd.read_parquet(train_path, engine="pyarrow")
        val_df   = pd.read_parquet(val_path, engine="pyarrow")
        test_df  = pd.read_parquet(test_path, engine="pyarrow")
        # Keep required columns; note that we now need "Project ID" and "Case ID" for image data.
        cols = ["gene_embed_seq", "OS.time", "OS", "type", "description_embeddings", "Project ID", "Case ID"]
        train_df = train_df[cols]
        val_df   = val_df[cols]
        test_df  = test_df[cols]

        # Create dataset objects.
        train_dataset = PreprocessedSequenceDataset(train_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)
        val_dataset   = PreprocessedSequenceDataset(val_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)
        test_dataset  = PreprocessedSequenceDataset(test_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)

        batch_size = 32
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_preprocessed, pin_memory=True)
        val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed, pin_memory=True)
        test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed, pin_memory=True)

        # Unpack a sample batch to infer embedding dimension.
        sample_emb, sample_scores, sample_cnas, sample_cancer_type, sample_desc, sample_time, sample_event, sample_mask, sample_img = next(iter(train_loader))
        print("Sample batch shape:", sample_emb.shape)
        d_gene = sample_emb.shape[-1]

        # Initialize model.
        model = PreprocessedTransformerSurvivalModel(d_gene=d_gene, d_model=256,
                                                       polyphen_hidden_dim=128, nhead=4, dropout=0.1,
                                                       desc_dim=sample_desc.shape[-1])
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
