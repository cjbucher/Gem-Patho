#!/usr/bin/env python
"""
Script to evaluate k-fold cross-validation predictions from the transformer survival model,
normalize each fold's risk scores, pool them together, and compute the global c-index
as well as the c-index for each cancer type.
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


# Set seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

##########################################
# Constants & Paths
##########################################
# Modify these paths as needed.
INPUT_DIR = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold"
MODEL_DIR = "/home/chb3333/yulab/chb3333/gem-patho/models/image_data/v1_jina"
CANCER_TYPE_MAPPING_CSV = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/tcga_study_abbreviations.csv"

##########################################
# Cancer Type Mapping
##########################################
def int_to_binary_vector(x, width=6):
    return [int(b) for b in format(x, f"0{width}b")]

df_ct = pd.read_csv(CANCER_TYPE_MAPPING_CSV)
unique_types = sorted(df_ct["Study Abbreviation"].unique())
cancer_type_mapping = {ct: int_to_binary_vector(i, 6) for i, ct in enumerate(unique_types)}
print("Cancer type mapping:", cancer_type_mapping)


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
# Dataset for Preprocessed Sequences (with cancer type acronym return)
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
# Collate Function for Padding (returning cancer type acronyms)
##########################################
def collate_fn_preprocessed(batch):
    emb_list, score_list, cna_list, cancer_type_list, desc_list, times, events, cancer_acronym_list = zip(*batch)
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
    return padded_emb, padded_scores, padded_cnas, cancer_types, descriptions, times, events, mask, list(cancer_acronym_list)

##########################################
# Transformer Survival Model (same as training)
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
# Evaluation functions: Get predictions from one fold
##########################################
def evaluate_single_fold(model, dataloader, device):
    """Get predictions for a single fold without computing metrics."""
    model.eval()
    fold_T, fold_E, fold_risk, fold_ct = [], [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            emb, scores, cnas, cancer_type, desc, T, E, mask, ct_list = batch
            risk = model(
                emb.to(device),
                scores.to(device),
                cnas.to(device),
                cancer_type.to(device),
                desc.to(device),
                src_key_padding_mask=mask.to(device)
            )
            fold_T.append(T.cpu().numpy())
            fold_E.append(E.cpu().numpy())
            fold_risk.append(risk.cpu().numpy())
            fold_ct.extend(ct_list)
    
    return (
        np.concatenate(fold_T).squeeze(),
        np.concatenate(fold_E).squeeze(),
        np.concatenate(fold_risk).squeeze(),
        fold_ct
    )

##########################################
# Main evaluation: Loop over folds, normalize predictions, pool data, and compute c-index
##########################################
def main():
    # Determine number of folds from directory structure (folders starting with "fold_")
    fold_dirs = sorted([d for d in os.listdir(INPUT_DIR) if d.startswith("fold_")])
    NUM_FOLDS = len(fold_dirs)
    print(f"Found {NUM_FOLDS} folds")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Lists to collect predictions and labels from all folds
    all_times = []
    all_events = []
    all_scores = []  # Will store normalized risk scores
    all_cancer_types = []
    
    for fold_dir in fold_dirs:
        fold_num = fold_dir.split("_")[-1]
        print(f"\nProcessing {fold_dir}...")
        
        # Load fold-specific data
        test_path = os.path.join(INPUT_DIR, fold_dir, "test.parquet")
        test_df = pd.read_parquet(test_path)
        test_dataset = PreprocessedSequenceDataset(
            test_df, 
            token_col="gene_embed_seq",
            cancer_type_mapping=cancer_type_mapping
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=32, 
            collate_fn=collate_fn_preprocessed
        )
        
        # Load model for this fold
        model_path = os.path.join(MODEL_DIR, fold_dir, f"best_model_fold_{fold_num}.pth")
        sample_batch = next(iter(test_loader))
        d_gene = sample_batch[0].shape[-1]
        model = PreprocessedTransformerSurvivalModel(
            d_gene=d_gene,
            d_model=256,
            polyphen_hidden_dim=128,
            nhead=4,
            dropout=0.1,
            desc_dim=sample_batch[4].shape[-1]
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Evaluate this fold
        times_fold, events_fold, risk_fold, ct_fold = evaluate_single_fold(model, test_loader, device)
        
        # Normalize the risk predictions for this fold using z-score normalization
        mu = np.mean(risk_fold) #take from train instead of test
        sigma = np.std(risk_fold) #take from train instead of test
        if sigma > 0:
            risk_fold_norm = (risk_fold - mu) / sigma
        else:
            risk_fold_norm = risk_fold  # if sigma == 0, leave unchanged
        
        # Append data from this fold
        all_times.append(times_fold)
        all_events.append(events_fold)
        all_scores.append(risk_fold_norm)
        all_cancer_types.extend(ct_fold)
    
    # Pool data from all folds
    global_times = np.concatenate(all_times)
    global_events = np.concatenate(all_events)
    global_scores = np.concatenate(all_scores)
    global_cancer_types = np.array(all_cancer_types)
    
    # Compute global c-index using pooled (normalized) risk scores
    global_cindex = concordance_index(global_times, -global_scores, global_events)
    print(f"\nGlobal pooled c-index: {global_cindex:.4f}")
    
    # Compute per-cancer-type c-index
    unique_cancer_types = np.unique(global_cancer_types)
    cancer_results = {}
    for ct in unique_cancer_types:
        mask = global_cancer_types == ct
        if np.sum(mask) < 2:
            cancer_results[ct] = np.nan
        else:
            cancer_results[ct] = concordance_index(global_times[mask], -global_scores[mask], global_events[mask])
    print("\nC-index by Cancer Type (pooled across folds):")
    for ct, cidx in cancer_results.items():
        print(f"  {ct}: {cidx:.4f}")
    
if __name__ == "__main__":
    main()
