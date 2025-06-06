#!/usr/bin/env python
"""
Script to compute test risk scores from scratch using the best model for each fold,
normalize the risk scores within each fold, and then compute global and per–cancer-type c-indices.
If a fold produces an error (e.g. no admissible pairs) or if the test risk file already exists,
that fold is skipped.
"""

import os
import copy
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Sampler
from lifelines.utils import concordance_index
from scipy.stats import pearsonr
from tqdm import tqdm

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
BASE_OUTPUT_DIR = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/models/actual_final_results/genetics_and_image/all_batch"
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
# Samplers and Collate Function
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

def collate_fn_preprocessed(batch):
    # Unpack list of tuples (including image features).
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
# Unified Loss Function
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
# Dataset for Preprocessed Sequences with Image Modality
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
        project_id = row.get("Project ID", None)
        if project_id is not None:
            fs_folder = f"/n/data2/hms/dbmi/kyu/lab/NCKU/foundation_model_features/WSI_features/{project_id}-FS/GIGAPATH/20X/pt_files(stain_norm)"
            pattern = os.path.join(fs_folder, f"{case_id}*.pt")
            fs_files = glob.glob(pattern)
            if fs_files:
                img_features = torch.load(fs_files[0])  # Expected shape: (N, D_feat)
            else:
                img_features = torch.zeros((1, 1536), dtype=torch.float)
        else:
            img_features = torch.zeros((1, 1536), dtype=torch.float)
        # ---------------------------------
        return embeddings, scores, cnas, cancer_type_tensor, description, time, event, case_id, img_features

##########################################
# Transformer Survival Model
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
        self.image_extractor = ImageFeatureExtractor(D_feat=1536, D_inner=512, D_out=d_model, droprate=0.1)
        # Fusion layer: fuse pooled text representation and image representation.
        self.fusion_linear = nn.Linear(2*d_model, 1)
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
        fusion = torch.cat([pooled, img_feats], dim=1)
        risk = self.fusion_linear(fusion)
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
            # Move image features to device.
            img_list = [img.to(device) for img in img_list]
            
            risk = model(emb_batch, score_batch, cna_batch, cancer_type_batch, desc_batch, img_list)
            
            # Convert cancer type binary vector to an index.
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
    #all_E = np.concatenate(all_E).squeeze()
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
                try:
                    c_idx = concordance_index(all_T[mask_ct], -all_risk[mask_ct], all_E[mask_ct])
                except ZeroDivisionError:
                    c_idx = np.nan
                    missing_cancer_types.append(int(ct))
        cindex_by_type[int(ct)] = c_idx
        if not np.isnan(c_idx):
            normalized_values.append(c_idx)
    normalized_avg_cindex = np.mean(normalized_values) if normalized_values else np.nan
    
    return avg_loss, global_cindex, cindex_by_type, normalized_avg_cindex, all_T, all_risk, all_ct, all_case_ids, all_E, missing_cancer_types

##########################################
# Main Function: Compute Test Risk Scores and c-indices
##########################################
def main():
    parser = argparse.ArgumentParser(description="Compute test risk scores from the best model and calculate c-indices.")
    parser.add_argument("--exp_folder", type=str, default="cancertypebatching_exp_0_phase2_10",
                        help="Experiment folder name (e.g., cancertypebatching_exp_0_phase2_10)")
    args = parser.parse_args()
    
    EXPERIMENT = args.exp_folder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We will collect test risk scores from each fold.
    all_test_dfs = []
    batch_size = 64

    for fold in range(1, NUM_FOLDS+1):
        print(f"\nProcessing Fold {fold} ...")
        # Check if test risk scores already exist for this fold.
        fold_save_dir = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT, f"fold_{fold}")
        test_risk_file = os.path.join(fold_save_dir, "test_risk_scores.csv")
        if os.path.exists(test_risk_file):
            print(f"Test risk scores already computed for fold {fold} at {test_risk_file}. Not Skipping.")
    


        
        fold_folder = os.path.join(INPUT_DIR, f"fold_{fold}")
        test_path = os.path.join(fold_folder, "test.parquet")
        if not os.path.exists(test_path):
            print(f"Test file not found: {test_path}. Skipping fold {fold}.")
            continue
        
        cols = ["Case ID", "gene_embed_seq", "OS.time", "OS", "type", "description_embeddings", "Project ID"]
        test_df = pd.read_parquet(test_path, engine="pyarrow")[cols]
        
        # Create test dataset and dataloader.
        test_dataset = PreprocessedSequenceDataset(test_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed)
        
        # Load best model for this fold.
        model_path = os.path.join(BASE_OUTPUT_DIR, EXPERIMENT, f"fold_{fold}", f"best_model_fold_{fold}.pth")
        if not os.path.exists(model_path):
            print(f"Best model not found: {model_path}. Skipping fold {fold}.")
            continue

        # Determine embedding dimension (d_gene) from a sample batch.
        sample_batch = next(iter(test_loader))
        sample_emb = sample_batch[0]
        sample_desc = sample_batch[4]
        d_gene = sample_emb.shape[-1]
        
        model = PreprocessedTransformerSurvivalModel(d_gene=d_gene, d_model=256,
                                                       polyphen_hidden_dim=128, nhead=4, dropout=0.1,
                                                       desc_dim=sample_desc.shape[-1])
        model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        try:
            # Evaluate model on the test set to obtain risk scores.
            test_loss, test_cindex, test_cindex_by_type, normalized_avg_cindex, all_T, all_risk, all_ct, all_case_ids, all_E, missing_ct = evaluate_model(model, test_loader, device)

        except Exception as e:
            print(f"Error during evaluation for fold {fold}: {e}. Skipping fold {fold}.")
            continue
        
        # Normalize risk scores within this fold (using z-score normalization).
        all_risk = np.array(all_risk)
        mean_risk = np.mean(all_risk)
        std_risk = np.std(all_risk)
        if std_risk > 0:
            normalized_risk = (all_risk - mean_risk) / std_risk
        else:
            normalized_risk = all_risk
        
        # Save the test risk scores for this fold.
        df_test_risk = pd.DataFrame({
            "Case ID": all_case_ids,
            "OS.time": all_T,
            "OS": all_E,  # Include the event indicator
            "risk": all_risk,
            "normalized_risk": normalized_risk,
            "cancer_type": all_ct,
            "fold": fold
        })

        os.makedirs(fold_save_dir, exist_ok=True)
        save_path = os.path.join(fold_save_dir, "test_risk_scores.csv")
        df_test_risk.to_csv(save_path, index=False)
        print(f"Saved test risk scores for fold {fold} to {save_path}")
        
        all_test_dfs.append(df_test_risk)
    
    # Concatenate all fold data and compute global c-index.
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
        cancer_types = all_test_df["cancer_type"].unique()
        ct_cindices = {}
        for ct in cancer_types:
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
        
        # Save the global test risk scores and c-index summary.
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
