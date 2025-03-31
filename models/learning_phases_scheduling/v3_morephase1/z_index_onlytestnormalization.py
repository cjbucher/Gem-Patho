#!/usr/bin/env python
"""
Script to evaluate k-fold cross-validation predictions from the transformer survival model,
pool the folds together across each experiment (using only test data), normalize the test risk scores globally,
and compute the overall c-index as well as the c-index for each cancer type.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from lifelines.utils import concordance_index
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
INPUT_DIR = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold"
MODEL_DIR = "/home/chb3333/yulab/chb3333/gem-patho/models/learning_phases_scheduling/v3_morephase1"
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
# Dataset for Preprocessed Sequences
##########################################
class PreprocessedSequenceDataset(Dataset):
    def __init__(self, df, token_col="gene_embed_seq", cancer_type_mapping=None):
        self.df = df.reset_index(drop=True)
        self.token_col = token_col
        self.cancer_type_mapping = cancer_type_mapping if cancer_type_mapping is not None else {}
        self.has_description = "description_embeddings" in self.df.columns

        # Infer gene embedding dimension from first nonempty token list.
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
        print(f"Dataset: {total_samples} samples, {self.default_token_count} with default tokens ({default_percentage:.2f}%).")

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
            print("No description embedding found; using zero vector.")
            description = torch.zeros(self.genename_dim, dtype=torch.float)

        time = torch.tensor(row["OS.time"], dtype=torch.float)
        event = torch.tensor(row["OS"], dtype=torch.float)
        return embeddings, scores, cnas, cancer_type_tensor, description, time, event, cancer_type_acronym

##########################################
# Collate Function for Padding
##########################################
def collate_fn_preprocessed(batch):
    emb_list, score_list, cna_list, cancer_type_list, desc_list, times, events, cancer_acronym_list = zip(*batch)
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
    return padded_emb, padded_scores, padded_cnas, cancer_types, descriptions, times, events, mask, cancer_acronym_list

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
# Evaluation Function for a Fold
##########################################
def evaluate_fold(exp_dir, fold, device):
    fold_name = f"fold_{fold}"
    fold_folder = os.path.join(INPUT_DIR, fold_name)
    train_path = os.path.join(fold_folder, "train.parquet")
    val_path   = os.path.join(fold_folder, "val.parquet")
    test_path  = os.path.join(fold_folder, "test.parquet")

    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path)
    test_df  = pd.read_parquet(test_path)

    cols = ["gene_embed_seq", "OS.time", "OS", "type", "description_embeddings"]
    train_df = train_df[cols]
    val_df   = val_df[cols]
    test_df  = test_df[cols]

    train_dataset = PreprocessedSequenceDataset(train_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)
    val_dataset   = PreprocessedSequenceDataset(val_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)
    test_dataset  = PreprocessedSequenceDataset(test_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_preprocessed)

    sample_batch = next(iter(train_loader))
    d_gene = sample_batch[0].shape[-1]
    desc_dim = sample_batch[4].shape[-1]

    model = PreprocessedTransformerSurvivalModel(d_gene=d_gene, d_model=256,
                                                   polyphen_hidden_dim=128, nhead=4,
                                                   dropout=0.1, desc_dim=desc_dim).to(device)
    model_file = os.path.join(exp_dir, fold_name, f"best_model_fold_{fold}.pth")
    if not os.path.exists(model_file):
        print(f"Model file {model_file} not found. Skipping fold {fold}.")
        return None
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    def get_predictions(loader):
        all_T, all_E, all_risk, all_ct = [], [], [], []

        with torch.no_grad():
            for batch in loader:
                emb, scores, cnas, cancer_type, desc, T, E, mask, ct_list = batch
                risk = model(emb.to(device),
                             scores.to(device),
                             cnas.to(device),
                             cancer_type.to(device),
                             desc.to(device),
                             src_key_padding_mask=mask.to(device))
                all_T.append(T.cpu().numpy())
                all_E.append(E.cpu().numpy())
                all_risk.append(risk.cpu().numpy())
                all_ct.extend(ct_list)
        return np.concatenate(all_T).squeeze(), np.concatenate(all_E).squeeze(), np.concatenate(all_risk).squeeze(), np.array(all_ct)

    train_T, train_E, train_risk, _ = get_predictions(train_loader)
    val_T, val_E, val_risk, _ = get_predictions(val_loader)
    test_T, test_E, test_risk, test_ct = get_predictions(test_loader)

    return {
        "fold": fold,
        "train_T": train_T,
        "train_E": train_E,
        "train_risk": train_risk,
        "val_T": val_T,
        "val_E": val_E,
        "val_risk": val_risk,
        "test_T": test_T,
        "test_E": test_E,
        "test_risk": test_risk,
        "test_ct": test_ct
    }

##########################################
# Main Evaluation Loop
##########################################
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    exp_folders = [os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR) if d.startswith("exp_")]
    if not exp_folders:
        print("No experiment folders found.")
        return

    # Collect fold-level predictions organized by experiment.
    experiments = {}
    for exp_dir in exp_folders:
        exp_name = os.path.basename(exp_dir)
        print(f"\nProcessing experiment: {exp_name}")
        fold_dirs = [d for d in os.listdir(exp_dir) if d.startswith("fold_")]
        fold_results = []
        for fd in fold_dirs:
            try:
                fold_num = int(fd.split("_")[-1])
            except:
                continue
            res = evaluate_fold(exp_dir, fold_num, device)
            if res is not None:
                fold_results.append(res)
        if not fold_results:
            print(f"No valid fold evaluations found for experiment {exp_name}.")
            continue

        # Pool test predictions across all folds for the experiment.
        exp_test_risk = np.concatenate([res["test_risk"] for res in fold_results])
        exp_test_T = np.concatenate([res["test_T"] for res in fold_results])
        exp_test_E = np.concatenate([res["test_E"] for res in fold_results])
        exp_test_ct = np.concatenate([res["test_ct"] for res in fold_results])

        # Compute global normalization parameters using only the pooled test risk predictions.
        overall_mu = np.mean(exp_test_risk)
        overall_sigma = np.std(exp_test_risk)
        if overall_sigma > 0:
            norm_test_risk = (exp_test_risk - overall_mu) / overall_sigma
        else:
            norm_test_risk = exp_test_risk

        # Compute global c-index on the pooled test predictions.
        global_cindex = concordance_index(exp_test_T, -norm_test_risk, exp_test_E)

        # Compute per-cancer-type c-index.
        cancer_results = {}
        unique_ct = np.unique(exp_test_ct)
        for ct in unique_ct:
            mask = exp_test_ct == ct
            if np.sum(mask) < 2:
                cancer_results[ct] = np.nan
            else:
                try:
                    cancer_results[ct] = concordance_index(exp_test_T[mask], -norm_test_risk[mask], exp_test_E[mask])
                except ZeroDivisionError:
                    print(f"Warning: No admissible pairs for cancer type {ct}. Setting c-index to NaN.")
                    cancer_results[ct] = np.nan

        # Store experiment-level results.
        experiments[exp_name] = {
            "experiment": exp_name,
            "global_cindex": global_cindex,
            "per_cancer_cindex": cancer_results
        }
        print(f"\nResults for experiment {exp_name}:")
        print(f"  Global c-index: {global_cindex:.4f}")
        print("  Per-cancer-type c-index:")
        for ct, cidx in cancer_results.items():
            print(f"    {ct}: {cidx:.4f}" if not np.isnan(cidx) else f"    {ct}: NaN")

    if not experiments:
        print("No experiment results to summarize.")
        return

    # Save summary CSV: one row per experiment.
    rows = []
    for exp_name, res in experiments.items():
        base = {
            "experiment": res["experiment"],
            "global_cindex": res["global_cindex"]
        }
        for ct in sorted(res["per_cancer_cindex"].keys()):
            base[f"cindex_ct_{ct}"] = res["per_cancer_cindex"].get(ct, np.nan)
        rows.append(base)
    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(MODEL_DIR, "experiments_test_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary CSV to {summary_csv}")

    # Identify best experiment per cancer type.
    best_by_ct = {}
    for ct in sorted(list({ct for res in experiments.values() for ct in res["per_cancer_cindex"].keys()})):
        best_val = -np.inf
        best_exp = None
        for res in experiments.values():
            cidx = res["per_cancer_cindex"].get(ct, np.nan)
            if np.isnan(cidx):
                continue
            if cidx > best_val:
                best_val = cidx
                best_exp = res["experiment"]
        best_by_ct[f"cancer_type_{ct}"] = {"experiment": best_exp, "cindex": best_val}

    print("\nBest experiment per cancer type:")
    for ct, info in best_by_ct.items():
        print(f"  {ct}: Experiment {info['experiment']} with c-index {info['cindex']:.4f}")

if __name__ == "__main__":
    main()
