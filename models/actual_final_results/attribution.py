# %%
# Attribution script for multimodal transformer model (genetics + image)

import os
import torch
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader
import torch.nn as nn
import glob

# %%
# Load fold data
fold_folder = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold_20val/fold_1"
train_path = os.path.join(fold_folder, "train.parquet")
cols = ["Case ID", "gene_embed_seq", "OS.time", "OS", "type", "description_embeddings", "Project ID"]
train_df = pd.read_parquet(train_path, engine="pyarrow")[cols]

# Load gene list
gene_list_path = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/data_extraction/cancer_gene_list_selection/combined_genelist.csv"
gene_list = pd.read_csv(gene_list_path)["Gene Symbol"].tolist()

# Cancer type mapping
cancer_csv = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/data_extraction/cancertype_location_description/tcga_study_abbreviations.csv"
df_ct = pd.read_csv(cancer_csv)
unique_types = sorted(df_ct["Study Abbreviation"].unique())
def int_to_binary_vector(x, width=6):
    return [int(b) for b in format(x, f"0{width}b")]
cancer_type_mapping = {ct: int_to_binary_vector(i, 6) for i, ct in enumerate(unique_types)}
type_to_index = {ct: i for i, ct in enumerate(unique_types)}

# %%
# Dataset
class PreprocessedSequenceDataset(nn.Module):
    def __init__(self, df, token_col="gene_embed_seq", cancer_type_mapping=None):
        self.df = df.reset_index(drop=True)
        self.token_col = token_col
        self.cancer_type_mapping = cancer_type_mapping
        self.has_description = "description_embeddings" in self.df.columns

        self.genename_dim = None
        for idx in range(len(self.df)):
            tokens = self.df.iloc[idx][token_col]
            if isinstance(tokens, np.ndarray):
                tokens = tokens.tolist()
            if tokens:
                self.genename_dim = len(tokens[0]["embedding"])
                break

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tokens = row[self.token_col]
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()
        if not tokens:
            tokens = [{"gene": "", "embedding": [0.0]*self.genename_dim, "score": 0.0, "cna": 0.0}]
        gene_names = [token.get("gene", "") or gene_list[i] for i, token in enumerate(tokens)]
        embeddings = [torch.tensor(token["embedding"], dtype=torch.float) for token in tokens]
        scores = [torch.tensor(token["score"], dtype=torch.float) for token in tokens]
        cnas = [torch.tensor(token.get("cna", 0.0), dtype=torch.float) for token in tokens]

        ct_acronym = row.get("type")
        ct_vector = cancer_type_mapping.get(ct_acronym, [0]*6)
        cancer_type_tensor = torch.tensor(ct_vector, dtype=torch.float)
        description = torch.tensor(row["description_embeddings"], dtype=torch.float) if self.has_description else torch.zeros(self.genename_dim)
        time = torch.tensor(row["OS.time"], dtype=torch.float)
        event = torch.tensor(row["OS"], dtype=torch.float)
        case_id = row["Case ID"]

        # Image features
        project_id = row.get("Project ID")
        fs_folder = f"/n/data2/hms/dbmi/kyu/lab/NCKU/foundation_model_features/WSI_features/{project_id}-FS/GIGAPATH/20X/pt_files(stain_norm)"
        pattern = os.path.join(fs_folder, f"{case_id}*.pt")
        fs_files = glob.glob(pattern)
        img_features = torch.load(fs_files[0]) if fs_files else torch.zeros((1, 1536))

        return embeddings, scores, cnas, cancer_type_tensor, description, time, event, case_id, img_features, gene_names

# %%
# Collate function
def collate_fn_with_image(batch):
    emb_list, score_list, cna_list, ct_list, desc_list, times, events, case_ids, img_list, gene_names_list = zip(*batch)
    padded_emb = nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in emb_list], batch_first=True)
    padded_scores = nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in score_list], batch_first=True)
    padded_cnas = nn.utils.rnn.pad_sequence([torch.stack(seq) for seq in cna_list], batch_first=True)
    return padded_emb, padded_scores, padded_cnas, torch.stack(ct_list), torch.stack(desc_list), torch.stack(times), torch.stack(events), list(case_ids), list(img_list), list(gene_names_list)

# %%
# Load model
model_path = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/models/actual_final_results/genetics_and_image/all_batch/cancertypebatching_exp_0_phase2_10/fold_1/best_model_fold_1.pth"
model = torch.load(model_path, map_location=torch.device("cpu"))
model.eval()

# Monkey-patch for attention
for layer in model.transformer_encoder.layers:
    orig_forward = layer.self_attn.forward
    def new_forward(query, key, value, orig_forward=orig_forward, **kwargs):
        kwargs.pop("need_weights", None)
        return orig_forward(query, key, value, need_weights=True, **kwargs)
    layer.self_attn.forward = new_forward

# %%
# Build dataloader
dataset = PreprocessedSequenceDataset(train_df, token_col="gene_embed_seq", cancer_type_mapping=cancer_type_mapping)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn_with_image)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# %%
# Attribution (gradient-based)
def get_gradient_attributions(model, dataloader):
    gradient_data = {}
    model.eval()
    for emb, scores, cnas, ct, desc, times, events, case_ids, img_list, gene_names_list in dataloader:
        emb, scores, cnas, ct, desc, times, events = emb.to(device), scores.to(device), cnas.to(device), ct.to(device), desc.to(device), times.to(device), events.to(device)
        img_list = [img.to(device) for img in img_list]
        emb.requires_grad_(True)
        risk = model(emb, scores, cnas, ct, desc, img_list)
        risk.sum().backward()
        grads = emb.grad
        attributions = (emb * grads).sum(dim=-1).detach().cpu().numpy()
        for i in range(len(case_ids)):
            ct_list = ct[i].cpu().tolist()
            sample_ct = type_to_index.get([k for k,v in cancer_type_mapping.items() if v == [int(x) for x in ct_list]][0], -1)
            gradient_data[case_ids[i]] = {
                "cancer_type": sample_ct,
                "gene_names": gene_names_list[i],
                "gradient_attributions": attributions[i][1:]
            }
    return gradient_data

# %%
# Aggregation
def aggregate_attributions(attr_dict, key="gradient_attributions"):
    aggregated = defaultdict(list)
    for sample in attr_dict.values():
        for gene, score in zip(sample["gene_names"], sample[key]):
            aggregated[gene].append((sample["cancer_type"], score))
    return aggregated

# %%
# Run attribution
gradient_results = get_gradient_attributions(model, dataloader)
aggregated_gradient = aggregate_attributions(gradient_results)

# %%
# Show top genes
gene_means = {g: np.mean([s for (_, s) in vals]) for g, vals in aggregated_gradient.items()}
sorted_genes = sorted(gene_means.items(), key=lambda x: x[1], reverse=True)
print("\n🔺 Top 20 genes increasing risk:")
for g, s in sorted_genes[:20]:
    print(f"{g}: {s:.4f}")

print("\n🔻 Top 20 genes decreasing risk:")
for g, s in sorted_genes[-20:]:
    print(f"{g}: {s:.4f}")

# %%
# Save to CSV
out_path = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/models/actual_final_results/fold_1_gradient_attributions.csv"
rows = []
for case_id, info in gradient_results.items():
    for gene, score in zip(info["gene_names"], info["gradient_attributions"]):
        rows.append({"Case ID": case_id, "Gene": gene, "Score": score, "Cancer Type": info["cancer_type"]})
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"Saved attributions to {out_path}")
