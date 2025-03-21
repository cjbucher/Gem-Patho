import os
import pandas as pd
import numpy as np

# --- Paths ---
KFOLD_DIR = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/kfolds/master_kfold"
OUTPUT_BASE_DIR = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/kfolds/jinaai_kfold"
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

# --- Load Metadata ---
# Description embeddings metadata
desc_embs_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/description_embeddings/JinaAI/jinaai_description_embeddings.parquet"
df_desc = pd.read_parquet(desc_embs_path, engine="pyarrow")

# Master mutation data (only needed columns)
master_df_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/patient_mutation_polyphen_vector_OStime/master_df.parquet"
df_master = pd.read_parquet(master_df_path, engine="pyarrow")
master_cols = ["polyphen_score", "mutated_genes", "OS.time", "OS", "type", "Case ID"]
df_master_subset = df_master[master_cols].copy()

# Gene list (extracting the 'Gene Symbol' column)
gene_list_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancer_gene_list_selection/combined_genelist.csv"
df_genelist = pd.read_csv(gene_list_path)
gene_list = df_genelist["Gene Symbol"].tolist()

# Gene embeddings (as a dict: {gene: embedding})
gene_embs_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/genename_embeddings/text_embedding_models/jinaai/embeddings.parquet"
df_gene_embs = pd.read_parquet(gene_embs_path, engine="pyarrow")
gene_embeddings_dict = {gene: row.values for gene, row in df_gene_embs.iterrows()}

# --- Load CNA Data ---
cna_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/CNAs/cna_data.parquet"
df_cna = pd.read_parquet(cna_path, engine="pyarrow")


# --- Gene Name Resolution ---
gene_synonyms = {
    "C15orf65": "CCDC186",
    "SEPT5": "SEPTIN5",
    "SEPT6": "SEPTIN6",
    "SEPT9": "SEPTIN9",
    "CARS": "CARS1"
}
genes_to_skip = {
    "CTNNB1", "FGFR1OP", "H3F3A", "H3F3B", "HIST1H3B", "HIST1H4I",
    "HMGN2P46", "IGH", "IGK", "IGL", "TRA", "TRB", "TRD", "TRG",
    "H3P6", "NBEAP1", "NHERF1", "TERC", "TMSB4XP8"
}

def resolve_gene_name(gene):
    if gene in genes_to_skip:
        return None
    return gene_synonyms.get(gene, gene)

# --- Helper Functions for CNA ---
def normalize_patient_id(pid):
    """
    Normalize a TCGA patient barcode by taking the first three parts.
    E.g. "TCGA-OR-A5J8-01" becomes "TCGA-OR-A5J8".
    """
    parts = pid.split('-')
    if len(parts) >= 4:
        return "-".join(parts[:3])
    else:
        return pid

def get_patient_cna(case_id, cna_df):
    """
    Retrieve the CNA record for a given patient using the normalized 'Case ID'.
    """
    norm_pid = normalize_patient_id(case_id)
    patient_cna = cna_df[cna_df["normalized_patient_id"] == norm_pid]
    return patient_cna

# --- Modified Mapping Function to Include CNA Info ---
def map_polyphen_to_gene_embeds(polyphen_vector, gene_list, gene_embeddings, case_id, cna_df):
    """
    For every nonzero score in polyphen_vector (aligned with gene_list),
    return a list of dicts with resolved gene name, its embedding, the score, and the CNA value.
    """
    seq = []
    if len(polyphen_vector) != len(gene_list):
        print(f"Length mismatch: {len(polyphen_vector)} vs {len(gene_list)}")
        return seq
    
    # Retrieve the patient's CNA data (if available)
    patient_cna = get_patient_cna(case_id, cna_df)
    
    for gene, score in zip(gene_list, polyphen_vector):
        if score == 0:
            continue
        resolved = resolve_gene_name(gene)
        if resolved is None:
            continue
        if resolved in gene_embeddings:
            emb = gene_embeddings[resolved]
            if isinstance(emb, np.ndarray):
                emb = emb.tolist()
            # For CNA lookup, switch 'CTNNB1' to 'CTNNBIP1'
            gene_mapped = "CTNNBIP1" if resolved == "CTNNB1" else resolved
            # Get CNA number: default to 2 if not found or if CNA data is missing
            if patient_cna.empty:
                cna_number = 2
            else:
                if gene_mapped in patient_cna.columns:
                    cna_number = patient_cna.iloc[0][gene_mapped]
                    if pd.isna(cna_number):
                        cna_number = 2
                else:
                    cna_number = 2
            seq.append({"gene": resolved, "embedding": emb, "score": score, "cna": cna_number})
        else:
            print(f"Warning: Embedding for '{resolved}' not found.")
    return seq


# --- Process Description Metadata ---
# Keep only needed columns, split the comma-separated "Case IDs", and explode
df_desc_subset = df_desc[["Case IDs", "CANCER_TYPE_ACRONYM", "description_embeddings"]].copy()
df_desc_subset["Case IDs"] = df_desc_subset["Case IDs"].apply(lambda x: [case.strip() for case in x.split(",")])
df_desc_exploded = df_desc_subset.explode("Case IDs").rename(columns={"Case IDs": "Case ID"})

# --- Loop over folds and splits ---
splits = ["train", "val", "test"]
num_folds = 10

for foldz in range(num_folds):
    fold = foldz + 1
    # Create an output folder for the fold; note: fold folders are 1-indexed.
    fold_out_dir = os.path.join(OUTPUT_BASE_DIR, f"fold_{fold}")
    os.makedirs(fold_out_dir, exist_ok=True)
    
    for split in splits:
        fold_path = os.path.join(KFOLD_DIR, f"fold_{fold}", f"{split}.parquet")
        if not os.path.exists(fold_path):
            print(f"File not found: {fold_path}. Skipping.")
            continue

        # Load kfold file
        df_fold = pd.read_parquet(fold_path, engine="pyarrow")
        print(f"Fold {fold} {split} set loaded: {df_fold.shape}")

        # Merge with description embeddings and master mutation data using "Case ID"
        df_merged = pd.merge(df_fold, df_desc_exploded, how="left", on="Case ID")
        df_merged = pd.merge(df_merged, df_master_subset, how="left", on="Case ID")

        # Map polyphen scores to gene embedding sequences if column exists
        if "polyphen_score" in df_merged.columns:
            df_merged["gene_embed_seq"] = df_merged.apply(
                lambda row: map_polyphen_to_gene_embeds(
                    row["polyphen_score"], gene_list, gene_embeddings_dict, row["Case ID"], df_cna
                ),
                axis=1
            )
        else:
            print(f"Warning: 'polyphen_score' not in fold {fold} {split}.")

        # Clean: drop rows missing OS.time, OS, or type; drop CANCER_TYPE_ACRONYM if present
        df_clean = df_merged.dropna(subset=["OS.time", "OS", "type"])


        # Save processed file: file name is simply split.parquet in the corresponding fold folder.
        output_file = os.path.join(fold_out_dir, f"{split}.parquet")
        df_clean.to_parquet(output_file, engine="pyarrow", index=False)
        print(f"Saved processed fold {fold} {split} to {output_file}")
