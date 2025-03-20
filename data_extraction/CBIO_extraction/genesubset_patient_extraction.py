import os
import pandas as pd
from tqdm import tqdm

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

# --- Configuration ---
excel_path = '/home/chb3333/yulab/chb3333/gem-patho/data_extraction/TCGA-CDR-SupplementalTableS1.csv'
download_base_dir = '/home/chb3333/yulab/chb3333/gem-patho/data_extraction/CBIO_extraction/download'
final_output_dir = '/home/chb3333/yulab/chb3333/gem-patho/data_extraction/CNAs'
os.makedirs(final_output_dir, exist_ok=True)
final_parquet_path = os.path.join(final_output_dir, "cna_data.parquet")
report_path = os.path.join(final_output_dir, "report.txt")
gene_list_path = '/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancer_gene_list_selection/combined_genelist.csv'

# --- Load Gene List from File ---
df_genelist = pd.read_csv(gene_list_path)
gene_list = df_genelist["Gene Symbol"].tolist()

print(f"Gene list loaded with {len(gene_list)} genes.")

# --- Logging Setup ---
report_lines = []
def log(msg):
    print(msg)
    report_lines.append(msg)

# --- Read the Excel File ---
try:
    df_excel = pd.read_csv(excel_path)
    log(f"Excel file read successfully: {excel_path}")
except Exception as e:
    log(f"Error reading Excel file: {e}")
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    exit(1)

# Excel must have "bcr_patient_barcode" and "type"
if "bcr_patient_barcode" not in df_excel.columns or "type" not in df_excel.columns:
    log("Excel file must contain both 'bcr_patient_barcode' and 'type' columns.")
    with open(report_path, 'w') as f:
        f.write("\n".join(report_lines))
    exit(1)

# --- Build Mapping: cancer_type Folder -> Set of Normalized Excel Patient IDs ---
candidate_to_patient_set = {}
for _, row in df_excel.iterrows():
    typ = str(row["type"]).lower().strip()
    pid = str(row["bcr_patient_barcode"]).strip()
    norm_pid = normalize_patient_id(pid)
    # For cancer types "coad" or "read", use the combined cancer_type folder.
    if typ in ["coad", "read"]:
        cancer_type = "coadread_tcga_pan_can_atlas_2018"
    else:
        cancer_type = f"{typ}_tcga_pan_can_atlas_2018"
    candidate_to_patient_set.setdefault(cancer_type, set()).add(norm_pid)

# --- Initialize Summary Containers ---
candidate_summary = {}   # Summary info per cancer_type.
all_patient_records = [] # List to collect matched patient records.

# --- Process Each cancer_type ---
for cancer_type, excel_patient_set in candidate_to_patient_set.items():
    try:
        excel_count = len(excel_patient_set)
        candidate_summary[cancer_type] = {
            "excel_count": excel_count,
            "matched_count": 0,
            "missing_patients": 0,
            "missing_gene_entries": 0,
            "missing_details": [],
            "unmatched_patient_ids": []
        }
        log(f"Processing cancer_type: {cancer_type} (Excel patient count: {excel_count})")
        candidate_dir = os.path.join(download_base_dir, cancer_type)
        target_file_path = os.path.join(candidate_dir, "data_cna.txt")
        log(f"Looking for CNA file: {target_file_path}")
        if not os.path.exists(target_file_path):
            log(f"ERROR: CNA file not found for cancer_type {cancer_type}")
            continue

        try:
            df_cna = pd.read_csv(target_file_path, sep="\t", engine="python")
            log(f"CNA file read for cancer_type {cancer_type} (shape: {df_cna.shape})")
        except Exception as e:
            log(f"ERROR reading CNA file for cancer_type {cancer_type}: {e}")
            continue

        if "Hugo_Symbol" not in df_cna.columns:
            log(f"ERROR: 'Hugo_Symbol' column not found in cancer_type {cancer_type}")
            continue

        # Warn if any gene is missing.
        available_genes = df_cna["Hugo_Symbol"].tolist()
        for gene in gene_list:
            if gene not in available_genes:
                log(f"WARNING: Gene {gene} not found in cancer_type {cancer_type}")

        # Filter for rows corresponding to genes in the gene list.
        df_filtered = df_cna[df_cna["Hugo_Symbol"].isin(gene_list)]
        if df_filtered.empty:
            log(f"No gene rows found for cancer_type {cancer_type}")
            continue

        df_filtered = df_filtered.set_index("Hugo_Symbol")
        # Drop non-patient columns.
        for col in ["Entrez_Gene_Id", "Cytoband"]:
            if col in df_filtered.columns:
                df_filtered = df_filtered.drop(columns=[col])

        # Transpose so that each row becomes a patient.
        df_patient = df_filtered.transpose().reset_index()
        df_patient.rename(columns={"index": "patient_id"}, inplace=True)
        # Create a normalized patient ID column.
        df_patient["normalized_patient_id"] = df_patient["patient_id"].apply(normalize_patient_id)
        df_patient["cancer_type"] = cancer_type
        log(f"cancer_type {cancer_type} - Total patients in CNA file: {df_patient.shape[0]}")

        # Match only patients from the Excel file.
        matched_df = df_patient[df_patient["normalized_patient_id"].isin(excel_patient_set)]
        matched_count = matched_df.shape[0]
        candidate_summary[cancer_type]["matched_count"] = matched_count
        candidate_summary[cancer_type]["missing_patients"] = max(0, excel_count - matched_count)
        candidate_summary[cancer_type]["unmatched_patient_ids"] = list(excel_patient_set - set(matched_df["normalized_patient_id"]))
        log(f"cancer_type {cancer_type} - Matched patients: {matched_count}, Missing patients: {excel_count - matched_count}")

        # Process each matched patient.
        for _, row in tqdm(matched_df.iterrows(), total=matched_count, desc=f"cancer_type {cancer_type}"):
            record = row.to_dict()
            missing_for_patient = []
            for gene in gene_list:
                if gene not in record or pd.isna(record.get(gene)):
                    candidate_summary[cancer_type]["missing_gene_entries"] += 1
                    missing_for_patient.append(gene)
            if missing_for_patient:
                candidate_summary[cancer_type]["missing_details"].append(
                    f"Patient {record['normalized_patient_id']}: missing {', '.join(missing_for_patient)}"
                )
            all_patient_records.append(record)
    except Exception as e:
        log(f"Unexpected error processing cancer_type {cancer_type}: {e}")

# --- Combine All Matched Patient Records ---
final_df = pd.DataFrame(all_patient_records)
for gene in gene_list:
    if gene in final_df.columns:
        final_df[gene] = pd.to_numeric(final_df[gene], errors='coerce')

try:
    final_df.to_parquet(final_parquet_path, index=False)
    log(f"Parquet file saved at: {final_parquet_path}")
except Exception as e:
    log(f"ERROR saving Parquet file: {e}")

# --- Build Comprehensive Report ---
report_out = []
report_out.append("CNA Extraction Summary Report")
report_out.append("="*50)
report_out.append("")
report_out.append("Summary by cancer_type:")
for cancer_type, summary in candidate_summary.items():
    report_out.append(f"cancer_type: {cancer_type}")
    report_out.append(f"    Excel Patient Count:        {summary['excel_count']}")
    report_out.append(f"    Matched CNA Patient Count:  {summary['matched_count']}")
    report_out.append(f"    Missing Patients:           {summary['missing_patients']}")
    report_out.append(f"    Total Missing Gene Entries: {summary['missing_gene_entries']}")
    if summary["unmatched_patient_ids"]:
        report_out.append(f"    Unmatched Patient IDs:      {', '.join(summary['unmatched_patient_ids'])}")
    report_out.append("")
report_out.append("="*50)
report_out.append("Detailed Missing Data Per cancer_type:")
for cancer_type, summary in candidate_summary.items():
    if summary["missing_details"]:
        report_out.append(f"cancer_type: {cancer_type}")
        for detail in summary["missing_details"]:
            report_out.append(f"    {detail}")
        report_out.append("")
report_out.append("="*50)
report_out.append("Additional Processing Messages:")
report_out.extend(report_lines)

with open(report_path, 'w') as f:
    f.write("\n".join(report_out))

log("Processing complete. Please check the output Parquet file and report.txt for details.")
