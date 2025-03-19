import os
import pandas as pd

# Define base directory where all tumor folders are stored
base_dir = "/home/chb3333/yulab/chb3333/data_extraction/CBIO_extraction/download"
# Define output directory where the final CSV, parquet and report will be saved
output_dir = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description"
# Create a subfolder "icd codes" inside the output directory
icd_codes_dir = os.path.join(output_dir, "icd_codes")
os.makedirs(icd_codes_dir, exist_ok=True)

# Define output file paths
output_csv = os.path.join(icd_codes_dir, "icd_data.csv")
output_parquet = os.path.join(icd_codes_dir, "icd_data.parquet")
report_file = os.path.join(icd_codes_dir, "icd_extraction.txt")

# List to hold individual dataframes and list for folders missing required columns
dfs = []
missing_folders = []

# Loop through each folder in the base directory
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    # Ensure we're looking at a directory
    if os.path.isdir(folder_path):
        file_path = os.path.join(folder_path, "data_clinical_patient.txt")
        if os.path.exists(file_path):
            try:
                # Read file using tab separator and skipping lines starting with '#' 
                df = pd.read_csv(file_path, sep="\t", comment="#")
                # Define the columns we want to keep in the order we want:
                # PATIENT_ID, CANCER_TYPE_ACRONYM, ICD_10, ICD_O_3_HISTOLOGY, ICD_O_3_SITE
                required_columns = ["PATIENT_ID", "CANCER_TYPE_ACRONYM", "ICD_10", "ICD_O_3_HISTOLOGY", "ICD_O_3_SITE"]
                # Check if all required columns are present
                if all(col in df.columns for col in required_columns):
                    dfs.append(df[required_columns])
                else:
                    missing_folders.append(folder)
                    print(f"Skipping {file_path}: Not all required columns found.")
            except Exception as e:
                missing_folders.append(folder)
                print(f"Error processing {file_path}: {e}")
        else:
            missing_folders.append(folder)
            print(f"File not found: {file_path}")

# Concatenate all dataframes if any were loaded
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    # Save to CSV
    combined_df.to_csv(output_csv, index=False)
    # Save to parquet (requires pyarrow or fastparquet to be installed)
    combined_df.to_parquet(output_parquet, index=False)
    print("Data successfully saved as CSV and parquet.")
else:
    print("No data found to process.")

# Write the report of folders missing the required codes
with open(report_file, "w") as f:
    if missing_folders:
        f.write("Folders missing required ICD codes columns or file not found:\n")
        for folder in missing_folders:
            f.write(f"- {folder}\n")
    else:
        f.write("All folders contained the required ICD codes columns.\n")

print(f"ICD extraction report saved to: {report_file}")
