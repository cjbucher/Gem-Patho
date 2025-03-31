import os
import requests

# Base directory where you have your TCGA cancer-set folders
base_dir = '/home/chb3333/yulab/chb3333/gem-patho/data_extraction/CBIO_extraction/download'

# List all subdirectories in base_dir that end with "tcga_pan_can_atlas_2018"
cancer_set_dirs = [d for d in os.listdir(base_dir)
                   if os.path.isdir(os.path.join(base_dir, d)) and d.endswith("tcga_pan_can_atlas_2018")]

print(f"Found {len(cancer_set_dirs)} cancer set folders: {cancer_set_dirs}")

# Loop over each folder and download the data_clinical_patient.txt file
for folder in cancer_set_dirs:
    folder_path = os.path.join(base_dir, folder)
    # Construct the URL based on the folder name
    url = f"https://media.githubusercontent.com/media/cBioPortal/datahub/refs/heads/master/public/{folder}/data_clinical_patient.txt"
    output_file = os.path.join(folder_path, "data_clinical_patient.txt")
    print(f"Downloading clinical data for {folder}...")
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"Downloaded to: {output_file}")
        else:
            print(f"ERROR: Failed to download for {folder} (status code {response.status_code}).")
    except Exception as e:
        print(f"ERROR downloading for {folder}: {e}")
