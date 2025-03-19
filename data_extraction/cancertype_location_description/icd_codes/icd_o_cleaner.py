import pandas as pd

input_file = "/home/chb3333/yulab/chb3333/data_extraction/sample_location_tumor_description/icd_codes/ICD-O-3.2_MFin_17042019_web.xls"
output_file = "/home/chb3333/yulab/chb3333/data_extraction/sample_location_tumor_description/icd_codes/ICD-O-3.2_MFin_17042019_web.csv"

df = pd.read_excel(input_file, sheet_name=0, dtype=str, header=0)

# Debug: Print column names to check for issues
print("Columns in the dataframe:", df.columns.tolist())

# Ensure consistent column names by stripping spaces
df.columns = df.columns.str.strip()

# Debug: Check if 'Level' is in the columns
if "Level" not in df.columns:
    print(f"'Level' column not found. Columns available: {df.columns.tolist()}")

# Filter only rows where 'Level' is 'Preferred'
df_filtered = df[df["Unnamed: 1"].str.strip() == "Preferred"]

# Keep only the 'ICDO3.2' and 'Term' columns
df_filtered = df_filtered[["ICD-O- Third Edition, Second Revision Morphology", "Unnamed: 2"]]
df_filtered.columns = ["Codes", "Morphology_Description"]
# Save the filtered data to a CSV file
df_filtered.to_csv(output_file, index=False)

print(f"Filtered CSV saved to: {output_file}")