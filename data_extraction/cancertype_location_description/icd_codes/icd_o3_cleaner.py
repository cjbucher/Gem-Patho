import pandas as pd

input_file = "/home/chb3333/yulab/chb3333/data_extraction/sample_location_tumor_description/icd_codes/ICD-O3 Topography.xlsx"
output_file = "/home/chb3333/yulab/chb3333/data_extraction/sample_location_tumor_description/icd_codes/ICD-O3 Topography.csv"

df_filtered = pd.read_excel(input_file)

df_filtered = df_filtered[["icdo3_code", "description"]]

df_filtered.to_csv(output_file, index=False)

print(f"Filtered CSV saved to: {output_file}")