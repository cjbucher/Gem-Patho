# %%
print("Starting the script...")

# %%
import pandas as pd
import glob
import os
from dotenv import load_dotenv


env_path = '/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/OpenAI_key.env'
load_dotenv(env_path)
# %%
# ----- Part 1: Process the TCGA sample sheet -----

# %%
print("Loading the cleaned TCGA sample sheet...")
sample_sheet_path = "/home/chb3333/yulab/chb3333/data_extraction/wxs_sample_sheet_clean.tsv"
sample_df = pd.read_csv(sample_sheet_path, sep="\t")
print("Sample sheet loaded. Total rows:", len(sample_df))

# %%
print("Filtering for TCGA projects...")
tcga_df = sample_df[sample_df["Project ID"].str.contains("TCGA", na=False)].copy()
print("TCGA projects filtered. Rows after filtering:", len(tcga_df))

# %%
print("Normalizing sample types...")
def normalize_sample_type(sample_str):
    parts = [s.strip() for s in str(sample_str).split(",")]
    return ", ".join(sorted(parts))

# %%
tcga_df["Normalized Sample Type"] = tcga_df["Sample Type"].apply(normalize_sample_type)
print("Sample types normalized.")

# %%
print("Extracting TCGA Cancer Type Abbreviation...")
tcga_df["Cancer Type Abbrev"] = tcga_df["Project ID"].str.replace("TCGA-", "", regex=False)

# %%
print("Merging full TCGA Cancer Type name from mapping file...")
tcga_map_path = "/home/chb3333/yulab/chb3333/data_extraction/sample_location_tumor_description/tcga_study_abbreviations.csv"
tcga_map_df = pd.read_csv(tcga_map_path)
tcga_df = pd.merge(tcga_df, tcga_map_df, left_on="Cancer Type Abbrev", right_on="Study Abbreviation", how="left")
print("TCGA mapping merge complete. Rows in tcga_df:", len(tcga_df))

# %%
tcga_df = tcga_df.drop(['Cancer Type Abbrev', "File ID", "File Name", "Data Category", "Data Type" ], axis=1)

# %%
tcga_df

# %%
icd_df = pd.read_parquet("/home/chb3333/yulab/chb3333/data_extraction/sample_location_tumor_description/icd_codes/icd_data.parquet")

# %%
icd_df

# %%
merged_df = pd.merge(tcga_df, icd_df, left_on="Case ID", right_on="PATIENT_ID", how="inner")

# %%
merged_df

# %%
merged_df = merged_df.drop(columns=['Project ID', 'Sample ID', 'Sample Type'])

# Select only the desired columns
result_df = merged_df[['Case ID', 'CANCER_TYPE_ACRONYM', 'ICD_10', 'ICD_O_3_HISTOLOGY', 'ICD_O_3_SITE', 'Normalized Sample Type']]



# %%
# Display the resulting dataframe
result_df.head()

# %%
group_cols = ['CANCER_TYPE_ACRONYM', 'ICD_10', 'ICD_O_3_HISTOLOGY', 'ICD_O_3_SITE', 'Normalized Sample Type']

# Group by these columns and aggregate the Case IDs into a list
grouped_df = result_df.groupby(group_cols)['Case ID'].apply(list).reset_index()

# If you prefer a comma-separated string instead of a list, you can convert it as follows:
grouped_df['Case IDs'] = grouped_df['Case ID'].apply(lambda ids: ', '.join(ids))

# Drop the original 'Case ID' column (if only the aggregated version is needed)
grouped_df = grouped_df.drop(columns=['Case ID'])

# %%
grouped_df

# %%
# Translating Codes to Text

# %%

file_path = "/home/chb3333/yulab/chb3333/data_extraction/sample_location_tumor_description/icd_codes/ICD-O-3.2_MFin_17042019_web.csv"

# Load the Excel file
morphology_description = pd.read_csv(file_path)

# %%
morphology_description.columns

# %%
file_path = "/home/chb3333/yulab/chb3333/data_extraction/sample_location_tumor_description/icd_codes/ICD-O3 Topography.csv"
icd_o3 = pd.read_csv(file_path)

# %%
icd_o3.columns

# %%
morphology_description,


# %%
icd_mapping = dict(zip(icd_o3['icdo3_code'], icd_o3['description']))
morph_mapping = dict(zip(morphology_description['Codes'], morphology_description['Morphology_Description']))

# Translate ICD_10 and ICD_O_3_SITE codes using icd_o3 mapping
grouped_df['ICD_10_desc'] = grouped_df['ICD_10'].map(icd_mapping)
grouped_df['ICD_O_3_SITE_desc'] = grouped_df['ICD_O_3_SITE'].map(icd_mapping)

# Translate ICD_O_3_HISTOLOGY codes using morphology_description mapping
grouped_df['ICD_O_3_HISTOLOGY_desc'] = grouped_df['ICD_O_3_HISTOLOGY'].map(morph_mapping)

study_abbrev = pd.read_csv("/home/chb3333/yulab/chb3333/data_extraction/sample_location_tumor_description/tcga_study_abbreviations.csv")
study_mapping = dict(zip(study_abbrev['Study Abbreviation'], study_abbrev['Study Name']))
grouped_df['CANCER_TYPE_NAME'] = grouped_df['CANCER_TYPE_ACRONYM'].map(study_mapping)

# %%
print('Saving: /home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/location_description/description_meta.parquet')
grouped_df.to_parquet("/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/location_description/description_meta.parquet")

# %%
chatgpt_query_info = grouped_df[['CANCER_TYPE_NAME', 'ICD_10_desc', 'ICD_O_3_HISTOLOGY_desc', 'ICD_O_3_SITE_desc', 'Normalized Sample Type']]

# %%
#chatgpt_query_info_deduplicated = chatgpt_query_info.drop_duplicates()
chatgpt_query_info_deduplicated = chatgpt_query_info


# %%
chatgpt_query_info_deduplicated = chatgpt_query_info_deduplicated.rename(columns={'Normalized Sample Type': 'Normalized_Sample_Type'})

# %%
final_chatgpt_query_info = chatgpt_query_info_deduplicated.fillna("UNKOWN")

# %%
final_chatgpt_query_info

# %%
template = """
response = client.chat.completions.create(
  model="gpt-4o",
  messages=[
    {{
      "role": "system",
      "content": [
        {{
          "type": "text",
          "text": "You are an expert oncologist specializing in cancer survival. You are explaining complex oncological and genomic factors to another expert in a structured, precise manner. Focus on step-by-step reasoning, integrating knowledge of tumor origin, histology, genetic mutations, and sampling biases. Avoid redundant explanations and ensure biological clarity while emphasizing survival impact."
        }}
      ]
    }},
    {{
      "role": "user",
      "content": [
        {{
          "type": "text",
          "text": "Describe how {CANCER_TYPE_NAME} (per ICD-10: {ICD_10_desc}) impacts survival, focusing on:\n- Tumor Origin: How {ICD_O_3_SITE_desc} influences metastatic patterns, patient survival, and treatment accessibility.\n- Histology: How {ICD_O_3_HISTOLOGY_desc} interacts with mutation profiles to drive outcomes.\n- Sampling Bias: Limitations of {Normalized_Sample_Type} in genomic analysis.\n- Key Genes: Identify 5-8 survival-associated genes, explaining their biological mechanisms (e.g., proliferation, immune response, apoptosis regulation).\n\nFor each aspect, explain step by step:\n- How it influences survival (positive/negative)\n- Biological rationale without excessive jargon\n- Potential biases in genomic interpretation\n\nConclude with a synthesis of key survival risk factors, emphasizing clinically relevant insights and their impact on patient survival."
        }}
      ]
    }}
  ],
  response_format={{ "type": "text" }},
  temperature=1,
  max_completion_tokens=2048,
  top_p=1,
  frequency_penalty=0,
  presence_penalty=0
)
"""

# %%
prompts = []
for index, row in final_chatgpt_query_info.iterrows():
    prompt_filled = template.format(
        CANCER_TYPE_NAME=row["CANCER_TYPE_NAME"],
        ICD_10_desc=row["ICD_10_desc"],
        ICD_O_3_SITE_desc=row["ICD_O_3_SITE_desc"],
        ICD_O_3_HISTOLOGY_desc=row["ICD_O_3_HISTOLOGY_desc"],
        Normalized_Sample_Type=row["Normalized_Sample_Type"]
    )
    prompts.append(prompt_filled)

# Now 'prompts' is a list where each element is a fully formatted prompt.
print(prompts[0])  # Print the first prompt as an example.

# %%
from openai import OpenAI

# %%
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# %%
# def get_response(prompt):
#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4o",
#             messages=[
#                 {
#                     "role": "system",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": (
#                                 "You are an expert oncologist specializing in cancer survival. "
#                                 "You are explaining complex oncological and genomic factors to another expert "
#                                 "in a structured, precise manner. Focus on step-by-step reasoning, integrating knowledge "
#                                 "of tumor origin, histology, genetic mutations, and sampling biases. Avoid redundant explanations "
#                                 "and ensure biological clarity while emphasizing survival impact."
#                             )
#                         }
#                     ]
#                 },
#                 {
#                     "role": "user",
#                     "content": [
#                         {
#                             "type": "text",
#                             "text": prompt
#                         }
#                     ]
#                 }
#             ],
#             response_format={"type": "text"},
#             temperature=1,
#             max_tokens=2048,
#             top_p=1,
#             frequency_penalty=0,
#             presence_penalty=0
#         )
#         return response
#     except Exception as e:
#         print(f"Error processing prompt: {e}")
#         return None


# %%
responses = []

for idx, prompt in enumerate(prompts):
    print(f"Processing prompt {idx+1}/{len(prompts)}...")
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an expert oncologist specializing in cancer survival. You are explaining complex oncological and genomic factors to another expert in a structured, precise manner. Focus on step-by-step reasoning, integrating knowledge of tumor origin, histology, genetic mutations, and sampling biases. Avoid redundant explanations and ensure biological clarity while emphasizing survival impact."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            response_format={"type": "text"},
            temperature=1,
            max_completion_tokens=2048,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        responses.append(response)
        print(response)
    except Exception as e:
        print(f"Error processing prompt {idx+1}: {e}")

# %%
data = []
for prompt, resp in zip(prompts, responses):
    try:
        # Extract the text answer from the ChatCompletion object
        answer_text = resp.choices[0].message.content
    except Exception as e:
        answer_text = None
        print(f"Error extracting answer: {e}")
    data.append({
        "prompt": prompt,
        "response": answer_text
    })

# Create a DataFrame from the collected data
df_responses = pd.DataFrame(data)

csv_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/location_description/chat_responses.csv"
parquet_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/location_description/chat_responses.parquet"

# Save the DataFrame as a CSV file
df_responses.to_csv(csv_path, index=False)
print(f"Responses saved to {csv_path}")

# Save the DataFrame as a Parquet file (requires pyarrow or fastparquet)
df_responses.to_parquet(parquet_path, index=False)
print(f"Responses saved to {parquet_path}")

# %%



