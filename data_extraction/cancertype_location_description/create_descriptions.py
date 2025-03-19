# %%
import pandas as pd
import os
from dotenv import load_dotenv
from openai import OpenAI

env_path = '/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/OpenAI_key.env'
load_dotenv(env_path)

# %%
print("Loading existing dataframe from description_meta.parquet...")
grouped_df = pd.read_parquet("/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/location_description/description_meta.parquet")
print("Dataframe loaded. Shape:", grouped_df.shape)

# %%
# Ensure the column naming matches what we expect (rename if necessary)
if "Normalized Sample Type" in grouped_df.columns:
    grouped_df = grouped_df.rename(columns={"Normalized Sample Type": "Normalized_Sample_Type"})

# Build the query dataframe, including the 'Case IDs'
chatgpt_query_info = grouped_df[['CANCER_TYPE_NAME', 'ICD_10_desc', 'ICD_O_3_HISTOLOGY_desc', 'ICD_O_3_SITE_desc', 'Normalized_Sample_Type', 'Case IDs']]
chatgpt_query_info = chatgpt_query_info.fillna("UNKOWN")

# %%
# Define the prompt template
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
# Build the list of prompts and store the corresponding Case IDs
prompts = []
case_ids_list = []

for index, row in chatgpt_query_info.iterrows():
    prompt_filled = template.format(
        CANCER_TYPE_NAME=row["CANCER_TYPE_NAME"],
        ICD_10_desc=row["ICD_10_desc"],
        ICD_O_3_SITE_desc=row["ICD_O_3_SITE_desc"],
        ICD_O_3_HISTOLOGY_desc=row["ICD_O_3_HISTOLOGY_desc"],
        Normalized_Sample_Type=row["Normalized_Sample_Type"]
    )
    prompts.append(prompt_filled)
    case_ids_list.append(row["Case IDs"])

print("First prompt example:")
print(prompts[0])

# %%
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        print("Response received.")
    except Exception as e:
        print(f"Error processing prompt {idx+1}: {e}")

# %%
# Build a DataFrame that includes the prompt, response, and associated Case IDs
data = []
for prompt, resp, case_ids in zip(prompts, responses, case_ids_list):
    try:
        answer_text = resp.choices[0].message.content
    except Exception as e:
        answer_text = None
        print(f"Error extracting answer: {e}")
    data.append({
        "prompt": prompt,
        "response": answer_text,
        "Case IDs": case_ids
    })

df_responses = pd.DataFrame(data)

csv_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/location_description/chat_responses.csv"
parquet_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/location_description/chat_responses.parquet"

df_responses.to_csv(csv_path, index=False)
print(f"Responses saved to {csv_path}")

df_responses.to_parquet(parquet_path, index=False)
print(f"Responses saved to {parquet_path}")
