import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# -------------------------------
# 1. Set up the device (GPU if available)
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# -------------------------------
# 2. Load and merge the data
# -------------------------------
df_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/location_description/description_meta_with_answers.parquet"
df = pd.read_parquet(df_path)

df_path2 = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/location_desciption_withoutICD/description_meta_with_answers.parquet"
df2 = pd.read_parquet(df_path2)

df = pd.concat([df, df2], ignore_index=True, sort=False)

# -------------------------------
# 3. Load the Stella model via SentenceTransformer
# -------------------------------
model_name = "dunzhang/stella_en_400M_v5"
model = SentenceTransformer(
    model_name,
    trust_remote_code=True,
    device=device,
    config_kwargs={"use_memory_efficient_attention": False, "unpad_inputs": False}
)
print(f"Loaded model {model_name} on {device}.")

# -------------------------------
# 4. Encode the descriptions
# -------------------------------
print("Encoding descriptions...")
# Convert descriptions to string and encode them (batch_size can be adjusted based on your GPU memory)
embeddings = model.encode(
    df["generated_description"].astype(str).tolist(),
    batch_size=64,
    show_progress_bar=True
)

# Add embeddings as a new column in the DataFrame
df["description_embeddings"] = [emb.tolist() for emb in embeddings]

# -------------------------------
# 5. Save the embeddings to files
# -------------------------------
# Define the output path for the Parquet file (using Stella)
output_parquet_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/description_embeddings/Stella/stella_description_embeddings.parquet"
df.to_parquet(output_parquet_path, index=False)
print("Saved embeddings to", output_parquet_path)

# Also save to CSV if needed
output_csv_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/description_embeddings/Stella/stella_description_embeddings.csv"
df.to_csv(output_csv_path, index=False)
print("Saved embeddings to", output_csv_path)
