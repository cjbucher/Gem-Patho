
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

print(device)

df_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/location_description/description_meta_with_answers.parquet"
df = pd.read_parquet(df_path)

model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True).to(device)

embeddings = model.encode(
    df["generated_description"].astype(str).tolist(),
    task="separation",
    prompt_name="separation",
    batch_size=64,
    show_progress_bar=True
)

df["description_embeddings"] = embeddings.tolist()

output_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/description_embeddings/GenePT/genept_description_embeddings.parquet"
df.to_parquet(output_path, index=False)

output_path = "/home/chb3333/yulab/chb3333/gem-patho/data_extraction/cancertype_location_description/description_embeddings/GenePT/genept_description_embeddings.csv"
df.to_csv(output_path, index=False)