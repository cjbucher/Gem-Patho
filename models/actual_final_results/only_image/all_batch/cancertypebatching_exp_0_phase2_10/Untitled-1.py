# %%
# %%
import pandas as pd
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# %%
# %%
# Initialize dictionary to collect results for genetic models.
# Here we include the original "only_genetics" model.
all_results = defaultdict(list)
base_path = "/n/data2/hms/dbmi/kyu/lab/chb3333/gem-patho/models/actual_final_results"

# Loop over the "only_genetics" model for the given batches and experiments.
for gen_image in ['genetics_and_image', 'only_genetics', "only_image"]:
    for batch in ['all_batch']:
        for exp in ['cancertypebatching_exp_0_phase2_10', 
                    'cancertypebatching_exp_1_phase2_0']:
            path = f"{base_path}/{gen_image}/{batch}/{exp}/global_test_cindices.csv"
            try:
                data = pd.read_csv(path)
                for cancer, c_idx in zip(data['cancer_type'], data['c_index']):
                    all_results['gen_image'].append(gen_image)
                    all_results['batch'].append(batch)
                    all_results['exp'].append(exp)
                    all_results['cancer'].append(cancer)
                    all_results['c_index'].append(c_idx)
            except Exception as e:
                # Skip missing or problematic files.
                print(f"Skipping file {path} due to error: {e}")


# %%
# %%
# Create a DataFrame from the collected results.
df = pd.DataFrame(all_results)
print("Raw Data:")
print(df.head())

# Group by model identifiers and calculate the average c-index and its standard error.
grouped = df.groupby(['gen_image', 'batch', 'exp'])['c_index'].agg(['mean', 'sem']).reset_index()
grouped.rename(columns={'mean': 'avg_c_index', 'sem': 'se_c_index'}, inplace=True)

# Create a column that uniquely identifies each model.
grouped["model"] = grouped["gen_image"] + "_" + grouped["batch"] + "_" + grouped["exp"]

# Compute the 95% confidence interval (CI = 1.96 * standard error).
grouped["ci95"] = 1.96 * grouped["se_c_index"]

print("Grouped Data:")
print(grouped)


# %%
grouped.iloc[[0, 2]] = grouped.iloc[[2, 0]].values
grouped.iloc[[1, 3]] = grouped.iloc[[3, 1]].values

grouped.iloc[[2, 4]] = grouped.iloc[[4, 2]].values
grouped.iloc[[3, 5]] = grouped.iloc[[5, 3]].values


# %%
# %%
# Assign sequential model numbers to each aggregated group.
grouped["model_number"] = range(1, len(grouped) + 1)
print("Model Numbers:")
print(grouped[['model_number', 'model']])


# %%
# %%
# Define model label mapping.
# For the two only_genetics experiments we assign different names,
# and assign names for one_hot and ct_specific groups.
model_labels = {
    "PhasedCTBatchModel": "Phased CT-Batch Model",  # for exp "cancertypebatching_exp_0_phase2_10"
    "UniformBatchModel": "Uniform Batch Model"      # for exp "cancertypebatching_exp_1_phase2_0"
}

# Function to generate the label based on the "model" string.
def get_model_label(model_str):
    if "cancertypebatching_exp_0_phase2_10" in model_str:
         return model_labels["PhasedCTBatchModel"]
    elif "cancertypebatching_exp_1_phase2_0" in model_str:
         return model_labels["UniformBatchModel"]
    else:
         return model_str

# Apply the label mapping.
grouped["model_label"] = grouped["model"].apply(get_model_label)


# %%
grouped["model"]

# %%
grouped

# %%
# %%
# Replace the color_map with:
color_map = {
    "only_genetics": "tab:green",
    "only_image": "tab:orange",  # Your new purple
    "genetics_and_image": "tab:blue"
}
# Function to determine the color based on the model's key.
def get_color(model_str):
    if model_str.startswith("only_genetics"):
        return color_map["only_genetics"]
    elif model_str.startswith("only_image"):
        return color_map["only_image"]
    elif model_str.startswith("genetics_and_image"):
        return color_map["genetics_and_image"]
    else:
        return "tab:gray"

# Map each model to its assigned color.
colors = grouped["model"].apply(get_color)


# %%
# %%
# Create the final bar plot.
plt.figure(figsize=(12, 7))
bars = plt.bar(range(len(grouped)), grouped["avg_c_index"],
               yerr=grouped["ci95"], capsize=5, color=colors)

# Keep numbers at the bottom.
plt.xticks(range(len(grouped)), grouped["model_number"], rotation=45, ha='right')
plt.ylabel("Mean C-Index (Across Cancer Types)")
plt.xlabel("Model Number")
plt.title("Genetic Models: Average C-Index Across Cancer Types with 95% CI")
plt.tight_layout()
plt.grid(axis="y", linestyle="--", alpha=0.7)

# Add numerical labels atop each bar.
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, 0.60, f'{height:.3f}', ha='center', va='bottom', fontsize=10)


# Build a custom legend.
# Build a custom legend that includes model numbers and labels.
legend_patches = []
for _, row in grouped.iterrows():
    legend_label = f"{row['model_number']}: {row['model_label']}"
    legend_patches.append(mpatches.Patch(color=get_color(row['model']), label=legend_label))
plt.legend(handles=legend_patches, title="Model Groups")

plt.show()


# %%
grouped

# %% [markdown]
# METRICS CALCULATION

# %%
cancer_mapping = {
    'ACC': 0, 'BLCA': 1, 'BRCA': 2, 'CESC': 3, 'CHOL': 4, 'CNTL': 5,
    'COAD': 6, 'DLBC': 7, 'ESCA': 8, 'FPPP': 9, 'GBM': 10, 'HNSC': 11,
    'KICH': 12, 'KIRC': 13, 'KIRP': 14, 'LAML': 15, 'LCML': 16, 'LGG': 17,
    'LIHC': 18, 'LUAD': 19, 'LUSC': 20, 'MESO': 21, 'MISC': 22, 'OV': 23,
    'PAAD': 24, 'PCPG': 25, 'PRAD': 26, 'READ': 27, 'SARC': 28, 'SKCM': 29,
    'STAD': 30, 'TGCT': 31, 'THCA': 32, 'THYM': 33, 'UCEC': 34, 'UCS': 35,
    'UVM': 36
}

inv_mapping = {v: k for k, v in cancer_mapping.items()}

# %%
import pandas as pd
import numpy as np

# Replace with your actual file path
file_path = '/home/chb3333/yulab/chb3333/gem-patho/models/actual_final_results/only_genetics/all_batch/cancertypebatching_exp_0_phase2_10/global_test_cindices.csv'

# Load the CSV file
df = pd.read_csv(file_path)

# Calculate the average of the 'c_index' column
mean_c_index = df['c_index'].mean()
print("Average c_index:", mean_c_index)

# Make a copy to avoid SettingWithCopyWarning
filtered_df = df[df['c_index'] > 0.55].copy()

# Count how many rows have a c_index above 0.55
count_over = len(filtered_df)

# Calculate the average of the c_index values for those rows
average_over = filtered_df['c_index'].mean()

print("Count of c_index values over 0.55:", count_over)
print("Average c_index for values over 0.55:", average_over)

# Select the top 5 c_index values
top5 = df['c_index'].nlargest(5)

# Calculate mean and standard deviation
top5_avg = top5.mean()
top5_std = top5.std(ddof=1)  # sample std dev

# Compute standard error
top5_se = top5_std / np.sqrt(len(top5))

# Compute 95% confidence interval using normal approximation (1.96 * SE)
top5_ci95 = 1.96 * top5_se

print("Average of top 5 c_index values:", top5_avg)
print("Standard deviation of top 5 c_index values:", top5_std)
print("Standard error:", top5_se)
print(f"95% CI: [{top5_avg - top5_ci95:.4f}, {top5_avg + top5_ci95:.4f}]")

# Replace cancer_type using inverse mapping
filtered_df['cancer_type'] = filtered_df['cancer_type'].map(inv_mapping)

print("\nRows with c_index > 0.55:")
print(filtered_df)


# %%
# Replace with your actual file path
file_path = '/home/chb3333/yulab/chb3333/gem-patho/models/actual_final_results/only_genetics/all_batch/cancertypebatching_exp_1_phase2_0/global_test_cindices.csv'

# Load the CSV file
df = pd.read_csv(file_path)

# Calculate the average of the 'c_index' column
mean_c_index = df['c_index'].mean()
print("Average c_index:", mean_c_index)

# Make a copy to avoid SettingWithCopyWarning
filtered_df = df[df['c_index'] > 0.55].copy()

# Count how many rows have a c_index above 0.55
count_over = len(filtered_df)

# Calculate the average of the c_index values for those rows
average_over = filtered_df['c_index'].mean()

print("Count of c_index values over 0.55:", count_over)
print("Average c_index for values over 0.55:", average_over)

# Select the top 5 c_index values
top5 = df['c_index'].nlargest(5)

# Calculate mean and standard deviation
top5_avg = top5.mean()
top5_std = top5.std(ddof=1)  # sample std dev

# Compute standard error
top5_se = top5_std / np.sqrt(len(top5))

# Compute 95% confidence interval using normal approximation (1.96 * SE)
top5_ci95 = 1.96 * top5_se

print("Average of top 5 c_index values:", top5_avg)
print("Standard deviation of top 5 c_index values:", top5_std)
print("Standard error:", top5_se)
print(f"95% CI: [{top5_avg - top5_ci95:.4f}, {top5_avg + top5_ci95:.4f}]")

# Replace cancer_type using inverse mapping
filtered_df['cancer_type'] = filtered_df['cancer_type'].map(inv_mapping)

print("\nRows with c_index > 0.55:")
print(filtered_df)

# %%
one_hot_model_path = "/home/chb3333/yulab/chb3333/gem-patho/models/clean_baseline_results/pure_genetics/one_hot_encoded/global_test_cindices.csv"


# Load the CSV file
df = pd.read_csv(one_hot_model_path)

# Calculate the average of the 'c_index' column
mean_c_index = df['c_index'].mean()

print("Average c_index:", mean_c_index)


filtered_df = df[df['c_index'] > 0.55]

# Count how many rows have a c_index above 0.55
count_over = len(filtered_df)

# Calculate the average of the c_index values for those rows
average_over = filtered_df['c_index'].mean()

print("Count of c_index values over 0.55:", count_over)
print("Average c_index for values over 0.55:", average_over)

# Select the top 5 c_index values
top5 = df['c_index'].nlargest(5)

# Calculate mean and standard deviation
top5_avg = top5.mean()
top5_std = top5.std(ddof=1)  # sample std dev

# Compute standard error
top5_se = top5_std / np.sqrt(len(top5))

# Compute 95% confidence interval using normal approximation (1.96 * SE)
top5_ci95 = 1.96 * top5_se

print("Average of top 5 c_index values:", top5_avg)
print("Standard deviation of top 5 c_index values:", top5_std)
print("Standard error:", top5_se)
print(f"95% CI: [{top5_avg - top5_ci95:.4f}, {top5_avg + top5_ci95:.4f}]")

print("\nRows with c_index > 0.55:")
print(filtered_df)

# %%
cancer_type_specific_models = "/home/chb3333/yulab/chb3333/gem-patho/models/per_ct_models/global_test_cindices_by_cancer.csv"


# Load the CSV file
df = pd.read_csv(cancer_type_specific_models)

# Calculate the average of the 'c_index' column
mean_c_index = df['c_index'].mean()

print("Average c_index:", mean_c_index)


filtered_df = df[df['c_index'] > 0.55]

# Count how many rows have a c_index above 0.55
count_over = len(filtered_df)

# Calculate the average of the c_index values for those rows
average_over = filtered_df['c_index'].mean()

print("Count of c_index values over 0.55:", count_over)
print("Average c_index for values over 0.55:", average_over)

# Select the top 5 c_index values
top5 = df['c_index'].nlargest(5)

# Calculate mean and standard deviation
top5_avg = top5.mean()
top5_std = top5.std(ddof=1)  # sample std dev

# Compute standard error
top5_se = top5_std / np.sqrt(len(top5))

# Compute 95% confidence interval using normal approximation (1.96 * SE)
top5_ci95 = 1.96 * top5_se

print("Average of top 5 c_index values:", top5_avg)
print("Standard deviation of top 5 c_index values:", top5_std)
print("Standard error:", top5_se)
print(f"95% CI: [{top5_avg - top5_ci95:.4f}, {top5_avg + top5_ci95:.4f}]")

print("\nRows with c_index > 0.55:")
print(filtered_df)

# %%



