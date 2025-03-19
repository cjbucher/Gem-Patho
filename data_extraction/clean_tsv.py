import pandas as pd

def clean_case_id(case_id):
    """
    Splits a case_id string on commas, trims whitespace, removes duplicates,
    and rejoins the unique values with a comma and space.
    """
    if not isinstance(case_id, str):
        return case_id  # return as-is if not a string
    # Split the string by comma and remove extra whitespace
    parts = [part.strip() for part in case_id.split(',')]
    # Remove duplicates while preserving order
    seen = set()
    unique_parts = []
    for part in parts:
        if part not in seen:
            unique_parts.append(part)
            seen.add(part)
    return ', '.join(unique_parts)

# Define the input and output file paths
input_file = '/home/chb3333/yulab/chb3333/data_extraction/wxs_sample_sheet.tsv'
output_file = '/home/chb3333/yulab/chb3333/data_extraction/wxs_sample_sheet_clean.tsv'

# Read the TSV file
df = pd.read_csv(input_file, sep='\t')

# Apply the cleaning function to the "Case ID" column
df['Case ID'] = df['Case ID'].apply(clean_case_id)

# Save the cleaned DataFrame to a new TSV file
df.to_csv(output_file, sep='\t', index=False)

print(f"Cleaned file saved to {output_file}")
