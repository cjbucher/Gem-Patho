import csv

# List of dictionaries containing the study abbreviations and names.
data = [
    {"Study Abbreviation": "LAML", "Study Name": "Acute Myelogenous Leukemia"},
    {"Study Abbreviation": "ACC", "Study Name": "Adrenocortical carcinoma"},
    {"Study Abbreviation": "BLCA", "Study Name": "Bladder Urothelial Carcinoma"},
    {"Study Abbreviation": "LGG", "Study Name": "Brain Lower Grade Glioma"},
    {"Study Abbreviation": "BRCA", "Study Name": "Breast invasive carcinoma"},
    {"Study Abbreviation": "CESC", "Study Name": "Cervical squamous cell carcinoma and endocervical adenocarcinoma"},
    {"Study Abbreviation": "CHOL", "Study Name": "Cholangiocarcinoma"},
    {"Study Abbreviation": "LCML", "Study Name": "Chronic Myelogenous Leukemia"},
    {"Study Abbreviation": "COAD", "Study Name": "Colon adenocarcinoma"},
    {"Study Abbreviation": "CNTL", "Study Name": "Controls"},
    {"Study Abbreviation": "ESCA", "Study Name": "Esophageal carcinoma"},
    {"Study Abbreviation": "FPPP", "Study Name": "FFPE Pilot Phase II"},
    {"Study Abbreviation": "GBM", "Study Name": "Glioblastoma multiforme"},
    {"Study Abbreviation": "HNSC", "Study Name": "Head and Neck squamous cell carcinoma"},
    {"Study Abbreviation": "KICH", "Study Name": "Kidney Chromophobe"},
    {"Study Abbreviation": "KIRC", "Study Name": "Kidney renal clear cell carcinoma"},
    {"Study Abbreviation": "KIRP", "Study Name": "Kidney renal papillary cell carcinoma"},
    {"Study Abbreviation": "LIHC", "Study Name": "Liver hepatocellular carcinoma"},
    {"Study Abbreviation": "LUAD", "Study Name": "Lung adenocarcinoma"},
    {"Study Abbreviation": "LUSC", "Study Name": "Lung squamous cell carcinoma"},
    {"Study Abbreviation": "DLBC", "Study Name": "Lymphoid Neoplasm Diffuse Large B-cell Lymphoma"},
    {"Study Abbreviation": "MESO", "Study Name": "Mesothelioma"},
    {"Study Abbreviation": "MISC", "Study Name": "Miscellaneous"},
    {"Study Abbreviation": "OV", "Study Name": "Ovarian serous cystadenocarcinoma"},
    {"Study Abbreviation": "PAAD", "Study Name": "Pancreatic adenocarcinoma"},
    {"Study Abbreviation": "PCPG", "Study Name": "Pheochromocytoma and Paraganglioma"},
    {"Study Abbreviation": "PRAD", "Study Name": "Prostate adenocarcinoma"},
    {"Study Abbreviation": "READ", "Study Name": "Rectum adenocarcinoma"},
    {"Study Abbreviation": "SARC", "Study Name": "Sarcoma"},
    {"Study Abbreviation": "SKCM", "Study Name": "Skin Cutaneous Melanoma"},
    {"Study Abbreviation": "STAD", "Study Name": "Stomach adenocarcinoma"},
    {"Study Abbreviation": "TGCT", "Study Name": "Testicular Germ Cell Tumors"},
    {"Study Abbreviation": "THYM", "Study Name": "Thymoma"},
    {"Study Abbreviation": "THCA", "Study Name": "Thyroid carcinoma"},
    {"Study Abbreviation": "UCS", "Study Name": "Uterine Carcinosarcoma"},
    {"Study Abbreviation": "UCEC", "Study Name": "Uterine Corpus Endometrial Carcinoma"},
    {"Study Abbreviation": "UVM", "Study Name": "Uveal Melanoma"}
]
 
output_csv = "tcga_study_abbreviations.csv"
with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    fieldnames = ["Study Abbreviation", "Study Name"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in data:
        writer.writerow(row)
 
print(f"CSV file created: {output_csv}")
