import io
import time
import pandas as pd
import requests
import json
from pathlib import Path

# === dirs ====
project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / 'data'

# === get case_ids (use as key to merge with tiles later) ========================
#case ids , submitter ids (-> gdc api as endpoint https://docs.gdc.cancer.gov/API/Users_Guide/Python_Examples/ )

cases_ep = "https://api.gdc.cancer.gov/cases"

fields = [
    "case_id", 
    "submitter_id"
    ]

fields = ','.join(fields)

params = {
    "fields": fields,
    "filters": json.dumps({
        "op": "in",
        "content": {
            "field": "project.project_id", "value": ["TCGA-BRCA"]
            }
        }),
    "format": "JSON",
    "size": 1500
    }

response = requests.get(cases_ep, params = params)
cases_data = response.json()["data"]["hits"]

case_ids = [case["case_id"] for case in cases_data]
#print(case_ids)

submitter_ids = { case["case_id"]: case["submitter_id"] for case in cases_data }
#print(submitter_ids)


# ============download gene expression files (especially: MKI67 TPM = ENSG00000148773.14) ============================
gene_ep = "https://api.gdc.cancer.gov/files"

fields = [
    "file_id",
    "cases.submitter_id",
    "cases.case_id"
    ]

fields = ",".join(fields)

gene_files = {}   # gene_files: {case_id : file_id}
batch_size = 100

for i in range (0 , len(case_ids), batch_size):
  batch = case_ids[i:i+batch_size]

  params = {
    "fields": fields ,
    "filters": json.dumps({
        "op": "and",
        "content": [
            {"op": "in" , "content": { "field": "cases.case_id", "value": batch }},
            {"op": "in",  "content": { "field": "files.data_type", "value": ["Gene Expression Quantification"] }}
        ]
    }),
    "format": "JSON",
    "size": batch_size
  }

  response = requests.get(gene_ep , params = params)
  batch_data = response.json()["data"]["hits"]

  for file in batch_data:
    case_id = file["cases"][0]["case_id"]
    gene_files[case_id] = file["file_id"]


max_retry = 10
delay = 5

genes = {
    "MKI67": "ENSG00000148773.14",
    "ERBB2": "ENSG00000141736.14",
    "ESR1": "ENSG00000091831.24",
    "PGR": "ENSG00000082175.15",
}

gene_exp = {gene : {} for gene in genes} # nested dictionary {MKI67: {} , ERBB2: {} , ...}


for case_id , file_id in gene_files.items():

    file_url = f"https://api.gdc.cancer.gov/data/{file_id}"

    for attempt in range(max_retry):
      try:
        response = requests.get(file_url, timeout=60)
        response.raise_for_status()


    #if response.status_code == 200:    #200 = success
        df = pd.read_csv(io.StringIO(response.text), sep="\t", header=1, skiprows=0 )
        #print(df[df["gene_id"] == "ENSG00000148773.14"])

        for gene , gene_code in genes.items():
          gene_exp[gene][case_id] = df[df["gene_id"] == gene_code]["tpm_unstranded"].values[0]
        #mki67 = df[df["gene_id"] == "ENSG00000148773.14"]["tpm_unstranded"].values[0]
        #mki67_exp[case_id] = mki67
        break #exit  loop if success

      except requests.exceptions.RequestException as e:
          print(f"Attempt {attempt +1 }: Download failed {file_id},  Retry in {delay} sec. {e}")
          time.sleep(delay)
      except  (KeyError, IndexError) as e:
          print(f"Error downloading mki67 for file: {file_id} , case: {case_id} : {e}")
          break

#print(mki67_exp)    #note: if tsv manually downloaded from gdc, gene expression * 10

df = pd.DataFrame({"case_id": case_ids})

df["submitter_id"] = df["case_id"].map(submitter_ids)

#note: gene_exp is "nested" dictionary: each gene is key, hand has a further dict. as value, within those case_id = key, gene expression = value
# e.g. gene_exp["MKI67"] calls the MKI67 dictionary (case_id: gene value) within gene_exp, .map() function maps via keys (case id)
df["MKI67_tpm_unst"] = df["case_id"].map(gene_exp["MKI67"])
df["ERBB2_tpm_unst"] = df["case_id"].map(gene_exp["ERBB2"])
df["ESR1_tpm_unst"] = df["case_id"].map(gene_exp["ESR1"])
df["PGR_tpm_unst"] = df["case_id"].map(gene_exp["PGR"])

#print(df.head())
out_path = data_dir / "tcga_brca_gene_expression_tpm_unstr.csv"
df.to_csv(out_path)