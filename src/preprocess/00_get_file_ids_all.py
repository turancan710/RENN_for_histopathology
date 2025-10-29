import pandas as pd
import requests
import json
#import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
data_dir = project_root / 'data'
data_dir.mkdir(parents=True, exist_ok=True)

files_ep = "https://api.gdc.cancer.gov/files"

fields = ["file_id",
          "file_name",
          "experimental_strategy",
          "cases.case_id",
          "cases.submitter_id",
          "cases.project.project_id",
          "cases.tissue_source_site.code",
          "cases.tissue_source_site.name"
         ]

fields = ','.join(fields)

params = {"filters": 
          json.dumps({
              "op": "and", "content": [
                     {"op": "in", "content": {"field": "cases.project.project_id", "value": ["TCGA-BRCA"]}},
                     {"op": "in", "content": {"field": "files.data_format", "value": ["SVS"]}},
                     {"op": "in", "content": {"field": "files.experimental_strategy","value": ["Diagnostic Slide"]}}
                     ]}),
          "fields": fields,
          "format": "JSON",
          "size": 1500
         }

response = requests.get(files_ep, params=params)
file_ids = response.json()["data"]["hits"]

#print(json.dumps(file_ids, indent=2))

file_id_list = []

for case in file_ids:
    case_id = case["cases"][0].get("case_id", None)
    submitter_id = case["cases"][0].get("submitter_id", None)
    project = case["cases"][0].get("project" , {}).get("project_id", None)
    tss_code = case["cases"][0].get("tissue_source_site", {}).get("code", None)
    tss_name = case["cases"][0].get("tissue_source_site", {}).get("name", None)
    file_name = case.get("file_name", None)
    file_id = case.get("file_id", None)
    file_type = case.get("experimental_strategy", None)

    file_id_list.append(
        {
            "case_id": case_id,
            "submitter_id": submitter_id,
            "project": project,
            "tss_code": tss_code,
            "tss_name": tss_name,
            "file_name": file_name,
            "file_id": file_id,
            "file_type": file_type
        })

df = pd.DataFrame(file_id_list)

#print(df.head())
out_path = data_dir / "tcga_brca_file_ids.csv"
df.to_csv(out_path, index=False)