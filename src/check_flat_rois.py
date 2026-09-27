# check_flat_rois.py
import numpy as np
import pandas as pd
import os
from abide_data_loader import AbideDataLoader

pheno_df = pd.read_csv('../Phenotypic_V1_0b_preprocessed1.csv')
data_dir = '../abide_data/Outputs/cpac/filt_global/rois_cc200'
data_loader = AbideDataLoader(data_dir=data_dir, pheno_df=pheno_df)

records = []

for path in os.listdir(data_dir):
    try:
        timeseries, label, site, sid, _, _= data_loader.load_subject(path)
        stds = timeseries.std(axis=1)
        flat_rois = np.where(stds == 0)[0]
        records.append({
            "subject_id": sid,
            "site": site,
            "label": label,
            "n_flat": len(flat_rois),
            "flat_roi_indices": flat_rois.tolist()
        })
    except Exception as e:
        print(f"Failed on {path}: {e}")

df = pd.DataFrame(records)
df.to_csv("flat_roi_diagnostic.csv", index=False)

print(f"Checked {len(df)} subjects")
print(f"Subjects with 0 flat ROIs: {(df['n_flat'] == 0).sum()}")
print(f"Subjects with >5 flat ROIs: {(df['n_flat'] > 5).sum()}")

from collections import Counter
all_flat = [r for row in df["flat_roi_indices"] for r in row]
print("\nMost commonly flat ROIs across subjects:")
print(Counter(all_flat).most_common(10))

print("\nFlat-ROI rate by site:")
print(df.groupby("site")["n_flat"].mean().sort_values(ascending=False))