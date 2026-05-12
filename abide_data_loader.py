import numpy as np
import pandas as pd
import os

def load_subject(subject_id, data_dir, pheno_df):
    """
    Load data for a single ABIDE subject.
    
    Parameters
    ----------
    subject_id : str
        The FILE_ID of the subject (e.g. 'Caltech_00501456')
    data_dir : str
        Path to folder containing .1D abide preprocessed files
    pheno_df : pd.DataFrame
        The phenotypic dataframe
    
    Returns
    -------
    X_time : np.ndarray, shape (200, timepoints)
        ROI time series matrix
    y : int
        1 = ASD, 0 = TDC (typical control)
    site : str
        Scanning site (e.g. 'NYU', 'Caltech')
    subject_id : str
        The subject FILE_ID
    """

    # Load ROI time series
    file_name = os.path.join(data_dir, f"{subject_id}_rois_cc200.1D")

    if not os.path.exists(file_name):
        raise FileNotFoundError(f"No .1D file found for subject: {subject_id}")

    # Load ROI time series matrix and transpose so (200, timepoints)
    X_time = np.loadtxt(file_name).T

    # Get Phenotypic Data
    row = pheno_df[pheno_df['FILE_ID'] == subject_id]

    if row.empty:
        raise ValueError(f"Subject {subject_id} not found in phenotypic file")

    #DX_GROUP: 1=ASD, 2=TDC convert into 1=ASD, 0=TDC
    dx = row['DX_GROUP'].values[0]
    y = 1 if dx == 1 else 0
    
    site = row['SITE_ID'].values[0]
    id = row['SUB_ID'].values[0]
    
    return X_time, y, site, id


# Usage
"""
pheno_df = pd.read_csv('Phenotypic_V1_0b_preprocessed1.csv')
data_dir = './abide_data/Outputs/cpac/filt_global/rois_cc200'

# Load a single subject
X_time, y, site, sid = load_subject(
    subject_id='Pitt_0050030',
    data_dir=data_dir,
    pheno_df=pheno_df
)

print(f"Subject:    {sid}")
print(f"Site:       {site}")
print(f"Diagnosis:  {'ASD' if y == 1 else 'TDC'}")
print(f"X_time shape: {X_time.shape}")   # should be (200, timepoints)
"""