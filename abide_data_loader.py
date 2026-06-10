import numpy as np
import pandas as pd
import os

class AbideDataLoader:
    def __init__(self, data_dir, pheno_df):
        """
        Load data for a single ABIDE subject.
        data_dir : str
            Path to folder containing .1D abide preprocessed files
        pheno_df : pd.DataFrame
            The phenotypic dataframe
        """
        self.pheno_df = pheno_df
        self.data_dir = data_dir

    def load_subject(self, file_name):
        """
        Load data for a single ABIDE subject.
        
        Parameters
        ----------
        subject_id : str
            The FILE_ID of the subject (e.g. 'Caltech_00501456')
        
        
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
        path = os.path.join(self.data_dir, file_name)
        subject_id = file_name.split("_rois")[0]

        if not os.path.exists(path):
            raise FileNotFoundError(f"No .1D file found for subject: {subject_id}")

        # Load ROI time series matrix and transpose so (200, timepoints)
        X_time = np.loadtxt(path).T

        # Get Phenotypic Data
        row = self.pheno_df[self.pheno_df['FILE_ID'] == subject_id]

        if row.empty:
            raise ValueError(f"Subject {subject_id} not found in phenotypic file")

        #DX_GROUP: 1=ASD, 2=TDC convert into 1=ASD, 0=TDC
        dx = row['DX_GROUP'].values[0]
        y = 1 if dx == 1 else 0
        
        site = row['SITE_ID'].values[0]
        id = row['SUB_ID'].values[0]
        
        return X_time, y, site, id