# abide-brain-graph-learning

## Data download
This project uses preprocessed ABIDE-I ROI time-series data from the ABIDE Preprocessed Connectomes Project. The data were obtained using the CPAC pipeline, filt_global preprocessing strategy, and CC200 atlas (rois_cc200). For download instructions, refer to the official ABIDE download guide http://preprocessed-connectomes-project.org/abide/

## Methods
The repository implements:

- Functional connectivity (FC) and effective connectivity (EC) graph construction.
- Baseline GCN classification (GCN)
- GCN classification with learned ROI identity embeddings (GCN1)
- CORAL-based feature alignment.
- Logistic regression using connectivity features.
- Random-split and leave-one-site-out (LOSO) evaluation.

## Execution
- Graphs were generated from the generate_pytorch_geometric_data.py script.
- Models were tested using loso_experiment across 5 random seeds. Modifications were made within the code to execute the different models and robustness methods.

## Implementation Notice
This code was developed for the experiments conducted in this research project. It has not been packaged as a general-purpose software library, and some scripts depend on the original directory structure and experimental configuration.

## Dependencies
The implementation uses Python and libraries including PyTorch, PyTorch Geometric, NumPy, pandas, and scikit-learn.

## Acknowledgements
The ABIDE preprocessed data were obtained from the Preprocessed Connectomes Project http://preprocessed-connectomes-project.org/abide/. Please refer to the original project for dataset documentation and relevant acknowledgements.