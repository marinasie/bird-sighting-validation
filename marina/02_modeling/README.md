# Data preparation
Here, all notebooks that implement models for the data from ornitho.de and ornitho.ch can be found.<br>

***
#### List of Contents:
- `01_Emergent_Filters.ipynb` : This notebook implements the Emergent Filters approach as applied at eBird. We want to use it as a benchmark.
- `02_Outlier_Detection.ipynb` : This notebook implements our Outlier Detection approach which aims to identify anomalous data points. We employ DBSCAN, Isolation Forest and AutoEncoder and provide an initial assessment of the model's performance on the water pipit. The primary objective of this notebook is to establish the modeling pipeline, making it easier to integrate subsequent optimizations based on our evaluation investigations.