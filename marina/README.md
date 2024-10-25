<img src="../img/README_marina.png" alt="A buzzard" style="width: 100%; border-radius: 20px;"/>

In this subsection, all notebooks related to data preparation, model design, and model evaluation for identifying anomalous bird observations in data from *ornitho.ch* and *ornitho.de* are collected.
<br><br><br>

## Structure

**Part of Grundprojekt:**
- `01_data_preparation` This folder contains all data preparation notebooks. They document and execute the step-by-step process from raw data to the datasets used for model training. Further, a summary for developers is provided.

- `02_modeling` This folder includes all modeling approaches for the identification of anomalous bird observations. This encompasses the implementation of Emergent Filters, as well as the development of Outlier Detection ML algorithms such as DBSCAN, Isolation Forest, and AutoEncoder. Evaluation and tuning are not part of this section; they will be conducted by @Simon Birker.

**Part of Hauptprojekt:**
- `04_drift_analysis` In this folder, all notebooks related to drift analysis are collected, including data drift analysis, time series construction, preprocessing, and drift detection.
<br><br><br>

## Dataset Download
The required datasets to run each Notebook are linked in the respective Notebooks.<br>
All data can be found here: https://drive.google.com/drive/folders/1Of664UN6K9zs8SHYmgKMFKQ1gjXanwH9
<br><br><br>

## Provided file types
Each Notebook is provided as `.ipynb` and as `.html`. 

Use `.ipynb` files to:
- run the code locally
- view the executed notebook with static outputs on GitHub

Use `.html` files to:
- view the executed notebook with interactive plots locally
<br><br><br>

## Setup

#### 1. Clone repository

```console
git clone https://github.com/marinasie/bird-sighting-validation.git
```
#### 2. Install dependencies

```
cd bird-sighting-validation/marina/
pip install -r requirements.txt
```
