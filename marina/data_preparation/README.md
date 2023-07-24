# Data preparation
Here, all notebooks that analyze and prepare the data from ornitho.de and ornitho.ch for modeling purposes can be found.<br>
The resulting datasets are all stored under the respective names on the Gdrive [here](https://drive.google.com/drive/folders/1HRut-trvpeH6Iqm7KN79vWGcBLjvCoBH).<br>
The following notebooks can be found here:

- `00_Data_Merging.ipynb` : This notebook aligns all features from the separate datasets of Switzerland and Germany and merges them into a master dataset, which contains bird sightings from both countries in the period 2018-2022. → *master_bird_data.csv*
- `01_Bird_Analysis.ipynb` : This notebook provides an initial look at the data and conducts an explanatory data analysis to explore various aspects.
- `02_Dataset_Reduction_27_species.ipynb` : The ornithologists from ornitho.de and ornitho.ch have agreed on 27 bird species for the initial modeling. This notebook generates a reduced dataset from the master dataset. → *master_bird_data_selected_species.csv*
- `03_EEA_Grid_Assignment.ipynb` : In this notebook, each bird sighting in the reduced dataset of 27 bird species is mapped to an EEA-Grid, and the ID of the grid is added as an additional feature to the dataset. This step aims to discretize the coordinates. → *selected_bird_species_with_grids.csv*


