# Data preparation
Here, all notebooks that prepare the data from ornitho.de and ornitho.ch for modeling purposes can be found.<br>
The resulting datasets are all stored under the respective names on the Gdrive [here](https://drive.google.com/drive/folders/1HRut-trvpeH6Iqm7KN79vWGcBLjvCoBH).<br>

***
#### List of Contents:
- `01_Data_Merging.ipynb` : This notebook aligns all features from the separate datasets of Switzerland and Germany and merges them into a master dataset, which contains bird sightings from both countries in the period 2018-2022. → *master_bird_data.csv*
- `02_Dataset_Reduction_27_species.ipynb` : The ornithologists from ornitho.de and ornitho.ch have agreed on 27 bird species for the initial modeling. This notebook generates a reduced dataset from the master dataset only containing these species. → *master_bird_data_selected_species.csv*
- `03_EEA_Grid_Assignment.ipynb` : In this notebook, each bird sighting in the reduced dataset of 27 bird species is mapped to an EEA-Grid, and the ID of the grid is added as an additional feature to the dataset. This step aims to discretize the coordinates. → *selected_bird_species_with_grids.csv*
- `04_data_preparation_summary.ipynb` : This notebook summarizes all steps from the above notebooks 01-03 using methods from our utils module. If you aim to recreate the datasets or get an overview of all preparation steps, use this notebook. For a deeper understanding of the taken steps, refer to notebooks 01-03.
