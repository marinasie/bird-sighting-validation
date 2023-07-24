# Data preparation
Hier liegen alle Notebooks, die die Daten von ornitho.de und ornitho.ch analysieren und für die Modellierungen vorbereiten.<br>
Die jeweils entstehenden Datensets sind alle unter dem jeweils angegebenen Namen auf der Gdrive [hier](https://drive.google.com/drive/folders/1HRut-trvpeH6Iqm7KN79vWGcBLjvCoBH) abgelegt.<br>
Es liegen folgende Notebooks vor:

- `00_Data_Merging.ipynb` : Dieses Notebook aligned alle features der beiden getrennten Datensätze von der Schweiz und Deutschland und merged sie in ein Master-Datenset, das Vogelsichtungen von beiden Ländern im Zeitraum 2018-2022 verzeichnet. &rarr; *master_bird_data.csv*
- `01_Bird_Analysis.ipynb` : In diesem Notebook wird anhand einer explanatory data analysis ein erster Blick auf die Daten geworfen und verschiedene Aspekte analysiert.
- `02_Dataset_Reduction_27_species.ipynb` : Die Ornithologen von ornitho.de und ornitho.ch haben sich auf 27 Vogelarten geeinigt, auf das die ersten Modellierungen beschränkt werden soll. In diesem Notebook wird ein reduziertes Datenset aus dem Master-Dataset generiert. &rarr; *master_bird_data_selected_species.csv*
- `03_EEA_Grid_Assignment.ipynb` : In diesem Notebook wird jede Vogelsichtung in dem auf 27 Vogelarten reduzierte Datenset auf ein EEA-Grid gemappt und die ID des Grids als zusätzliches feature in das Datenset aufgenommen. Damit möchten wir die Koordinaten diskretisieren. &rarr; *selected_bird_species_with_grids.csv*
