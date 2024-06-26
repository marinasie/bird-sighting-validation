<img src="../../img/hp_readme_header.png" alt="Spatio-Temporal Shifts in Citizen Science Data: Drift Analysis for Bird Sightings" style="width: 100%; border-radius: 20px;"/>
<br><br>

In this subsection, all notebooks related to drift analysis are collected.


#### List of Contents:
- `01_drift_analysis.ipynb` : This notebook will analyze the presence of data drift for all 27 species of interest in order to assess the extent of data drift in the dataset and identify the species and features that are especially affected.
- `02_distribution_drift.ipynb` : The characteristics of distribution drifts and their impact on model performance will be analyzed. The pygmy owl (Sperlingskauz) will serve as a case study, and it is assessed how insights gained for this species can be transferred to other species exhibiting high distribution drift.
- `03_date_drift.ipynb` : The characteristics of date drifts and their impact on model performance will be analyzed. The icterine warbler (Gelbspötter) will serve as a case study, and it is assessed how insights gained for this species can be transferred to other species exhibiting high date drift.
- `04_timeseries_construction.ipynb`: For drift detections, time series will be constructed based on the sighting dataset, utilizing the number of sightings over time for each species and EEA grid.
- `05_continuous_change_point_detection.ipynb` : As an initial approach to drift detection, this notebook will apply rolling window smoothing and KSWIN for change point detection. As this approach showed limitations, alternative methods will be explored in future analyses as part of the Master Thesis.
- `06_time_series_decade_smoothing`: A more accurate preprocessing method for the time series is applied, which includes aggregating sightings in decades and replacing the number of sightings with the ratio of days where sightings occurred.