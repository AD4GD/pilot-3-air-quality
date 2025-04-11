<p align="center">
<img alt="image" src="https://github.com/user-attachments/assets/4a215329-7835-4d37-aeb6-92141f14b6b5">
</p>


# AD4GD Pilot 3: Air Quality

**Pilot 3** focuses on leveraging IoT sensor data to monitor air quality, with an emphasis on understanding the impact of air pollution on health. This study zeroes in on regions like Benelux and Northern Italy's Po Valley—areas with distinct pollution profiles. By analyzing data from low-cost sensors (LCS) and Citizen Science contributions, we aim to enhance air quality monitoring, identify pollution hotspots, and provide real-time insights with high spatial resolution. Ultimately, this pilot will explore how IoT data can complement traditional monitoring methods, support public health policy, and improve air quality forecasts.

## What's in here
```
.
├── README.md <--- You are here.
├── assets // Images and other assets.
├── cross_data_store // Metadata for pilot 3 dataset on xds.ecmwf.int
├── demos // Demo notebooks showing how to access and use the data
└── rainbow // Semantic Metadata for bringing the Pilot 3 datasets into RAINBOW
```

## Demos
[Download and visualise the air quality dataset](demos/download_air_quality_data.ipynb)

## Dataset

A high resolution dataset for particulate matter (PM2.5 and PM10) generated from low cost sensor (LCS) measurements, in the framework of the Horizon Europe [All Data for Green Deal (AD4GD)](http://ad4gd.eu/) project.

For this purpose, observations from the [sensor.community project](sensor.community) were combined with meteorological data from [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) and air quality data from [CAMS Europe model](https://ads.atmosphere.copernicus.eu/datasets/cams-europe-air-quality-forecasts?tab=overview). To enhance the trustworthiness of the low cost sensor measurements, a machine learning model was trained against EEA reference measurements solely using meteorological and sensor data as feature input. Ultimately, the corrected data were gridded to a high-resolution regular lat-lon lattice in two ways. 

1. Ordinary Kriging for the wider European region, based solely based on the LCS data. 
2. Universal Kriging just for central Europe, combining the LCS with data from the CAMS European Air Quality model.

### Variables

Particulate Matter (<2.5µm) and Particulate Matter (<10µm)

The mass concentration of fine particulate matter (particles with a aerodynamic diameter of 2.5 micrometers or less) present in the ambient air. Values are derived from a mixture of data from low cost sensors and and the CAMS Europe Air Quality Forecast. Units: µg m^-3
