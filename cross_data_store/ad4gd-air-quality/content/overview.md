A high resolution dataset for particulate matter (PM2.5 and PM10) generated from low cost sensor (LCS) measurements, in the framework of the Horizon Europe [All Data for Green Deal (AD4GD)](http://ad4gd.eu/) project.

For this purpose, observations from the [sensor.community project](sensor.community) were combined with meteorological data from [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=overview) and air quality data from [CAMS Europe model](https://ads.atmosphere.copernicus.eu/datasets/cams-europe-air-quality-forecasts?tab=overview). To enhance the trustworthiness of the low cost sensor measurements, a machine learning model was trained against EEA reference measurements solely using meteorological and sensor data as feature input. Ultimately, the corrected data were gridded to a high-resolution regular lat-lon lattice in two ways. 

1. Ordinary Kriging for the wider European region, based solely based on the LCS data. 
2. Universal Kriging just for central Europe, combining the LCS with data from the CAMS European Air Quality model.



