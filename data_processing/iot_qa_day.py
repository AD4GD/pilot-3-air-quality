# -*- coding: utf-8 -*-

from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import xgboost as xgb
from sklearn.neighbors import BallTree
from numba import njit
from datetime import datetime
from tqdm import tqdm
from argparse import ArgumentParser
from dateutil import rrule, relativedelta
from scipy import ndimage, signal
from scipy.spatial import KDTree
import joblib

import warnings
warnings.filterwarnings("ignore")


vardict = {'pm10': {'camsname': 'pm10_conc',
                    'eeaname': 'PM10_Concentration'
                    },
        'pm25': {'camsname': 'pm2p5_conc',
                 'eeaname': 'PM2.5_Concentration',
                 },
        'NO2': {'camsname': 'no2_conc',
                'eeaname': 'NO2_Concentration',
                'outname': 'no2'},
        'O3': {'camsname': 'o3_conc',
                'eeaname': 'O3_Concentration',
                'outname': 'o3'}
            }


@njit('float64[:](float64[:,:], float64)')
def nan_percentile_numba(arr, p):
    """
    Calculate the percentile of an array ignoring NaN values.
    Original nan percentile function of numpy is too slow and inefficient.

    Parameters
    ----------
    arr : 2D array
        Array containing the data
    p : float
        Percentile value

    Returns
    -------
    percentiles : 1D array
        Array containing the percentile values
    """

    # Pre-allocate the output array with NaNs
    # (assuming the worst case where all are NaN)
    percentiles = np.full(arr.shape[1], np.nan, dtype=np.float64)

    for i in range(arr.shape[1]):
        # Extract the column
        col = arr[:, i]
        # Filter out NaNs from the column
        valid_values = col[~np.isnan(col)]

        if valid_values.size > 0:  # Check if there are any non-NaN values
            # Sort the non-NaN values
            percentiles[i] = np.percentile(valid_values, p)

    return percentiles


# Function to convert lat-lon to geocentric (Cartesian) coordinates
def latlon_to_geocentric(lat, lon):
    """
    Convert latitude and longitude to geocentric (Cartesian) coordinates.

    Parameters
    ----------
    lat : np.array
        Latitude in degrees.
    lon : np.array
        Longitude in degrees.

    Returns
    -------
    coords : np.array
        Geocentric coordinates.
    """

    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    x = np.cos(lat) * np.cos(lon)
    y = np.cos(lat) * np.sin(lon)
    z = np.sin(lat)

    coords = np.vstack([x, y, z]).T
    return coords


class Corrector:
    """
    Class correcting Sensor.Community data,
    combining it with already corrected PurpleAir PM2.5 data,
    and filtering for outliers using CAMS data.

    Parameters
    ----------
    month : datetime
        Month for which to run the process
    pollutant : str
        Pollutant for which to run the process
    sam_num : int
        Number of cluster size for Sensor.Community data
    eea_num : int
        Number of EEA stations in IoT clusters data
    outputfolder : pathlib.Path
        Folder where to save the combined dataset.
    """

    def __init__(self,
                 scomfn: Path,
                 month: datetime,
                 pollutant: str,
                 meteofn: Path,
                 camsfn: Path,
                 outputfolder: Path=Path.home()):

        self.scomfn = scomfn
        self.month = month
        self.poll = pollutant
        self.meteofn = meteofn
        self.camsfn = camsfn
        self.outputfolder = outputfolder

        self.iotvar = 'hour_median'
        self.distance_thres = 2

        scriptdir = Path(__file__).resolve().parent
        self.xgbfold = Path(scriptdir, 'input_for_correction', 'daily')

        self.xgbfn = Path(self.xgbfold, f'xgb_day_{self.poll}.json')
        self.keepfn = Path(self.xgbfold, f'kept_clusters_day_{self.poll}.csv')
        self.labfn = Path(self.xgbfold, f'iot_{self.poll}_labels.csv')
        self.centerfn = Path(self.xgbfold, f'iot_{self.poll}_cluster_coordinates.csv')

        self.outfold = Path(outputfolder, pollutant)


    def correct_sensor_community(self,
                                 scomds: xr.Dataset,
                                 erads: xr.Dataset,
                                 thres_hour_obs: int=10,
                                 max_thres: float=999.9) -> xr.Dataset:
        """
        Function to correct Sensor.Community data using XGBoost model and ERA5 data

        Parameters
        ----------
        scomds : xarray dataset
            Sensor.Community dataset
        erads : xarray dataset
            ERA5 dataset containing daily mean meteorological data
        camsds : xarray dataset
            CAMS Europe dataset containing daily mean pollutant data
        thres_hour_obs : int, optional
            Minimum number of hourly observations for correction. The default is 20.
        max_thres : float, optional
            Maximum threshold for pollutant values. The default is 999.9.

        Returns
        -------
        scomds : xarray dataset
            Sensor.Community dataset containing corrected PM data
        """

        iotpolname = f'iot_{self.poll}'

        # load XGBoost model and read input data of XGBoost training to get unique cluster categories
        xgbmodel = xgb.XGBRegressor(enable_categorical=True)
        xgbmodel.load_model(self.xgbfn)

        centerdf = pd.read_csv(self.centerfn, sep=';', index_col=0)
        keepdf = pd.read_csv(self.keepfn, sep=';', index_col=0)
        xgb_labels = list(keepdf.index)
        xgbdict = {}
        for idx in keepdf.index:
            xgblat = centerdf['cluster_centroids_lat'].loc[idx]
            xgblon = centerdf['cluster_centroids_lon'].loc[idx]
            xgb_xyz = latlon_to_geocentric(xgblat, xgblon)
            xgbdict[idx] = {'xyz': xgb_xyz}

        xgb_categories = pd.CategoricalDtype(categories=xgb_labels)


        # load label and coordinate data
        print('Loading label information...')
        labdf = pd.read_csv(self.labfn, sep=';', index_col=0)

        # get coordinates of Sensor.Community stations
        iotlat = scomds['lat'].values
        iotlon = scomds['lon'].values


        # read information from clustered/labeled Sensor.Community stations
        cluster_labels = labdf['cluster']
        lab_lat = labdf['Latitude']
        lab_lon = labdf['Longitude']
        unique_labels = np.unique(cluster_labels)

        # get coordinates of clustered Sensor.Community stations
        labX = latlon_to_geocentric(lab_lat, lab_lon)
        kdtree = KDTree(labX)
        geoX = latlon_to_geocentric(iotlat, iotlon)
        dist, ind = kdtree.query(geoX, k=1)

        # distance_threshold = 2 # km
        earth_radius = 6371.0
        distkm = 2 * earth_radius * np.arcsin(dist / 2)
        dist_sel = distkm <= self.distance_thres

        iotlabel = np.full(iotlat.shape, -1)
        iotlabel[dist_sel] = cluster_labels[ind][dist_sel]
        scomds['cluster'] = ('location', iotlabel)

        # interpolate ERA5 data to Sensor.Community stations
        interds = erads.interp(latitude=scomds.lat, longitude=scomds.lon)
        interds = interds.rename({v: f'iot_{v}' for v in interds.data_vars})

        # add day of year to features and expand it to size of other features
        interds['doy'] = (('time', 'location'), np.tile(interds.time.dt.dayofyear, (len(interds.location), 1)).T)

        # set up feature input for XGBoost model
        doy = interds['time.dayofyear'].values
        interds['cos_doy'] = np.cos(2*np.pi*interds['doy'] / 366)
        interds['sin_doy'] = np.sin(2*np.pi*interds['doy'] / 366)
        interds['dayofyear'] = interds['doy']

        # create flag for new years eve
        # add new year eve
        interds['new_year'] = ((interds['time'].dt.month == 12) &
                          (interds['time'].dt.day == 31)) | \
                         ((interds['time'].dt.month == 1) &
                         (interds['time'].dt.day == 1))
        interds['new_year'] = interds['new_year'].astype(int)

        # prefilter IoT data
        iotds = scomds.where(scomds['num_obs'] > thres_hour_obs, drop=True)
        iotds = iotds.where(iotds[self.iotvar] < max_thres, drop=True)
        iotds = iotds.where(iotds[self.iotvar] > 1, drop=True)

        resam_iotds = iotds.resample(time='1D').mean()
        resam_count = iotds.resample(time='1D').count()
        resam_iotds = resam_iotds.where(resam_count.hour_median >= 12, drop=True)
        resam_iotds['lat'] = scomds['lat'].sel(location=resam_iotds.location)
        resam_iotds['lon'] = scomds['lon'].sel(location=resam_iotds.location)
        resam_iotds['cluster'] = scomds['cluster'].sel(location=resam_iotds.location)

        # collect data in dict of dataframes for output
        outdict = {k: {} for k in ['day_corr', 'ens_bool', 'ens_std']}

        # check if IoT station is in cluster
        loc_cluster = resam_iotds['location'][resam_iotds['cluster'].isin(xgb_labels)]
        no_loc_cluster = resam_iotds['location'][~resam_iotds['cluster'].isin(xgb_labels)]

        print('Correcting data for clustered IoT stations...')
        for location in tqdm(loc_cluster.values, disable=False):
            lociot = resam_iotds.sel(location=location)
            locera = interds.sel(location=location)

            # prepare ERA5 feature input for XGBoost model
            # era_tmpdf = locera.to_dataframe()
            era_tmpdf = locera.to_pandas()
            era_tmpdf = era_tmpdf.drop(columns=['location', 'latitude', 'longitude', 'doy'])

            clusterval = lociot['cluster'].values

            iot_tmpdf = lociot[['hour_median']].to_pandas()
            iot_tmpdf = iot_tmpdf.drop(columns=['location'])
            iot_tmpdf = iot_tmpdf.rename(columns={'hour_median': iotpolname})

            # merge feature inputs
            featdf = pd.concat([iot_tmpdf, era_tmpdf], axis=1)
            featdf = featdf.dropna()
            featdf['cluster'] = clusterval
            featdf['cluster'] = featdf['cluster'].astype(xgb_categories)

            # predict correction factor for IoT data
            inputdf = featdf[xgbmodel.feature_names_in_]
            iot_corrected = xgbmodel.predict(inputdf)

            ens_mean = pd.Series(iot_corrected, index=featdf.index)
            ens_std = pd.Series(np.full(ens_mean.shape, np.nan), index=ens_mean.index)
            ens_bool = False

            # add corrected data to output dictionary
            outdict['day_corr'][location] = ens_mean
            outdict['ens_bool'][location] = ens_bool
            outdict['ens_std'][location] = ens_std.astype(np.float32)

        daycorr = pd.DataFrame(outdict['day_corr']).T
        ens_std = pd.DataFrame(outdict['ens_std']).T
        ens_bool = pd.Series(outdict['ens_bool']).astype(np.int8).T
        daycorr.index.name = 'location'
        ens_std.index.name = 'location'
        ens_bool.index.name = 'location'


        # now correct Sensor.Community stations which are not in cluster
        nlc_iot = resam_iotds.sel(location=no_loc_cluster)
        nlc_era = interds.sel(location=no_loc_cluster)

        nlc_era = nlc_era.to_dataframe()
        nlc_era = nlc_era.drop(columns=['latitude', 'longitude', 'doy'])

        # prepare IoT data for XGBoost model
        iot_tmpdf = nlc_iot['hour_median'].to_dataframe().rename(columns={'hour_median': iotpolname})

        # merge feature inputs
        featdf = pd.concat([iot_tmpdf, nlc_era], axis=1)
        nona = featdf.dropna()

        # calculate geocentric coordinates for IoT station
        tmp_loclat = nlc_iot['lat']
        tmp_loclon = nlc_iot['lon']
        loc_xyz = latlon_to_geocentric(tmp_loclat, tmp_loclon)

        # for stations without cluster, calculate correction factor from each cluster
        print('Correcting data for non-clustered IoT stations...')
        clusdict = {}
        for k, cluster in tqdm(enumerate(xgb_labels), total=len(xgb_labels), disable=False):

            tmp_featdf = nona.copy()
            tmp_featdf['cluster'] = cluster
            tmp_featdf['cluster'] = tmp_featdf['cluster'].astype(xgb_categories)
            tmp_xgbxyz = xgbdict[cluster]['xyz']
            inputdf = tmp_featdf[xgbmodel.feature_names_in_]

            # predict correction factor for IoT data
            outdf = xgbmodel.predict(inputdf)
            inputdf['corrected'] = outdf

            # convert to xarray dataset
            clusda = inputdf['corrected'].to_xarray()
            clusds = clusda.to_dataset(name='iot_corrected')
            xgb_dist = np.linalg.norm(loc_xyz - tmp_xgbxyz, axis=1)
            arcdist = 2*np.arcsin(xgb_dist/2)

            # add distance to xarray dataset
            clusds['distance'] = xr.DataArray(arcdist, dims='location')
            clusdict[cluster] = clusds

        # merge output datasets
        conds = xr.concat(clusdict.values(), dim='cluster')
        conds['cluster'] = list(clusdict.keys())

        # fill distance variable along time dimension with duplicated values
        filldistance = np.tile(conds['distance'].values, (len(conds.time), 1, 1))
        filldistance = filldistance.transpose(1, 0, 2)
        conds['distance'] = xr.DataArray(filldistance, dims=['cluster', 'time', 'location'])

        # calculate weights for each cluster
        sqdist = conds['distance']**2
        weight = 1 / sqdist
        weight = weight.where(~np.isnan(conds['iot_corrected']))
        sumweight = weight.sum(dim='cluster', skipna=True)
        newweight = weight / sumweight

        weighted_corr = (newweight * conds['iot_corrected']).sum(dim='cluster')
        weighted_corr = weighted_corr.where(~newweight.isnull().all(dim='cluster'))
        nlc_daycorr = weighted_corr
        nlc_ens_std = conds['iot_corrected'].std(dim='cluster')
        nlc_ens_bool = xr.full_like(nlc_daycorr, True, dtype=bool)

        # merge clustered and non-clustered data
        corrected_data = xr.concat([xr.DataArray(daycorr), nlc_daycorr], 'location')
        corrected_ens_std = xr.concat([xr.DataArray(ens_std), nlc_ens_std], 'location')
        corrected_ens_bool = xr.concat([xr.DataArray(ens_bool), nlc_ens_bool], 'location').astype(np.int8)

        # add corrected data to dataset
        resam_iotds['day_corr'] = corrected_data
        resam_iotds['ens_std'] = corrected_ens_std
        resam_iotds['ens_bool'] = corrected_ens_bool

        # convert dtypes
        resam_iotds['day_corr'] = resam_iotds['day_corr'].astype(np.float32)
        resam_iotds['ens_bool'] = resam_iotds['ens_bool'].astype(bool)
        resam_iotds['cluster'] = resam_iotds['cluster'].astype('int32')
        resam_iotds['cluster'].attrs['xgb_labels'] = xgb_labels
        resam_iotds['cluster'].attrs['cluster_labels'] = unique_labels

        return resam_iotds


    def outlier_filtering(self,
                          combids: xr.Dataset,
                          hour_polds: xr.Dataset,
                          radii: List[float]=[50, 25, 10, 5]) -> xr.DataArray:
        """
        Perform data driven outlier filtering for IoT data using CAMS data and
        surrounding IoT stations

        Parameters
        ----------
        combids : xarray dataset
            Dataset containing (corrected) IoT data
        hour_polds : xarray dataset
            CAMS Europe dataset
        radii : list, optional
            List of radii in km for filtering. The default is [50, 25, 10, 5].

        Returns
        -------
        boolda : xarray dataarray
            Dataarray containing boolean values for filtering IoT data for
            different radii
        """


        # define maximum radius for filtering
        max_radius = max(radii)

        # calculate distance between IoT stations
        coor = np.column_stack((combids.lat, combids.lon))
        coor = np.deg2rad(coor)
        tree = BallTree(coor, metric='haversine')
        queryidx, querydist = tree.query_radius(coor, r=max_radius/6371,
                                                return_distance=True)

        # get CAMS data at IoT locations
        interpol = hour_polds.interp(latitude=combids.lat, longitude=combids.lon,
                                    method='nearest')

        daypol = interpol.resample(time='1D').mean()

        # now calculate alpha values at each IoT location
        alpha = np.log(combids['day_corr'] / daypol)

        # create empty array to store number of IoT stations in CAMS grid cell
        boolshape = (len(radii), len(combids.location), len(combids.time))
        boolarr = np.full(boolshape, False, dtype=bool)
        for statidx, (qdist, areaidx) in tqdm(enumerate(zip(querydist, queryidx)),
                                            total=len(queryidx), disable=False):

            # skip if less than 10 IoT stations in radius
            if len(qdist) < 10:
                continue

            # convert distance to km
            realdist = qdist * 6371.0

            # select pollutant and ratio values at IoT station
            loc_alpha = alpha.isel(location=statidx)
            loc_pol = combids['day_corr'].isel(location=statidx)

            # select ratio values at surrounding IoT stations
            area_alpha = alpha.isel(location=areaidx)
            areavals = area_alpha.values
            # areavals = areavals.T

            # iterate over radii
            for ridx, radius in enumerate(radii):

                # select surrounding IoT stations within radius
                distsel = realdist <= radius
                if np.count_nonzero(distsel) < 10:
                    continue

                # select ratio values at surrounding IoT stations
                radiusvals = areavals[distsel, :]
                radiusvals = radiusvals.astype(np.float64)

                # calculate statistics for filtering
                p25 = nan_percentile_numba(radiusvals, 25)
                p75 = nan_percentile_numba(radiusvals, 75)

                alpha_iqr = p75 - p25
                upper15 = p75 + 1.5*alpha_iqr
                lower15 = p25 - 1.5*alpha_iqr
                validcount = np.sum(~np.isnan(radiusvals), axis=0)

                # filter data
                boolidx = (loc_alpha <= upper15) & (loc_alpha >= lower15) & (validcount >= 10)

                # store boolean selection
                boolarr[ridx, statidx, :] = boolidx


        # add boolean array to dataset
        boolda = xr.DataArray(boolarr, dims=['radius', 'location', 'time'],
                            coords={'radius': radii, 'location': combids.location,
                            'time': combids.time})

        return boolda


    def prepare_modeldata(self):
        """
        Load meteorological data and CAMS data for the given month

        Parameters
        ----------
        None.

        Returns
        -------
        erads : xarray dataset
            ERA5 dataset containing daily mean meteorological data
        hour_polds : xarray dataset
            CAMS Europe dataset containing hourly pollutant data
        """

        # read ERA5 data
        erads = xr.open_dataset(self.meteofn)

        # now read hourly CAMS data and shift longitudes from 0-360 to -180-180
        hourds = xr.open_dataset(self.camsfn)
        hourds = hourds.squeeze()

        lonname = 'longitude'
        hourds = hourds.assign_coords(longitude=(((hourds[lonname] + 180) % 360) - 180)).sortby(lonname)
        hourds = hourds.sortby('latitude')
        hour_polds = hourds[vardict[self.poll]['camsname']]
        hour_polds = hour_polds.drop('level')
        camstime = pd.to_timedelta(hour_polds.time) + self.month
        hour_polds.coords['time'] = camstime

        return erads, hour_polds


    def run(self):
        """
        Generic function to process and filter IoT data for a specified pollutant.

        Parameters
        ----------
        None.

        Returns
        -------
        None, saves the combined dataset to a netCDF file.
        """

        max_thres = 999.9
        thres_hour_obs = 20

        # Load Sensor.Community data
        print(f'Loading Sensor.Community data for {self.poll}...')
        scomds = xr.open_dataset(self.scomfn)

        # Load meteorological and CAMS data
        print('Loading meteorological and CAMS data...')
        erads, hour_polds = self.prepare_modeldata()

        # Filter for IoT stations in CAMS Europe domain
        domain_filter = (scomds.lat >= 30) & \
                        (scomds.lat <= 72) & \
                        (scomds.lon >= -25) & \
                        (scomds.lon <= 45)
        scomds = scomds.where(domain_filter, drop=True)

        # Correct Sensor.Community data
        print('Correcting Sensor.Community data...')
        resam_scomds = self.correct_sensor_community(scomds,
                                                     erads,
                                                     thres_hour_obs,
                                                     max_thres)

        # rename location names
        resam_scomds['location'] = [f'SC{loc}' for loc in resam_scomds.location.values]

        # add already corrected PurpleAir data for PM2.5
        combids = resam_scomds

        # Filter for outliers
        print('Filtering for outliers...')
        boolda = self.outlier_filtering(combids, hour_polds)
        combids['day_corr_distflag'] = boolda

        combids = combids.rename({'hour_median': 'day_median',
                                  'day_corr': 'day_median_corr'})
        combids = combids.drop_vars(['hour_mean', 'hour_per10', 'hour_per90', 'hour_std'])

        # Save combined dataset
        outname = f'iot_day_corr_{self.poll}_{self.month:%Y%m}.nc'
        if not self.outfold.exists():
            self.outfold.mkdir(parents=True)

        outfn = self.outfold / outname
        combids.to_netcdf(outfn)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-m', '--month',
                        type=lambda s: datetime.strptime(s, '%Y%m'),
                        help='Month for which to run the process')
    parser.add_argument('-p', '--pollutant', choices=['pm10', 'pm25'],
                        type=str,
                        help='Pollutant for which to run the process')
    parser.add_argument('-s', '--scomfn', type=Path,
                        help='Sensor.Community data file')
    parser.add_argument('-m', '--meteofn', type=Path,
                        help='Meteorological data file')
    parser.add_argument('-c', '--camsfn', type=Path,
                        help='CAMS data file')
    parser.add_argument('-o', '--outputfolder', type=Path,
                        help='Output folder for corrected data')

    args = parser.parse_args()

    Corr = Corrector(args.scomfn,
                     args.month,
                     args.pollutant,
                     args.meteofn,
                     args.camsfn,
                     args.outputfolder)
    Corr.run()


