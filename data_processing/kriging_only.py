
import numpy as np
from pathlib import Path
from datetime import datetime
import xarray as xr
from tqdm import tqdm
import pandas as pd
import torch
from scipy.interpolate import RegularGridInterpolator as RGI
from sklearn import cluster
from sklearn.neighbors import BallTree
from scipy.spatial import KDTree
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_processing.utils import latlon_to_geocentric
from data_processing import variogram
from data_processing.pok import run_kriging
from typing import Literal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class KrigingIoT:
    """
    Class to perform penalized ordinary kriging on IoT data.

    Parameters
    ----------
    timeliness : str
        Timeliness of the data, either 'hourly' or 'daily' mean IoT data.
    date : datetime
        Date to process.
    iotmeasfn : Path
        Path to IoT measurements file.
    outfn : Path
        Path to save results to.
    no_filter : bool
        Whether to not filter outliers in the IoT data.
        Default is False.
    tree_model : str
        Model to use for nearest neighbor search, either 'kdtree' or 'balltree'.
        Default is 'kdtree'. BallTree is eventually more accurate but much slower.
    """

    def __init__(self,
                 timeliness: Literal['hourly', 'daily'],
                 date: datetime,
                 iotmeasfn: Path,
                 outfn: Path,
                 no_filter: bool=False,
                 tree_model: str='kdtree'):

        self.date = date
        self.tness = timeliness
        self.iotmeasfn = iotmeasfn
        self.outfn = outfn
        self.no_filter = no_filter
        self.tree_model = tree_model

        # spatial domains of interest
        areadict = {
                    'CentralEurope': {'latmin': 46.5, 'latmax': 55.2,
                                    'lonmin': 2.0, 'lonmax': 15.5,
                                    'ds': 0.01},
                    'CAMS': {'latmin': 35, 'latmax': 72,
                            'lonmin': -25, 'lonmax': 45,
                            'ds': 0.01},
                    }

        self.area = 'CAMS'  # set area of interest
        self.cdict = areadict[self.area]

        if self.tness == 'hourly':
            self.measvar = 'hour_median_corr'
            self.qavar = 'hour_median_corr_distflag'
        elif self.tness == 'daily':
            self.measvar = 'day_corr'
            self.qavar = 'day_corr_distflag'
        else:
            raise ValueError('Unknown timeliness')

        # define land sea mask file
        scriptdir = Path(__file__).resolve().parent
        self.lsmfn = Path(scriptdir, 'input_for_correction', 'gmt_lsm.nc')

    def run(self):
        """
        Main function to perform regridding of IoT data to given regular
        lat-lon grid using ordinary kriging on global and local scale.

        Parameters
        ----------
        No parameters, but uses class attributes.

        Returns
        -------
        No returns, but saves results to netCDF file.
        """

        # set up grid for kriging
        latmin = self.cdict['latmin']
        latmax = self.cdict['latmax']
        lonmin = self.cdict['lonmin']
        lonmax = self.cdict['lonmax']
        ds = self.cdict['ds']

        # read land sea mask
        lsmds = xr.open_dataset(self.lsmfn)
        lsm = lsmds.z.values > 0
        lsmfunc = RGI((lsmds.lat.values, lsmds.lon.values), lsm, method='nearest',
                    bounds_error=False)

        gridlon = np.arange(lonmin, lonmax, ds) + ds/2
        gridlat = np.arange(latmin, latmax, ds) + ds/2

        # set up mesh grid
        gX, gY = np.meshgrid(gridlon, gridlat, indexing='ij')
        gX_flat = gX.ravel()
        gY_flat = gY.ravel()

        # check which points are on land
        lsm_flat = lsmfunc((gY_flat, gX_flat))
        lsm_sel = np.where(lsm_flat)[0]

        # select kriging points on land
        kriglon = gX_flat[lsm_sel]
        kriglat = gY_flat[lsm_sel]


        # read IoT PM data set
        print(datetime.now(), 'Reading IoT data')
        measds = xr.open_dataset(self.iotmeasfn)

        # only use data within spatial domain
        combids = measds.where((measds.lat >= latmin) &
                                (measds.lat <= latmax) &
                                (measds.lon >= lonmin) &
                                (measds.lon <= lonmax),
                                drop=True)

        # select IoT data
        if self.no_filter:
            iotdata = combids[self.measvar]
        else:
            qaflag = combids[self.qavar].isel(radius=1)
            iotdata = combids[self.measvar].where(qaflag)

        # select data for kriging
        if self.tness == 'hourly':
            day = self.date.replace(hour=0, minute=0, second=0, microsecond=0)
            starttime = pd.Timestamp(day)
            endtime = pd.Timestamp(day) + pd.Timedelta(hours=23, minutes=59)
            seliotdata = iotdata.sel(time=slice(starttime, endtime))
        else:
            seliotdata = iotdata

        if len(seliotdata.time) == 0:
            raise ValueError('No data for this day')


        colds = []
        coltime = []
        for itime, timeval in enumerate(seliotdata.time.values):

            print(datetime.now(), 'Processing', timeval)

            # create dataframe for semivariogram
            timeda = seliotdata.isel(time=itime)
            df = timeda.to_dataframe()
            for dropcol in ['radius', 'time']:
                try:
                    df = df.drop(columns=[dropcol])
                except KeyError:
                    pass

            df = df.dropna()
            df['lat'] = combids.lat.sel(location=df.index).values
            df['lon'] = combids.lon.sel(location=df.index).values
            df = df.rename(columns={self.measvar: 'measurement'})
            df = df.astype('float32')
            df = df.where(df['measurement'] > 0).dropna()

            if len(df) == 0:
                print('No data for this time step')
                continue

            # get data for kriging
            iotlat = df.lat.values
            iotlon = df.lon.values
            iotvals = np.log(df.measurement.values)
            iotvals = iotvals.astype('float32')


            # identify clusters in IoT data for local kriging
            print(datetime.now(), 'Identifying clusters')
            X = np.vstack([iotlat, iotlon]).T
            try:
                optics = cluster.OPTICS(min_samples=100,
                                        algorithm='ball_tree',
                                        xi=0.05)
                optics.fit(np.deg2rad(X))
                labs = optics.labels_
            except (ValueError, RuntimeError) as e:
                print(e)
                print('Problems with finding clusters, skipping this time step')
                continue

            # check distance to nearest IoT station
            print(datetime.now(), 'Checking distance to nearest IoT station')
            distance_threshold = 10 # km
            earth_radius = 6371.0

            if self.tree_model == 'balltree':

                krigX = np.vstack([kriglat, kriglon]).T.astype('float32')
                radkrigX = np.deg2rad(krigX)
                print(datetime.now(), 'Create and query BallTree')

                # BallTree requires latitude as first and longitude as second coordinate
                X = np.vstack([iotlat, iotlon]).T
                balltree = BallTree(np.deg2rad(X),
                                    metric='haversine',
                                    leaf_size=15)
                dist, ind = balltree.query(radkrigX, k=1)
                dist = dist.ravel()
                ind = ind.ravel()

                print(datetime.now(), 'Calculate distances')
                distkm = dist * earth_radius
                dist_sel = distkm <= distance_threshold

            elif self.tree_model == 'kdtree':

                print(datetime.now(), 'Create and query KDTree')
                iotX = latlon_to_geocentric(iotlat, iotlon)
                kdtree = KDTree(iotX)
                geoX = latlon_to_geocentric(kriglat, kriglon)
                dist, ind = kdtree.query(geoX, k=1)

                print(datetime.now(), 'Calculate distances')
                distkm = 2 * earth_radius * np.arcsin(dist / 2)
                dist_sel = distkm <= distance_threshold

            else:
                raise ValueError('Unknown tree model')


            # set up latitude and longitude to krige to
            finkriglon = kriglon[dist_sel]
            finkriglat = kriglat[dist_sel]

            # perform ordinary kriging on all data points and for complete domain
            print(datetime.now(), 'Performing global kriging')
            logscale = True
            custom_lags = np.arange(0, 25, 2, dtype='float32')
            ok_val, ok_var = run_kriging(iotlon, iotlat, iotvals,
                                        finkriglon, finkriglat,
                                        custom_lags, penalty=1e-6, device=device,
                                        plotting=False, logscale=logscale)

            # perform local kriging
            print(datetime.now(), 'Performing local kriging')
            interlabel = labs[ind][dist_sel].ravel()
            distlab_threshold = 15
            dist_labsel = distkm[dist_sel] > distlab_threshold
            interlabel[dist_labsel] = -1

            interdata = np.full_like(finkriglon, np.nan, dtype='float32')
            intervar = np.full_like(finkriglon, np.nan, dtype='float32')
            for i in tqdm(range(0, labs.max()+1), disable=True):

                isel = labs == i

                # calculate semivariogram
                # use robust estimator for semivariogram (see Eq. 2.4.12 in Cressie (1993))
                sellon = iotlon[isel]
                sellat = iotlat[isel]
                selvals = iotvals[isel]

                # calculate average distances
                avgdist, mediandist, p75dist, maxdist = variogram.calc_avg_distance(sellat, sellon)

                # calculate semivariogram
                krigmax = np.round(0.5*maxdist)
                incdist = mediandist if mediandist != 0 else avgdist
                kriginc = np.round(incdist/4, 2)
                kriglags = np.arange(0, krigmax+kriginc/2, kriginc, dtype='float32')

                [semi, semicount,
                semilags] = variogram.semivariogram_robust(sellat, sellon,
                                                        selvals, kriglags)
                selidx = np.isnan(semi) | (semicount < 10)

                # calculate variogram parameters
                varmodel = 'spherical'
                [popt, pcov,
                infodict] = variogram.calc_variogram_params(semilags[~selidx],
                                                            semi[~selidx],
                                                            model=varmodel)

                # perform kriging, select points that are part of cluster
                krigsel = interlabel == i
                tmp_kriglon = finkriglon[krigsel]
                tmp_kriglat = finkriglat[krigsel]

                # create kriging matrix
                n = len(selvals)
                A = np.zeros((n+1, n+1), dtype='float32')
                for k in range(n):
                    tmplat = sellat[k]
                    tmplon = sellon[k]
                    tmpdist = variogram.haversine_distance_single(tmplat, tmplon,
                                                                sellat[k:], sellon[k:])

                    semivar = -1 * variogram.spherical_varmodel(tmpdist, *popt)

                    A[k, k:n] = semivar.T
                    A[k:n, k] = semivar.T

                np.fill_diagonal(A, 1e-6)
                A[-1, :-1] = 1.0
                A[:-1, -1] = 1.0
                A[-1, -1] = 0.0

                # calculate kriging weights
                cdist = variogram.haversine_distance_numpy(sellat, sellon,
                                                        tmp_kriglat, tmp_kriglon)

                # setup right hand side
                b = np.zeros((cdist.shape[0], n + 1))
                b[:, -1] = 1.0
                b[:, :-1] = -1 * variogram.spherical_varmodel(cdist, *popt)

                # solve kriging system
                weights = np.linalg.solve(A, b.T)

                # Calculate kriging estimate
                krigval = (weights[:n, :].T * selvals).sum(axis=1)

                # Calculate uncertainty of estimate
                krigvar = (weights * -b.T).sum(axis=0)

                # convert back to original scale
                if logscale:
                    mY = weights[-1, :]
                    finval = np.exp(krigval + krigvar/2 + mY)
                    finvar = (np.exp(krigvar)-1) * np.exp(2*krigval + krigvar)
                else:
                    finval = krigval
                    finvar = krigvar

                interdata[krigsel] = finval
                intervar[krigsel] = finvar


            # merge local kriging results with global kriging
            nansel = ~np.isnan(interdata)
            expdata = ok_val.copy()
            expweight = np.exp(-distkm[dist_sel])
            expweight[distkm[dist_sel] > 15] = 0
            expdata[nansel] = expweight[nansel]*interdata[nansel] + (1-expweight[nansel])*ok_val[nansel]

            # now reshape the data
            expkrigdata = np.full(len(gX_flat), np.nan, dtype=np.float32)
            sel = lsm_sel[dist_sel]
            expkrigdata[sel] = expdata
            expkrigdata = expkrigdata.reshape(gX.shape)

            # save data
            print(datetime.now(), 'Move data into xarray data set')
            outdict = {'global_kriging': ok_val,
                    'local_kriging': interdata,
                    'combined_kriging': expdata,
                    }

            outds = xr.Dataset()
            for dataname, data in outdict.items():
                tmp = np.full(len(gX_flat), np.nan, dtype=np.float32)
                tmp[sel] = data
                tmp = tmp.reshape(gX.shape)
                outda = xr.DataArray(tmp, dims=('lon', 'lat'),
                                    coords={'lat': gridlat, 'lon': gridlon})
                outds[dataname] = outda

            colds.append(outds)
            coltime.append(timeval)

        # merge data
        print(datetime.now(), 'Merging and saving data')
        mergeds = xr.concat(colds, dim='time')
        mergeds['time'] = np.array(coltime)
        encodedict = {v: {'zlib': True} for v in outdict.keys()}
        mergeds.to_netcdf(self.outfn, encoding=encodedict)


if __name__ == '__main__':
    pass
