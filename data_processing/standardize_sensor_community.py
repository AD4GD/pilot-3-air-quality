
from pathlib import Path
import pandas as pd
import numpy as np
import xarray as xr
import socket
import argparse
from datetime import datetime
from dateutil import rrule
from tqdm import tqdm
import warnings


class StandardizeData:
    """
    Class to standardize IoT data to hourly data.

    Parameters
    ----------
    date : datetime
        date to process
    sensor : str
        sensor platform to standardize
    inputfn : Path
        path to SDS011 parquet file
    outputfold : Path
        output folder to save netcdf files of PM10 and PM2.5
    """

    def __init__(self,
                 date: datetime,
                 sensor: str,
                 inputfn: Path,
                 outputfold: Path) -> None:

        self.date = date
        self.inputfn = inputfn
        self.sensor = sensor
        self.outputfold = outputfold


    def standard_sds(self, show_progress: bool = False) -> None:
        """
        Standardize SDS011 data to hourly data and save to netcdf files

        Parameters
        ----------
        show_progress : bool, optional
            If True, show progress bar, by default False.

        Returns
        -------
        No returns, but saves standardized data to netcdf file.
        """

        # filter user warnings
        warnings.filterwarnings('ignore', category=UserWarning)

        #  set month of intereset and path to data
        print('Start processing SDS011 data')
        ptfn = self.inputfn

        # read in data and perform basic quality control
        df = pd.read_parquet(ptfn)
        df.P2[df.P1 < df.P2] = np.nan
        df.P1[(df.P1 < 0)] = np.nan
        df.P2[(df.P2 < 0)] = np.nan
        df = df.dropna()

        #  reorganize observations and regrid to hourly data
        statdict = {}
        measdict = {'P1': {}, 'P2': {}}
        uniqlocs = sorted(df.location.unique())
        for uloc in tqdm(uniqlocs, disable=~show_progress):
            seldf = df[df.location == uloc]
            sellat = seldf.lat.iloc[0]
            sellon = seldf.lon.iloc[0]
            statdict[str(uloc)] = {'lat': sellat, 'lon': sellon}

            measdf = seldf.drop(['sensor_id', 'location', 'lat', 'lon'], axis=1)
            measdf = measdf.set_index('timestamp')

            for pm in measdict.keys():
                resamdata = measdf[pm].resample('1h')
                mediandf = resamdata.median()
                meandf = resamdata.mean()
                stddf = resamdata.std()
                per10df = resamdata.quantile(0.1)
                per90df = resamdata.quantile(0.9)
                countdf = resamdata.count()
                # mediandf = measdf[pm].resample('1h').median()
                # meandf = measdf[pm].resample('1h').mean()
                # stddf = measdf[pm].resample('1h').std()
                # per10df = measdf[pm].resample('1h').quantile(0.1)
                # per90df = measdf[pm].resample('1h').quantile(0.9)
                # countdf = measdf[pm].resample('1h').count()

                datadict = {'hour_median': mediandf, 'hour_mean': meandf,
                            'hour_std': stddf, 'hour_per10': per10df,
                            'hour_per90': per90df, 'num_obs': countdf}
                combdf = pd.DataFrame(datadict)

                measdict[pm][str(uloc)] = xr.Dataset(combdf)


        # convert station metadata to DataFrame
        statdf = pd.DataFrame.from_dict(statdict, orient='index')
        statdf.index.name = 'location'
        statdf.index = statdf.index.astype(str)

        # convert to xarray dataset
        for pm, pmdict in measdict.items():
            pmds = xr.concat(pmdict.values(), dim='location')
            pmds['location'] = list(pmdict.keys())
            pmds = pmds.rename({'timestamp': 'time'})
            pmds = pmds.astype(np.float32)
            pmds['num_obs'] = pmds['num_obs'].astype(np.int16)
            pmds['lat'] = statdf['lat']
            pmds['lon'] = statdf['lon']

            # save data to netcdf
            pmname = 'PM10' if pm == 'P1' else 'PM2.5'
            ncfn = Path(self.outputfold, f'SDS011_{pmname}_hourly_{self.date:%Y%m%d}.nc')
            pmds.to_netcdf(ncfn)


    def run(self, show_progress=False) -> None:
        """
        Run standardization procedure of IoT data to hourly data

        Parameters
        ----------
        show_progress : bool, optional
            If True, show progress bar, by default False.
        """

        #  set paths
        if not self.outputfold.exists():
            self.outputfold.mkdir(parents=True)

        if self.sensor == 'sds011':
            self.standard_sds()
        else:
            raise ValueError('Unknown sensor')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor',
                        choices=['sds011', 'bme280', 'dht22'],
                        help='sensor platform to standardize',
                        required=True)
    parser.add_argument('-d', '--date',
                        type=lambda x: datetime.strptime(x, '%Y%m%d'),
                        help='date of data to process (YYYYMMDD)')
    parser.add_argument('-i', '--inputfn',
                        type=lambda x: Path(x),
                        help='path to input file')
    parser.add_argument('-o', '--outputfold',
                        type=lambda x: Path(x),
                        help='path to output folder')
    args = parser.parse_args()

    Standard = StandardizeData(args.date,
                               args.sensor,
                               args.inputfn,
                               args.outputfold)
    Standard.run()