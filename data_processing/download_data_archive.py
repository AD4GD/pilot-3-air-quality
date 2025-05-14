
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
import socket
from dateutil import rrule
from datetime import datetime
from tqdm import tqdm
import argparse
from typing import Union, List


# function to convert string to float
def conv(x: str) -> float:
    """
    Convert string to float. If string cannot be converted to float,
    return np.nan.

    Parameters
    ----------
    x : str
        string to be converted to float

    Returns
    -------
    a : float
        float value of string
    """
    try:
        a = np.float32(x)
    except (ValueError, TypeError):
        a = np.nan
    return a


# function to read large csv file in chunks and clean up data
def read_large_csv(csvfn: Union[str, Path],
                   dropcols: List[str],
                   floatcols: List[str],
                   intcols: List[str],
                   vecconv: np.vectorize) -> pd.DataFrame:
    """
    Read large csv file in chunks and clean up data.

    Parameters
    ----------
    csvfn : str, Path
        path to csv file
    dropcols : list of str
        list of columns to be dropped from dataframe
    floatcols : list of str
        list of columns to be converted to float
    intcols : list of str
        list of columns to be converted to int
    vecconv : np.vectorize
        vectorized function to convert string to float
    """

    colchunks = []
    chunks = pd.read_csv(csvfn, sep=';', chunksize=1e6, low_memory=False)
    for chunkdf in tqdm(chunks):
        chunkdf = chunkdf.drop(dropcols, axis=1)
        chunkdf['timestamp'] = pd.to_datetime(chunkdf['timestamp'],
                                                format='mixed',
                                                errors='coerce')
        chunkdf[floatcols] = vecconv(chunkdf[floatcols])
        chunkdf[intcols] = chunkdf[intcols].astype(np.int32)
        colchunks.append(chunkdf)

    #  merge chunks of cleaned dataframes
    df = pd.concat(colchunks)
    df = df.dropna()

    return df


def main(sensor: str,
         startdate: datetime,
         enddate: datetime,
         iotpath: str,
         wgetpath: str = '/usr/bin/wget') -> None:
    """
    Download IoT data from sensor.community and save as parquet file.

    Parameters
    ----------
    sensor : str
        sensor type
    startdate : datetime.datetime
        start date of data download period
    enddate : datetime.datetime
        end date of data download period

    Returns
    -------
    No returns, but saves parquet file to disk.
    """

    #  vectorize function
    vecconv = np.vectorize(lambda x: conv(x))

    #  set dictionary for sensor data
    sensordict = {'sds011':
                {'dropcols': ["sensor_type", "durP1", "ratioP1",
                                "durP2", "ratioP2"],
                'floatcols': ['lat', 'lon', 'P1', 'P2'],
                'intcols': ['sensor_id', 'location']
                },
                'bme280':
                {'dropcols': ['altitude', 'pressure_sealevel', 'sensor_type'],
                'floatcols': ['pressure', 'temperature', 'humidity', 'lat', 'lon'],
                'intcols': ['sensor_id', 'location']
                },
                'dht22':
                {'dropcols': ['sensor_type'],
                'floatcols': ['temperature', 'humidity', 'lat', 'lon'],
                'intcols': ['sensor_id', 'location']
                }
                }

    # download IoT data from sensor.community
    archive_url = 'https://archive.sensor.community/csv_per_month'
    for dt in rrule.rrule(rrule.MONTHLY, dtstart=startdate, until=enddate):
        month_url = archive_url +'/' +  f"{dt:%Y-%m}" + '/'

        sensor_fold = Path(iotpath, "L0", sensor, f"{dt:%Y}")
        if not sensor_fold.exists():
            sensor_fold.mkdir(parents=True)

        #  download file from server
        # print('Downloading:', sensor, dt)
        sensor_url = month_url + f'{dt:%Y-%m}_{sensor}.zip'
        cmd = [wgetpath, sensor_url, '-P', str(sensor_fold)]
        check = subprocess.check_call(cmd)

        # unzip file
        print('Unzipping:', sensor, dt)
        rawname = sensor_url.split('/')[-1]
        downfn = Path(sensor_fold, rawname)
        cmd = ['/usr/bin/unzip', str(downfn), '-d', str(sensor_fold)]
        check = subprocess.check_call(cmd)

        # read unzipped file and clean up data
        csvfn = downfn.with_suffix(".csv")
        sdict = sensordict[sensor]
        floatcols = sdict['floatcols']
        intcols = sdict['intcols']
        dropcols = sdict['dropcols']
        df = read_large_csv(csvfn, dropcols, floatcols, intcols, vecconv)

        #  save dataframe as parquet file
        outfold = Path(iotpath, "L1A", sensor)
        if not outfold.exists():
            outfold.mkdir(parents=True)

        outfn = Path(outfold, rawname.replace('.zip', '.parquet'))
        df.to_parquet(outfn, index=False)

        #  delete unzipped csv file
        csvfn.unlink()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--sensor', type=str, help='sensor type',
                        choices=['sds011', 'bme280', 'dht22'], required=True)
    parser.add_argument('-s', '--startdate',
                        type=lambda x: datetime.strptime(x, '%Y%m%d'),
                        help='start date of data download period (YYYYMMDD)')
    parser.add_argument('-e', '--enddate',
                        type=lambda x: datetime.strptime(x, '%Y%m%d'),
                        help='end date of data download period (YYYYMMDD)')
    parser.add_argument('-i', '--iotpath', type=str,
                        help='path to IoT data folder')
    parser.add_argument('-w', '--wgetpath', type=str,
                        help='path to wget executable',
                        default='/usr/bin/wget')
    args = parser.parse_args()

    main(args.sensor,
         args.startdate,
         args.enddate,
         args.iotpath,
         args.wgetpath)
