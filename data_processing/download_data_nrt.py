

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path
from argparse import ArgumentParser
import xarray as xr
import numpy as np
import pandas as pd
from tqdm import tqdm
import earthkit.data
from earthkit.meteo import thermo


def get_sensor_community_urls(date: datetime,
                              downtxt: Path) -> None:
    """
    Get urls for sensor.community data for a given date and write to file.

    Parameters
    ----------
    date : datetime
        date to get urls for
    downtxt : Path
        filename to write urls to

    Returns
    -------
    No returns, but writes urls to file.
    """

    url = f"https://archive.sensor.community/{date:%Y-%m-%d}/"
    reqs = requests.get(url)

    soup = BeautifulSoup(reqs.text, 'html.parser')
    soupdata = soup.find_all('a')

    urls = [url+link.get('href') for link in soupdata]
    urls = [url for url in urls if url.endswith('.csv') and 'indoor' not in url and 'sds011' in url]

    # write urls to file line by line
    with open(downtxt, 'w') as f:
        for url in sorted(urls):
            f.write(url+'\n')


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


def processs_sensor_community_nrt(sensor: str,
                              date: datetime,
                              iotpath: str,
                              outfn: Path = None) -> None:
    """
    Process downloaded (NRT) IoT data from sensor.community and save as
    parquet file.

    Parameters
    ----------
    sensor : str
        sensor type
    date : datetime.datetime
        date corresponding to downloaded data
    iotpath : str
        path to IoT data folder
    outfn : Path, optional
        output filename to save parquet file, by default None.

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

    fnlist = sorted(iotpath.glob('*.csv'))
    if len(fnlist) == 0:
        raise ValueError(f"No files found for {sensor} on {date:%Y-%m-%d}.")

    # download IoT data from sensor.community
    coldf = []
    for csvfn in tqdm(fnlist, total=len(fnlist)):

        # read unzipped file and clean up data
        sdict = sensordict[sensor]
        floatcols = sdict['floatcols']
        intcols = sdict['intcols']
        dropcols = sdict['dropcols']

        df = pd.read_csv(csvfn, sep=';')
        df = df.drop(dropcols, axis=1)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed',
                                         errors='coerce')
        df[floatcols] = vecconv(df[floatcols])
        df[intcols] = df[intcols].astype(np.int32)
        df = df.dropna()
        coldf.append(df)

    #  merge chunks of cleaned dataframes
    mergedf = pd.concat(coldf)
    mergedf = mergedf.dropna()

    iotpath = Path(iotpath)
    outfold = Path(iotpath.parent, 'L1A', sensor)
    if not outfold.exists():
        outfold.mkdir(parents=True)

    if outfn is None:
        outname = f"{sensor}_{date:%Y%m%d}.parquet"
        outfn = Path(outfold, outname)

    mergedf.to_parquet(outfn, index=False)


def download_ecmwf_data(date: datetime,
                        outfold: Path) -> None:
    """
    Download meteorological and air quality data via earthkit and save as netCDF files.

    Parameters
    ----------
    date : datetime
        date to process
    outfold : Path
        folder to write output to

    Returns
    -------
    No returns, but writes netCDF files to disk.
    """

    params = {
              'param': ["2t", '2d', 'blh', 'sp'],
              'levtype': "sfc",
              'class': "od",
              'time': 00,
              'date': date,
              'step': [0, 'to', 23, 'by', 1],
              'stream': "oper",
              'expver': 1,
              'type': "fc",
              'grid': [0.125, 0.125],
              'area': [75, -27, 28, 47] #N,W,S,E
              }

    meteodata = earthkit.data.from_source('mars', params)
    meteods = meteodata.to_xarray()
    meteods = meteods.squeeze()
    rh = thermo.relative_humidity_from_dewpoint(meteods['2t'], meteods['2d'])
    meteods['rh'] = rh

    meteods = meteods.rename({'2t': 't2m', '2d': 'd2m'})

    # convert steps to datetime objects
    init_time = datetime.strptime(str(meteods.date), '%Y%m%d')
    init_time = init_time.replace(hour=meteods.time)
    valid_for = pd.Timestamp(init_time) + pd.TimedeltaIndex(meteods.step)

    # replace step variable and dimension with valid time
    meteods = meteods.rename({'step': 'time'})
    meteods = meteods.assign_coords({'time': valid_for})

    # before saving, convert to float32 and drop unnecessary attributes
    meteods = meteods.astype(np.float32)
    meteods = meteods.drop_attrs()

    # save dataset to netCDF
    outfn = Path(outfold, f"meteo_{date:%Y%m%d}.nc")
    meteods.to_netcdf(outfn)

    colpm = []
    for pmvar in ['particulate_matter_10um', 'particulate_matter_2.5um']:
        cams_params = {"variable": [pmvar],
                    "area": [72, -25, 30, 45],
                        "date": date,
                        'model': 'ensemble',
                        'level': '0',
                        'type': 'analysis',
                        'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                        ],
                        'leadtime_hour': '0',
                        }
        camsdata = earthkit.data.from_source("ads", 'cams-europe-air-quality-forecasts', cams_params)
        camsds = camsdata.to_xarray()
        camsds = camsds.squeeze()
        camsds = camsds.rename({'forecast_reference_time': 'time'})
        try:
            camsds = camsds.drop_vars(['step', 'surface', 'valid_time'])
        except ValueError:
            pass

        polname = 'pm10_conc' if '10um' in pmvar else 'pm2p5_conc'
        camsds = camsds.rename({'mdens': polname})
        colpm.append(camsds)

    camsds = xr.merge(colpm)

    # convert to float32 and drop attributes
    camsds = camsds.astype(np.float32)
    camsds = camsds.drop_attrs()
    outfn = Path(outfold, f"cams_{date:%Y%m%d}.nc")
    camsds.to_netcdf(outfn)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('-d', '--date',
                        type=lambda s: datetime.strptime(s, '%Y%m%d'),
                        help="Date to process (YYYYMMDD)", required=True)
    parser.add_argument('-o', '--outfold',
                        type=lambda s: Path(s),
                        help="Folder to write output to", required=True)
    parser.add_argument('-s', '--sensorcommunity', action='store_true',
                        help="Download sensor.community data")
    parser.add_argument('-e', '--ecmwf', action='store_true',
                        help="Download ECMWF data")
    args = parser.parse_args()

    if args.sensorcommunity:
        downtxt = Path(args.outfold, f"urls_{args.date:%Y%m%d}.txt")
        get_sensor_community_urls(args.date, downtxt)

    if args.ecmwf:
        download_ecmwf_data(args.date, args.outfold)


# how to download sensor community data:
# aria2c -x6 -i urls_20210101.txt -d /path/to/folder