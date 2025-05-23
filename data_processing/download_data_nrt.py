

import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pathlib import Path
from argparse import ArgumentParser
import xarray as xr
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
    rh = thermo.relative_humidity_from_dewpoint(meteods['t2m'], meteods['d2m'])
    meteods['rh'] = rh
    dayds = meteods.mean('step')
    dayds = dayds.drop_vars(['number', 'surface'])
    outfn = Path(outfold, f"meteo_{date:%Y%m%d}.nc")
    dayds.to_netcdf(outfn)

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
        camsds = camsds.drop_vars(['step', 'surface', 'valid_time'])

        polname = 'pm10_conc' if '10um' in pmvar else 'pm2p5_conc'
        camsds = camsds.rename({'mdens': polname})
        colpm.append(camsds)

    camsds = xr.merge(colpm)
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