
import numpy as np


def sat_water_vapor(T):
    """
    Calculate the water vapor pressure over liquid water according to Murphy and Koop (2005).

    Parameters
    ----------
    T : float, np.array
        Temperature in Kelvin.

    Returns
    -------
    e : float, np.array
        Water vapor pressure in Pa.
    """
    e = np.exp(
        54.842763 - 6763.22 / T - 4.210 * np.log(T) + 0.000367 * T
        + np.tanh(0.0415 * (T - 218.8)) * (
            53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T
        )
    )

    return e


def shiftgrid(ds, lonname):
    """
    Shift global longitude grid from 0 - 360 to -180 - 180 degrees East.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing longitude coordinate to be shifted.
    lonname : str
        Name of longitude coordinate.

    Returns
    -------
    newds : xarray.Dataset
        Dataset with shifted longitude coordinate.
    """

    newds = ds.assign_coords({lonname: (((ds[lonname] + 180) % 360) - 180)})
    return newds


def latlon_to_geocentric(lat: np.array,
                         lon: np.array) -> np.array:
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