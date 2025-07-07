
import numpy as np
from numba import njit
from scipy.optimize import curve_fit


@njit('float32[:](float32, float32, float32[:], float32[:])')
def haversine_distance_single(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points assuming an
    Earth radius of 6371 km. It does not create the full distance matrix
    but rather calculates the distance between each pair of points. Thus,
    it is more memory efficient and avoids MemoryErrors and 'freezing' of
    computer.

    Parameters
    ----------
    lat1 : float
        Latitude of first point.
    lon1 : float
        Longitude of first point.
    lat2 : float, np.array
        Latitude of second point(s).
    lon2 : float, np.array
        Longitude of second point(s).

    Returns
    -------
    distance : float, np.array
        Distance between the two points in km.
    """

    R = 6371.0  # Earth radius in km

    # Convert degrees to radians
    # lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    # Differences in coordinates
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    distance = distance.astype(np.float32)

    return distance


def haversine_distance_numpy(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distances between sets of points
    in batch, using the Haversine formula.

    Parameters
    ----------
    lat1, lon1 : torch.Tensor
        Latitudes and longitudes of the first set of points.
        Should be 1D tensors.
    lat2, lon2 : torch.Tensor
        Latitudes and longitudes of the second set of points.
        Should be 1D tensors.

    Returns
    -------
    distances : torch.Tensor
        A 2D tensor where each element [i, j] is the distance
        between the i-th point from the first set and the
        j-th point from the second set.
    """

    # Convert degrees to radians
    lat1 = np.deg2rad(lat1)
    lon1 = np.deg2rad(lon1)
    lat2 = np.deg2rad(lat2)
    lon2 = np.deg2rad(lon2)

    # Earth radius in kilometers
    R = 6371.0

    # Broadcasting the latitudes and longitudes to compute all pairs of differences
    dlat = lat2[:, np.newaxis] - lat1[np.newaxis, :]
    dlon = lon2[:, np.newaxis] - lon1[np.newaxis, :]

    # Haversine formula
    a = np.sin(dlat / 2.0)**2 + np.cos(lat1)[np.newaxis, :] * np.cos(lat2)[:, np.newaxis] * np.sin(dlon / 2.0)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distances = R * c

    return distances


@njit('Tuple((float32[:], int32[:]))(float32[:], float32[:], float32[:], float32[:])')
def semivariogram(lat, lon, values, lags):
    """
    Calculate the semivariogram using the classical estimator.

    Parameters
    ----------
    lat : float, np.array
        Latitude of all points.
    lon : float, np.array
        Longitude of all points.
    values : float, np.array
        Values at all points.
    lags : float, np.array
        Distances at which to calculate or bin the semivariogram.

    Returns
    -------
    semi : float, np.array
        Semivariogram values.
    semivariogram_sum : int, np.array
        Number of points used to calculate each semivariogram value.
    """

    nvals = len(values)
    nlags = len(lags)
    semivariogram_values = np.zeros(lags.shape, dtype=np.float32)
    semivariogram_sum = np.zeros(lags.shape, dtype=np.int32)

    for i in range(nvals-1):
        # Last point will already have been compared with all others

        dist = haversine_distance_single(lat[i], lon[i],
                                         np.asarray(lat[i+1:]),
                                         np.asarray(lon[i+1:]))

        # Determine which lag each distance corresponds to
        indices = np.digitize(dist, lags, right=True)

        # Update the semivariogram values based on binned distances
        for ind in range(nlags):
            selidx = indices == ind
            if not np.any(selidx):
                continue

            valid_vals = values[i+1:][selidx]
            semivariogram_values[ind] += np.sum((values[i] - valid_vals)**2)
            semivariogram_sum[ind] += len(valid_vals)

    # Normalize the semivariogram values
    semi = semivariogram_values / (2 * semivariogram_sum)

    # Convert to float32 and int32
    semi = semi.astype(np.float32)
    semivariogram_sum = semivariogram_sum.astype(np.int32)

    return semi, semivariogram_sum


@njit('Tuple((float32[:], int32[:], float32[:]))(float32[:], float32[:], float32[:], float32[:])')
def semivariogram_robust(lat, lon, values, lags):
    """
    Calculate the semivariogram using a robust estimator (Eq. 2.4.12) in
    Cressie (1993).

    Parameters
    ----------
    lat : float, np.array
        Latitude of all points.
    lon : float, np.array
        Longitude of all points.
    values : float, np.array
        Values at all points.
    lags : float, np.array
        Distances at which to calculate or bin the semivariogram.

    Returns
    -------
    semi : float, np.array
        Semivariogram values.
    semivariogram_sum : int, np.array
        Number of points used to calculate each semivariogram value.
    semilags : float, np.array
        Midpoints of the lags.
    """

    nvals = len(values)
    nlags = len(lags)
    variogram_values = np.zeros((nlags-1), dtype=np.float32)
    variogram_sum = np.zeros((nlags-1), dtype=np.int32)

    for i in range(nvals-1):

        # Last point will already have been compared with all others
        dist = haversine_distance_single(lat[i], lon[i],
                                         np.asarray(lat[i+1:]),
                                         np.asarray(lon[i+1:]))

        # Determine which lag each distance corresponds to
        indices = np.digitize(dist, lags, right=True)

        # Update the semivariogram values based on binned distances
        for ind in range(nlags-1):
            selidx = np.where((dist > lags[ind]) & (dist <= lags[ind+1]))[0]
            if len(selidx) == 0:
                continue

            valid_vals = values[i+1:][selidx]
            variogram_values[ind] += np.sum(np.sqrt(np.abs(values[i] - valid_vals)))
            variogram_sum[ind] += len(valid_vals)

    # now calculate numerator and demoninator of Eq. 2.4.12
    numerator = (variogram_values / variogram_sum)**4
    denominator = 0.457 + 0.494 / variogram_sum + 0.045 / variogram_sum**2

    # Normalize the semivariogram values
    semi = 0.5 * numerator / denominator

    # Convert to float32 and int32
    semi = semi.astype(np.float32)
    semivariogram_sum = variogram_sum.astype(np.int32)
    semimids = 0.5 * (lags[1:] + lags[:-1])
    semimids = semimids.astype(np.float32)

    return semi, semivariogram_sum, semimids


def exponential_varmodel(d, psill, range_, nugget):
    """
    Function of exponential variogram model.

    Parameters
    ----------
    d : float, np.array
        Distance values.
    psill : float
        (Partial) sill of the variogram.
    range_ : float
        Range of the variogram.
    nugget : float
        Nugget of the variogram.

    Returns
    -------
    var : float, np.array
        Variogram values.
    """

    var = psill * (1.0 - np.exp(-d / (range_ / 3.0))) + nugget
    return var


def power_varmodel(d, scale, expo, nugget):
    """
    Function of the power variogram model.

    Parameters
    ----------
    d : float, np.array
        Distance values.
    scale : float
        Scale of the variogram.
    expo : float
        Exponent of the variogram.
    nugget : float
        Nugget of the variogram.

    Returns
    -------
    var : float, np.array
        Variogram values.
    """

    var = scale * (d**expo) + nugget

    return var


def spherical_varmodel(d, psill, range_, nugget):
    """
    Function of the spherical variogram model.

    Parameters
    ----------
    d : float, np.array
        Distance values.
    psill : float
        (Partial) sill of the variogram.
    range_ : float
        Range of the variogram.
    nugget : float
        Nugget of the variogram.

    Returns
    -------
    var : float, np.array
        Variogram values.
    """

    var = np.piecewise(d, [d <= range_, d > range_],
                        [lambda x: psill * (1.5 * (x / range_) - 0.5 * (x / range_)**3) + nugget,
                         psill + nugget])

    return var


def gauss_varmodel(d, psill, range_, nugget):
    """
    Function of the gaussian variogram model.

    Parameters
    ----------
    d : float, np.array
        Distance values.
    psill : float
        (Partial) sill of the variogram.
    range_ : float
        Range of the variogram.
    nugget : float
        Nugget of the variogram.

    Returns
    -------
    var : float, np.array
        Variogram values.
    """

    var = psill * (1.0 - np.exp(-(d**2.0) / (range_ * 4.0 / 7.0) ** 2.0)) + nugget

    return var


def linear_varmodel(d, slope, range_, nugget):
    """
    Function of the linear variogram model.

    Parameters
    ----------
    d : float, np.array
        Distance values.
    slope : float
        Slope of the variogram.
    range_ : float
        Range of the variogram.
    nugget : float
        Nugget of the variogram.

    Returns
    -------
    var : float, np.array
        Variogram values.
    """

    var = np.piecewise(d, [d <= range_, d > range_],
                        [lambda x: slope * x + nugget,
                         slope * range_ + nugget])
    return var


def logistic_varmodel(d, sill, range_, factor):
    """
    Function of the logistic variogram model.

    Parameters
    ----------
    d : float, np.array
        Distance values.
    sill : float
        Sill of the variogram.
    range_ : float
        Range of the variogram.
    factor : float
        Factor of the variogram.

    Returns
    -------
    var : float, np.array
        Variogram values.
    """

    var = sill / (1 + factor*np.exp(-d / range_))

    return var


def calc_variogram_params(lags, semivariance, model='spherical', sigma=None):
    """
    Perform nonlinear fit of variogram function to semivariogram values
    for a given model. The function uses the scipy.optimize.curve_fit
    and a soft L1 loss function to account for outliers.

    Parameters
    ----------
    lags : float, np.array
        Distances at which semivariances were calculated.
    semivariance : float, np.array
        Semivariance values.
    model : str
        Model to fit to the semivariance values. Options are 'exponential',
        'spherical', 'power', 'gaussian', 'linear', 'logistic'. Default is
        'spherical'.
    sigma : float, np.array
        Standard deviation of the semivariance values for weighting during
        the non-linear fit. If None, no weighting is applied.

    Returns
    -------
    popt : float, np.array
        Estimated (optimal) values for the variogram model parameters
    pcov : float, np.array
        Estimated covariance of the optimal values.
    infodict : dict
        Dictionary containing information about the fit.
    """

    x0 = [np.amax(semivariance) - np.amin(semivariance),
          0.25 * np.amax(lags),
          np.amin(semivariance)]

    upperbounds = (10.0 * np.amax(semivariance),
                   np.amax(lags),
                   np.amax(semivariance))

    if model == 'exponential':
        modelfunc = exponential_varmodel
    elif model == 'spherical':
        modelfunc = spherical_varmodel
    elif model == 'power':
        modelfunc = power_varmodel
        x0 = [np.amax(semivariance), 2.0, np.amin(semivariance)]
        upperbounds = (10.0 * np.amax(semivariance), 5.0, np.amax(semivariance))
    elif model == 'gaussian':
        modelfunc = gauss_varmodel
    elif model == 'linear':
        modelfunc = linear_varmodel
    elif model == 'logistic':
        modelfunc = logistic_varmodel
    else:
        raise ValueError('Model not recognized')

    # bnds = (lowerbounds, upperbounds)

    # res = least_squares(variogram_residuals, x0, bounds=bnds,
    #                     args=(lags, semivariance, modelfunc),
    #                     loss='soft_l1')

    popt, pcov, infodict, _, _ = curve_fit(modelfunc, lags, semivariance,
                                           p0=x0, bounds=(0, upperbounds),
                                           sigma=sigma, loss='soft_l1',
                                           method='trf', full_output=True)

    return popt, pcov, infodict


def calc_avg_distance(lat, lon):
    """
    Calculate average distances between points.

    Parameters
    ----------
    lat : np.array
        Latitudes of the points.
    lon : np.array
        Longitudes of the points.

    Returns
    -------
    avgdist : float
        Average distance between points.
    mediandist : float
        Median distance between points.
    p75dist : float
        75th percentile distance between points.
    maxdist : float
        Maximum distance between points.
    """

    nvals = len(lat)
    sumdist = 0
    sum_elements = 0
    maxdist = 0
    medianlist = []
    for i in range(nvals-1):
        dist = haversine_distance_single(lat[i], lon[i],
                                         np.asarray(lat[i+1:]),
                                         np.asarray(lon[i+1:]))
        sumdist += dist.sum()
        sum_elements += len(dist)
        maxdist = max(maxdist, dist.max())
        medianlist.append(np.median(np.append(dist, 0)))

    avgdist = sumdist / (sum_elements + nvals)
    mediandist = np.median(medianlist)
    p25dist = np.percentile(medianlist, 25)
    p75dist = np.percentile(medianlist, 75)

    return avgdist, mediandist, p75dist, maxdist