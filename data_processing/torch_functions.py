
import torch


def exp_varmodel_torch(d, psill, range_, nugget):
    """
    Function of the exponential variogram model using PyTorch.

    Parameters
    ----------
    d : torch.Tensor
        Distance between two points.
    psill : float
        (Partial) sill of the variogram model.
    range_ : float
        Range of the variogram model.
    nugget : float
        Nugget of the variogram model.

    Returns
    -------
    var : torch.Tensor
        Variance of the variogram model.
    """

    var = psill * (1.0 - torch.exp(-d / (range_ / 3.0))) + nugget
    return var


def sph_varmodel_torch(d, psill, range_, nugget):
    """
    Function of the spherical variogram model using PyTorch.

    Parameters
    ----------
    d : torch.Tensor
        Distance between two points.
    psill : float
        (Partial) sill of the variogram model.
    range_ : float
        Range of the variogram model.
    nugget : float
        Nugget of the variogram model.

    Returns
    -------
    var : torch.Tensor
        Variance of the variogram model.
    """

    var = torch.where(d <= range_,
                      psill * (1.5 * (d / range_) - 0.5 * (d / range_)**3) + nugget,
                      psill + nugget)

    return var


def power_varmodel_torch(d, scale, expo, nugget):
    """
    Function of the power variogram modelusing PyTorch.

    Parameters
    ----------
    d : torch.Tensor
        Distance between two points.
    scale : float
        Scale of the variogram model.
    expo : float
        Exponent of the variogram model.
    nugget : float
        Nugget of the variogram model.

    Returns
    -------
    var : torch.Tensor
        Variance of the variogram model.
    """

    var = scale * (d**expo) + nugget
    return var


def gauss_varmodel_torch(d, psill, range_, nugget):
    """
    Function of the gaussian variogram model using PyTorch.

    Parameters
    ----------
    d : torch.Tensor
        Distance between two points.
    psill : float
        (Partial) sill of the variogram model.
    range_ : float
        Range of the variogram model.
    nugget : float
        Nugget of the variogram model.

    Returns
    -------
    var : torch.Tensor
        Variance of the variogram model.
    """

    var = psill * (1.0 - torch.exp(-(d**2.0) / (range_ * 4.0 / 7.0) ** 2.0)) + nugget
    return var


def linear_varmodel_torch(d, slope, range_, nugget):
    """
    Function of the linear variogram model using PyTorch.

    Parameters
    ----------
    d : torch.Tensor
        Distance between two points.
    slope : float
        Slope of the variogram model.
    range_ : float
        Range of the variogram model.
    nugget : float
        Nugget of the variogram model.

    Returns
    -------
    var : torch.Tensor
        Variance of the variogram model.
    """

    var = torch.where(d <= range_, slope * d + nugget, slope * range_ + nugget)
    return var


def haversine_distance_torch(lat1, lon1, lat2, lon2):
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
    # Ensure input tensors are floats and on the same device
    lat1, lon1, lat2, lon2 = [x.float() for x in [lat1, lon1, lat2, lon2]]

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])

    # Earth radius in kilometers
    R = 6371.0

    # Broadcasting the latitudes and longitudes to compute all pairs of differences
    dlat = lat2[:, None] - lat1[None, :]
    dlon = lon2[:, None] - lon1[None, :]

    # Haversine formula
    a = torch.sin(dlat / 2.0)**2 + torch.cos(lat1)[None, :] * torch.cos(lat2)[:, None] * torch.sin(dlon / 2.0)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    distances = R * c

    return distances


def haversine_distance_torch_batch(lat1, lon1, lat2, lon2, batch_size=100):
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
    batch_size : int, optional
        The number of points in each batch for memory-efficient calculation.

    Returns
    -------
    distances : torch.Tensor
        A 2D tensor where each element [i, j] is the distance
        between the i-th point from the first set and the
        j-th point from the second set.
    """
    # Ensure input tensors are floats and on the same device
    lat1, lon1, lat2, lon2 = [x.float() for x in [lat1, lon1, lat2, lon2]]

    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(torch.deg2rad, [lat1, lon1, lat2, lon2])

    # Earth radius in kilometers
    R = 6371.0

    # Initialize an empty tensor to store distances
    distances = torch.zeros((lat2.size(0), lat1.size(0)), device=lat1.device)

    # Compute distances in batches to save memory
    for i in range(0, lat2.size(0), batch_size):
        lat2_batch = lat2[i:i + batch_size]
        lon2_batch = lon2[i:i + batch_size]

        # Broadcasting differences in small batches
        dlat = lat2_batch[:, None] - lat1[None, :]
        dlon = lon2_batch[:, None] - lon1[None, :]

        # Haversine formula
        a = torch.sin(dlat / 2.0)**2 + torch.cos(lat1)[None, :] * torch.cos(lat2_batch)[:, None] * torch.sin(dlon / 2.0)**2
        c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

        distances[i:i + batch_size, :] = R * c

    return distances



