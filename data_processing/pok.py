
import numpy as np
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from data_processing import variogram, torch_functions
# import torch_functions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def torch_pok(iotlon, iotlat, iotvals, kriglon, kriglat, semivarparams,
              torchmodelfunc, logscale, device, penalty):
    """
    Perform (penalized) Ordinary Kriging using PyTorch on GPU.

    Parameters
    ----------
    iotlon : np.array
        Longitudes of the observation points.
    iotlat : np.array
        Latitudes of the observation points.
    iotvals : np.array
        Values of the observation points.
    kriglon : np.array
        Longitudes of the prediction locations.
    gridlat : np.array
        Latitudes of the prediction locations.
    semivarparams : np.array
        Parameters for the semivariogram model.
    torchmodelfunc : function
        PyTorch function to compute semivariogram values.
    logscale : bool
        Whether to perform calculations in log scale.
    device : torch.device
        Device to perform calculations on.
    penalty : float
        Regularization parameter to stabilize the solution.

    Returns
    -------
    newkrigval : np.array
        Kriging estimates.
    newkrigvar : np.array
        Kriging variances.
    """

    # set up float type for GPU calculations
    floattype = torch.float32

    # convert arrays to pytorch tensors
    iotlat = torch.tensor(iotlat, dtype=floattype, device=device)
    iotlon = torch.tensor(iotlon, dtype=floattype, device=device)
    iotvals = torch.tensor(iotvals, dtype=floattype, device=device)

    # Flatten gX and gY for vectorized distance calculations
    gX_flat = torch.tensor(kriglon, dtype=floattype, device=device)
    gY_flat = torch.tensor(kriglat, dtype=floattype, device=device)

    # create kriging matrix
    n = len(iotvals)
    A = torch.zeros((n+1, n+1), dtype=floattype, device=device)  # Ensure A is a tensor on the GPU
    for k in range(n):
        tmplat = iotlat[k].unsqueeze(0)  # Unsqueeze for dimension compatibility
        tmplon = iotlon[k].unsqueeze(0)
        tmpdist = torch_functions.haversine_distance_torch_batch(tmplat, tmplon,
                                                                 iotlat[k:],
                                                                 iotlon[k:],
                                                                 batch_size=3000)

        semivar = -1 * torchmodelfunc(tmpdist, *semivarparams)

        A[k, k:n] = semivar.T
        A[k:n, k] = semivar.T

    A.fill_diagonal_(penalty)
    A[-1, :-1] = 1.0
    A[:-1, -1] = 1.0
    A[-1, -1] = 0.0

    # due to memory limitations on GPU, run calculations in batches
    batch_size = 75 * 1000

    # set up empty tensors for kriging solutions
    kriging_values = torch.full(gX_flat.shape, float('nan'), dtype=floattype,
                                device=device)
    kriging_variance = torch.full(gX_flat.shape, float('nan'), dtype=floattype,
                                  device=device)
    lagrange = torch.full(gX_flat.shape, float('nan'), dtype=floattype,
                          device=device)
    for k in tqdm(range(0, len(gX_flat), batch_size), disable=True):

        # select batch
        tmp_gX = gX_flat[k:k+batch_size]
        tmp_gY = gY_flat[k:k+batch_size]

        # Assuming iotlat and iotlon are already tensors on the GPU
        # Use the revised haversine_distance_torch for batch calculation
        # No need for looping over each grid point
        c_batch = len(tmp_gX) // 3
        cdist = torch_functions.haversine_distance_torch_batch(iotlat, iotlon,
                                                               tmp_gY, tmp_gX,
                                                               batch_size=c_batch)

        # setup right hand side
        b = torch.zeros((cdist.size(0), n + 1), dtype=floattype,
                         device=cdist.device)
        b[:, -1] = 1.0
        b[:, :-1] = -1 * torchmodelfunc(cdist, *semivarparams)

        # solve kriging system
        weights = torch.linalg.solve(A, b.T)

        # Calculate kriging estimate
        # Ensure operations are performed with PyTorch methods and on the correct device
        krigval = (weights[:-1, :].T * iotvals).sum(dim=1)

        # Calculate uncertainty of estimate
        krigvar = (weights * -b.T).sum(dim=0)

        # now fill empty tensors with kriging results
        kriging_values[k:k+batch_size] = krigval
        kriging_variance[k:k+batch_size] = krigvar
        lagrange[k:k+batch_size] = weights[-1, :]

    # send results back to CPU
    cpu_krig_values = kriging_values.cpu().numpy()
    cpu_krig_variance = kriging_variance.cpu().numpy()
    cpu_lagrange = lagrange.cpu().numpy()

    if logscale:
        # backtransform to original scale
        import warnings
        warnings.filterwarnings('error', category=RuntimeWarning)

        newkrigval = np.exp(cpu_krig_values + cpu_krig_variance/2 + cpu_lagrange)
        newkrigvar = (np.exp(cpu_krig_variance)-1) * np.exp(2*cpu_krig_values + cpu_krig_variance)

    else:
        newkrigval = cpu_krig_values
        newkrigvar = cpu_krig_variance

    return newkrigval, newkrigvar


def run_kriging(iotlon, iotlat, iotvals, kriglon, kriglat, kriglags, penalty,
                device, plotting=False, logscale=False):
    """
    Perform kriging for a single time step.

    Parameters
    ----------
    iotlon : np.array
        Longitudes of the observation points.
    iotlat : np.array
        Latitudes of the observation points.
    iotvals : np.array
        Values of the observation points.
    kriglon : np.array
        Longitudes of the prediction locations.
    kriglat : np.array
        Latitudes of the prediction locations.
    kriglags : np.array
        Lag distances for the semivariogram.
    penalty : float
        Regularization parameter to stabilize the solution.
    device : torch.device
        Device to perform calculations on.
    plotting : bool
        Whether to plot the semivariogram, defaults to False.

    Returns
    -------
    ok_val : np.array
        Kriging estimates.
    ok_var : np.array
        Kriging variances.
    """

    # use robust estimator for semivariogram (see Eq. 2.4.12 in Cressie (1993))
    kriglags = kriglags.astype('float32')
    [semi, semi_bin_count,
     semilags] = variogram.semivariogram_robust(iotlat, iotlon, iotvals,
                                                kriglags)
    selidx = np.isnan(semi)

    if np.all(selidx):
        # timestr = time.dt.strftime('%Y%m%d/%H:%M').item()
        raise ValueError(f'No valid semivariogram for this time step')

    # calculate variogram parameters for linear and spherical model
    # and then use the one with the lowest residual
    models = ['spherical']
    res = []
    for model in models:
        [popt, pcov,
        infodict] = variogram.calc_variogram_params(semilags[~selidx],
                                                    semi[~selidx],
                                                    model=model)
        res.append(popt)

    varmodel = 'spherical'
    semivarparams = popt

    # get model function
    modelfunc = variogram.spherical_varmodel
    torchmodelfunc = torch_functions.sph_varmodel_torch

    # check variogram
    if plotting:
        plt.plot(semilags, semi, 'ro')
        plt.plot(kriglags, modelfunc(kriglags, *semivarparams), label=varmodel)
        plt.xlabel('Distance (km)')
        plt.ylabel('Semivariance')
        plt.tight_layout()
        plt.show()

    # perform kriging
    try:
        ok_val, ok_var = torch_pok(iotlon, iotlat, iotvals, kriglon, kriglat,
                                semivarparams, torchmodelfunc,
                                logscale=logscale, device=device, penalty=penalty)
    except RuntimeWarning as e:
        timestr = time.dt.strftime('%Y%m%d/%H:%M').item()
        print('-'*40)
        print(f'Error in kriging: {e} in {timestr}')
        print('Likely instable solution, skipping this time step')
        raise
    except torch._C._LinAlgError as e:
        timestr = time.dt.strftime('%Y%m%d/%H:%M').item()
        print('-'*40)
        print(f'Error in kriging: {e} in {timestr}')
        print('Singular Matrix, skipping this time step')
        raise

    if abs(ok_val).max() > np.exp(iotvals.max()+1):
        timestr = time.dt.strftime('%Y%m%d/%H:%M').item()
        print('-'*40)
        print(f'Error in kriging in {timestr}')
        print('Maximal kriging much higher than input data, skipping this time step')
        raise ValueError('Kriging error')

    return ok_val, ok_var


