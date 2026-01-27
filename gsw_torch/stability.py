"""
Vertical stability functions.

These work with tensors of profiles; use the `axis` keyword
argument to specify the axis along which pressure varies.
For example, the default, following the Matlab versions, is
`axis=0`, meaning the pressure varies along the first dimension.
Use `axis=-1` if pressure varies along the last dimension--that
is, along a row, as the column index increases, in the 2-D case.
"""

import torch

from ._core.stability import cabbeling, thermobaric
from ._utilities import axis_slicer, match_args_return

__all__ = [
    "Nsquared",
    "Turner_Rsubrho",
    "IPV_vs_fNsquared_ratio",
    "thermobaric",
    "cabbeling",
]

# Import core functions
from ._core.conversions import grav
from ._core.density import specvol_alpha_beta

# In the following, axis=0 matches the Matlab behavior.


@match_args_return
def Nsquared(SA, CT, p, lat=None, axis=0):
    """
    Calculate the square of the buoyancy frequency.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lat : array-like, 1-D, optional
        Latitude, degrees.
    axis : int, optional
        The dimension along which pressure increases.

    Returns
    -------
    N2 : torch.Tensor
        Buoyancy frequency-squared at pressure midpoints, 1/s^2.
        The shape along the pressure axis dimension is one
        less than that of the inputs.
        (Frequency N is in radians per second.)
    p_mid : torch.Tensor
        Pressure at midpoints of p, dbar.
        The array shape matches N2.
    """
    SA = torch.as_tensor(SA, dtype=torch.float64)
    CT = torch.as_tensor(CT, dtype=torch.float64)
    p = torch.as_tensor(p, dtype=torch.float64)

    if lat is not None:
        lat = torch.as_tensor(lat, dtype=torch.float64)
        if torch.any((lat < -90) | (lat > 90)):
            raise ValueError("lat is out of range")
        SA, CT, p, lat = torch.broadcast_tensors(SA, CT, p, lat)
        g = _grav(lat, p)
    else:
        SA, CT, p = torch.broadcast_tensors(SA, CT, p)
        g = torch.tensor(9.7963, dtype=torch.float64, device=SA.device)  # (Griffies, 2004)

    db_to_pa = 1e4
    shallow = axis_slicer(SA.ndim, slice(-1), axis)
    deep = axis_slicer(SA.ndim, slice(1, None), axis)
    if lat is not None:
        g_local = 0.5 * (g[shallow] + g[deep])
    else:
        g_local = g

    dSA = SA[deep] - SA[shallow]
    dCT = CT[deep] - CT[shallow]
    dp = p[deep] - p[shallow]
    SA_mid = 0.5 * (SA[shallow] + SA[deep])
    CT_mid = 0.5 * (CT[shallow] + CT[deep])
    p_mid = 0.5 * (p[shallow] + p[deep])

    specvol_mid, alpha_mid, beta_mid = specvol_alpha_beta(SA_mid, CT_mid, p_mid)

    N2 = ((g_local**2) / (specvol_mid * db_to_pa * dp))
    N2 *= beta_mid * dSA - alpha_mid * dCT

    return N2, p_mid


@match_args_return
def Turner_Rsubrho(SA, CT, p, axis=0):
    """
    Calculate the Turner Angle and the Stability Ratio.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    axis : int, optional
        The dimension along which pressure increases.

    Returns
    -------
    Tu : torch.Tensor
        Turner Angle at pressure midpoints, degrees.
        The shape along the pressure axis dimension is one
        less than that of the inputs.
    Rsubrho : torch.Tensor
        Stability Ratio, dimensionless.
        The shape matches Tu.
    p_mid : torch.Tensor
        Pressure at midpoints of p, dbar.
        The array shape matches Tu.
    """
    SA = torch.as_tensor(SA, dtype=torch.float64)
    SA = torch.clamp(SA, 0, 50)
    CT = torch.as_tensor(CT, dtype=torch.float64)
    p = torch.as_tensor(p, dtype=torch.float64)

    SA, CT, p = torch.broadcast_tensors(SA, CT, p)
    shallow = axis_slicer(SA.ndim, slice(-1), axis)
    deep = axis_slicer(SA.ndim, slice(1, None), axis)

    dSA = -SA[deep] + SA[shallow]
    dCT = -CT[deep] + CT[shallow]

    SA_mid = 0.5 * (SA[shallow] + SA[deep])
    CT_mid = 0.5 * (CT[shallow] + CT[deep])
    p_mid = 0.5 * (p[shallow] + p[deep])

    _, alpha, beta = _specvol_alpha_beta(SA_mid, CT_mid, p_mid)

    Tu = torch.atan2((alpha * dCT + beta * dSA), (alpha * dCT - beta * dSA))
    Tu = torch.rad2deg(Tu)

    igood = dSA != 0
    Rsubrho = torch.zeros_like(dSA)
    Rsubrho.fill_(float("nan"))
    Rsubrho[igood] = (alpha[igood] * dCT[igood]) / (beta[igood] * dSA[igood])

    return Tu, Rsubrho, p_mid


@match_args_return
def IPV_vs_fNsquared_ratio(SA, CT, p, p_ref=0, axis=0):
    """
    Calculates the ratio of the vertical gradient of potential density to
    the vertical gradient of locally-referenced potential density.  This
    is also the ratio of the planetary Isopycnal Potential Vorticity
    (IPV) to f times N^2, hence the name for this variable,
    IPV_vs_fNsquared_ratio (see Eqn. (3.20.17) of IOC et al. (2010)).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    p_ref : float
        Reference pressure, dbar
    axis : int, optional
        The dimension along which pressure increases.

    Returns
    -------
    IPV_vs_fNsquared_ratio : torch.Tensor
        The ratio of the vertical gradient of
        potential density referenced to p_ref, to the vertical
        gradient of locally-referenced potential density, dimensionless.
    p_mid : torch.Tensor
        Pressure at midpoints of p, dbar.
        The array shape matches IPV_vs_fNsquared_ratio.
    """
    SA = torch.as_tensor(SA, dtype=torch.float64)
    SA = torch.clamp(SA, 0, 50)
    CT = torch.as_tensor(CT, dtype=torch.float64)
    p = torch.as_tensor(p, dtype=torch.float64)
    p_ref = torch.as_tensor(p_ref, dtype=torch.float64)

    SA, CT, p = torch.broadcast_tensors(SA, CT, p)
    shallow = axis_slicer(SA.ndim, slice(-1), axis)
    deep = axis_slicer(SA.ndim, slice(1, None), axis)

    dSA = -SA[deep] + SA[shallow]
    dCT = -CT[deep] + CT[shallow]

    SA_mid = 0.5 * (SA[shallow] + SA[deep])
    CT_mid = 0.5 * (CT[shallow] + CT[deep])
    p_mid = 0.5 * (p[shallow] + p[deep])

    _, alpha, beta = specvol_alpha_beta(SA_mid, CT_mid, p_mid)
    _, alpha_pref, beta_pref = specvol_alpha_beta(SA_mid, CT_mid, p_ref)

    num = dCT * alpha_pref - dSA * beta_pref
    den = dCT * alpha - dSA * beta

    igood = den != 0
    IPV_vs_fNsquared_ratio = torch.zeros_like(dSA)
    IPV_vs_fNsquared_ratio.fill_(float("nan"))
    IPV_vs_fNsquared_ratio[igood] = num[igood] / den[igood]

    return IPV_vs_fNsquared_ratio, p_mid
