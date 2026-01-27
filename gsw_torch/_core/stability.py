"""
Core PyTorch implementations of stability functions.

These functions calculate stability-related properties like thermobaric and cabbeling coefficients.
"""

import torch

from .._utilities import as_tensor
from .._core.density import (
    specvol_alpha_beta,
    specvol_first_derivatives,
    specvol_second_derivatives,
)

__all__ = [
    "thermobaric",
    "cabbeling",
]


def thermobaric(SA, CT, p):
    """
    Calculates the thermobaric coefficient of seawater.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    thermobaric : torch.Tensor, 1/(K Pa)
        thermobaric coefficient with respect to Conservative Temperature

    Notes
    -----
    This is a pure PyTorch implementation. The formula is:
    thermobaric = (1/v) * (v_CT_p - v_CT * v_p / v)
    where v is specific volume and v_CT_p, v_CT, v_p are derivatives.
    """
    specvol, _, _ = specvol_alpha_beta(SA, CT, p)
    _, v_CT, v_p = specvol_first_derivatives(SA, CT, p)
    _, _, _, _, v_CT_p = specvol_second_derivatives(SA, CT, p)
    
    # Convert v_p from Pa to dbar for consistency (v_CT_p is already in Pa)
    # Actually, v_p and v_CT_p are both in Pa, so the formula should work directly
    # But let me check: thermobaric = (1/v) * (v_CT_p - v_CT * v_p / v)
    thermobaric = (1.0 / specvol) * (v_CT_p - v_CT * v_p / specvol)
    
    return thermobaric


def cabbeling(SA, CT, p):
    """
    Calculates the cabbeling coefficient of seawater.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    cabbeling : torch.Tensor, 1/K^2
        cabbeling coefficient with respect to Conservative Temperature

    Notes
    -----
    This is a pure PyTorch implementation. The formula is:
    cabbeling = (1/v) * (v_CT_CT - 2*v_CT*v_SA_CT/v_SA + v_CT^2*v_SA_SA/v_SA^2)
    where v is specific volume and the v_* are derivatives.
    """
    specvol, _, _ = specvol_alpha_beta(SA, CT, p)
    v_SA, v_CT, _ = specvol_first_derivatives(SA, CT, p)
    v_SA_SA, v_SA_CT, v_CT_CT, _, _ = specvol_second_derivatives(SA, CT, p)
    
    # cabbeling = (1/v) * (v_CT_CT - 2*v_CT*v_SA_CT/v_SA + v_CT^2*v_SA_SA/v_SA^2)
    cabbeling = (1.0 / specvol) * (
        v_CT_CT - 2.0 * v_CT * v_SA_CT / v_SA + v_CT**2 * v_SA_SA / v_SA**2
    )
    
    return cabbeling
