"""
Conversions involving temperature, salinity, entropy, pressure,
and height.

Those most commonly used probably include:

- :func:`gsw_torch.CT_from_t`
- :func:`gsw_torch.SA_from_SP`
- :func:`gsw_torch.SP_from_C`
- :func:`gsw_torch.p_from_z`
- :func:`gsw_torch.z_from_p`

"""

__all__ = [
    "t90_from_t68",
    "grav",
    "p_from_z",
    "z_from_p",
    "CT_from_t",
    "CT_from_pt",
    "SA_from_rho",
    "SA_from_SP",
    "SP_from_SA",
    "t_from_CT",
    "pt_from_CT",
    "pt_from_entropy",
    "pt_from_t",
    "pt0_from_t",
    "entropy_from_CT",
    "entropy_from_pt",
    "entropy_from_t",
    "entropy_first_derivatives",
    "entropy_second_derivatives",
    "adiabatic_lapse_rate_from_CT",
    "CT_from_enthalpy",
    "CT_from_entropy",
    "CT_from_rho",
    "CT_maxdensity",
    "CT_first_derivatives",
    "CT_first_derivatives_wrt_t_exact",
    "CT_from_enthalpy_exact",
    "CT_second_derivatives",
    "dilution_coefficient_t_exact",
    "pt_first_derivatives",
    "pt_second_derivatives",
    "SAAR",
    "deltaSA_from_SP",
    "C_from_SP",
    "SP_from_C",
    "SA_from_Sstar",
    "SP_from_SK",
    "SP_from_SR",
    "SP_from_Sstar",
    "SR_from_SP",
    "Sstar_from_SA",
    "Sstar_from_SP",
]

# Import core implementations (will be created in Phase 4)
# For now, we'll implement the simple pure Python function
from ._utilities import match_args_return


@match_args_return
def t90_from_t68(t68):
    """
    ITS-90 temperature from IPTS-68 temperature

    This conversion should be applied to all in-situ
    data collected between 1/1/1968 and 31/12/1989.

    Parameters
    ----------
    t68 : array-like
        IPTS-68 temperature, degrees C

    Returns
    -------
    t90 : array-like
        ITS-90 temperature, degrees C
    """
    import torch

    t68_tensor = torch.as_tensor(t68, dtype=torch.float64)
    return t68_tensor / 1.00024


# Import from core modules
from ._core.conversions import (  # noqa: E402
    SAAR,
    C_from_SP,
    CT_first_derivatives,
    CT_first_derivatives_wrt_t_exact,
    CT_from_enthalpy,
    CT_from_enthalpy_exact,
    CT_from_entropy,
    CT_from_pt,
    CT_from_rho,
    CT_from_t,
    CT_maxdensity,
    CT_second_derivatives,
    SA_from_rho,
    SA_from_SP,
    SA_from_Sstar,
    SP_from_C,
    SP_from_SA,
    SP_from_SK,
    SP_from_SR,
    SP_from_Sstar,
    SR_from_SP,
    Sstar_from_SA,
    Sstar_from_SP,
    adiabatic_lapse_rate_from_CT,
    deltaSA_from_SP,
    dilution_coefficient_t_exact,
    entropy_first_derivatives,
    entropy_from_CT,
    entropy_from_pt,
    entropy_from_t,
    entropy_second_derivatives,
    grav,
    p_from_z,
    pt0_from_t,
    pt_first_derivatives,
    pt_from_CT,
    pt_from_entropy,
    pt_from_t,
    pt_second_derivatives,
    t_from_CT,
    z_from_p,
)

# Placeholder imports - will be implemented incrementally
# These will import from _core modules once they're created
# For now, we create stubs that raise NotImplementedError


def _not_implemented(name):
    """Helper to create placeholder functions."""

    def func(*args, **kwargs):
        raise NotImplementedError(
            f"{name} is not yet implemented. This function will be available in a future release."
        )

    func.__name__ = name
    return func


# Placeholder functions - will be replaced with real implementations
# (pt_from_entropy and SA_from_rho are now implemented)
