"""
Functions about ice and melting, but not the freezing point.
"""

# Import from _core modules
from ._core.ice import (
    Helmholtz_energy_ice,
    adiabatic_lapse_rate_ice,
    alpha_wrt_t_ice,
    chem_potential_water_ice,
    cp_ice,
    enthalpy_ice,
    entropy_ice,
    ice_fraction_to_freeze_seawater,
    internal_energy_ice,
    kappa_const_t_ice,
    kappa_ice,
    melting_ice_equilibrium_SA_CT_ratio,
    melting_ice_into_seawater,
    melting_ice_SA_CT_ratio,
    melting_seaice_equilibrium_SA_CT_ratio,
    melting_seaice_into_seawater,
    melting_seaice_SA_CT_ratio,
    pot_enthalpy_from_pt_ice,
    pot_enthalpy_ice_freezing,
    pressure_coefficient_ice,
    pt0_from_t_ice,
    seaice_fraction_to_freeze_seawater,
)

__all__ = [
    "Helmholtz_energy_ice",
    "adiabatic_lapse_rate_ice",
    "alpha_wrt_t_ice",
    "chem_potential_water_ice",
    "cp_ice",
    "enthalpy_ice",
    "entropy_ice",
    "ice_fraction_to_freeze_seawater",
    "internal_energy_ice",
    "kappa_const_t_ice",
    "kappa_ice",
    "melting_ice_equilibrium_SA_CT_ratio",
    "melting_ice_into_seawater",
    "melting_ice_SA_CT_ratio",
    "melting_seaice_equilibrium_SA_CT_ratio",
    "melting_seaice_into_seawater",
    "melting_seaice_SA_CT_ratio",
    "pot_enthalpy_from_pt_ice",
    "pot_enthalpy_ice_freezing",
    "pressure_coefficient_ice",
    "pt0_from_t_ice",
    "seaice_fraction_to_freeze_seawater",
]
