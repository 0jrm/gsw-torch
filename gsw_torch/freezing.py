"""
Freezing-point functions.
"""
# Import from _core modules
from ._core.freezing import (
    CT_freezing,
    CT_freezing_first_derivatives,
    CT_freezing_first_derivatives_poly,
    CT_freezing_poly,
    SA_freezing_from_CT,
    SA_freezing_from_CT_poly,
    SA_freezing_from_t,
    SA_freezing_from_t_poly,
    pressure_freezing_CT,
    t_freezing,
    t_freezing_first_derivatives,
    t_freezing_first_derivatives_poly,
)

__all__ = [
    "CT_freezing",
    "CT_freezing_first_derivatives",
    "CT_freezing_poly",
    "SA_freezing_from_CT",
    "SA_freezing_from_t",
    "pressure_freezing_CT",
    "t_freezing",
    "t_freezing_first_derivatives",
]
