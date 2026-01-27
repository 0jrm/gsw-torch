"""
Integration tests that test multiple functions working together.
"""

import torch

import gsw_torch


def test_stability_workflow():
    """Test a typical oceanographic workflow using multiple functions."""
    # Create oceanographic profile data
    SA = torch.tensor([35.0] * 20, dtype=torch.float64)
    t = torch.linspace(25.0, 5.0, 20, dtype=torch.float64)  # Temperature decreasing with depth
    p = torch.linspace(0, 2000, 20, dtype=torch.float64)  # Pressure increasing

    # Convert to Conservative Temperature
    CT = gsw_torch.CT_from_t(SA, t, p)

    # Calculate specific volume and expansion coefficients
    specvol, alpha, beta = gsw_torch.specvol_alpha_beta(SA, CT, p)

    # Calculate buoyancy frequency
    N2, p_mid = gsw_torch.Nsquared(SA, CT, p)

    # Check that results are reasonable
    assert len(CT) == 20
    assert len(specvol) == 20
    assert len(N2) == 19  # One less than input
    assert len(p_mid) == 19

    # N2 should be positive for stable stratification (temperature decreasing)
    assert torch.all(N2 > 0)


def test_geostrophic_workflow():
    """Test geostrophic calculation workflow."""
    # Create section data
    lon = torch.tensor([0.0, 10.0, 20.0, 30.0], dtype=torch.float64)
    lat = torch.tensor([45.0, 45.0, 45.0, 45.0], dtype=torch.float64)

    # Calculate distances
    dist = gsw_torch.distance(lon, lat)

    # Calculate Coriolis parameter
    f = gsw_torch.f(lat)

    # Check results
    # distance returns 2D array when input is 1D (shape [1, n-1])
    assert dist.shape[-1] == 3  # One less than input along last dimension
    assert len(f) == 4
    assert torch.all(dist > 0)
    assert torch.all(torch.abs(f) > 0)


def test_conversion_roundtrips():
    """Test that conversion functions are properly inverse."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    t = torch.tensor([15.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)

    # CT <-> t conversion
    CT = gsw_torch.CT_from_t(SA, t, p)
    t_recovered = gsw_torch.t_from_CT(SA, CT, p)
    assert torch.allclose(t, t_recovered, rtol=1e-4, atol=1e-4)

    # SA <-> SP conversion (at equator)
    SP = torch.tensor([35.0], dtype=torch.float64)
    lon = torch.tensor([0.0], dtype=torch.float64)
    lat = torch.tensor([0.0], dtype=torch.float64)

    SA_calc = gsw_torch.SA_from_SP(SP, p, lon, lat)
    SP_recovered = gsw_torch.SP_from_SA(SA_calc, p, lon, lat)
    assert torch.allclose(SP, SP_recovered, rtol=1e-4, atol=1e-4)
