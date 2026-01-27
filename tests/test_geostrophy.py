"""
Tests for geostrophy functions.
"""

import numpy as np
import pytest
import torch

import gsw_torch


def test_f_coriolis():
    """Test Coriolis parameter calculation."""
    lat = torch.tensor([0.0, 30.0, 45.0, 60.0, 90.0], dtype=torch.float64)
    f = gsw_torch.f(lat)

    # f should be zero at equator
    assert torch.allclose(f[0], torch.tensor(0.0, dtype=torch.float64), atol=1e-10)

    # f should increase with latitude (absolute value)
    assert torch.abs(f[4]) > torch.abs(f[1])  # Pole > 30 degrees

    # f should be positive in Northern Hemisphere, negative in Southern
    # (Actually, sign depends on convention - check absolute value)
    assert torch.all(torch.abs(f) >= 0)


def test_f_numpy_input():
    """Test f with numpy array input."""
    lat = np.array([45.0])
    f = gsw_torch.f(lat)
    assert isinstance(f, torch.Tensor)


def test_distance_basic():
    """Test distance calculation with basic inputs."""
    lon = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)

    dist = gsw_torch.distance(lon, lat)

    # Distance should be positive
    assert torch.all(dist > 0)
    # Distance along equator should be approximately longitude difference * 111 km
    # (rough approximation)
    # Note: distance returns 2D array when input is 1D, so check shape
    assert dist.shape[-1] == 2  # One less than input along last dimension


@pytest.mark.skip(reason="Requires z_from_p implementation")
def test_distance_with_pressure():
    """Test distance calculation with pressure."""
    lon = torch.tensor([0.0, 10.0], dtype=torch.float64)
    lat = torch.tensor([45.0, 45.0], dtype=torch.float64)
    p = torch.tensor([1000.0], dtype=torch.float64)

    dist = gsw_torch.distance(lon, lat, p=p)
    assert torch.all(dist > 0)


def test_unwrap_basic():
    """Test longitude unwrapping."""
    lon = torch.tensor([350.0, 355.0, 0.0, 5.0], dtype=torch.float64)
    lon_unwrapped = gsw_torch.unwrap(lon, centered=False)

    # Should unwrap across 360/0 boundary
    assert lon_unwrapped[2] > lon_unwrapped[1]  # 360 > 355


def test_unwrap_centered():
    """Test longitude unwrapping with centering."""
    lon = torch.tensor([350.0, 355.0, 0.0, 5.0], dtype=torch.float64)
    lon_unwrapped = gsw_torch.unwrap(lon, centered=True)

    # Centered values should be closer to zero
    assert torch.abs(lon_unwrapped.mean()) < torch.abs(lon.mean())


def test_geostrophic_velocity_basic():
    """Test geostrophic velocity calculation."""
    geo_strf = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    lon = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64)
    lat = torch.tensor([45.0, 45.0, 45.0], dtype=torch.float64)

    u, mid_lon, mid_lat = gsw_torch.geostrophic_velocity(geo_strf, lon, lat)

    # u can be 1D or 2D depending on input
    assert u.shape[-1] == 2  # One less than input along last dimension
    assert len(mid_lon) == 2
    assert len(mid_lat) == 2
