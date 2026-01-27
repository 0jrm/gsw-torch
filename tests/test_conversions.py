"""
Tests for conversion functions.
"""

import numpy as np
import pytest
import torch

import gsw_torch


def test_t90_from_t68():
    """Test t90_from_t68 conversion."""
    t68 = torch.tensor([0.0, 10.0, 20.0, 30.0], dtype=torch.float64)
    t90 = gsw_torch.t90_from_t68(t68)
    expected = t68 / 1.00024
    assert torch.allclose(t90, expected)


def test_t90_from_t68_numpy_input():
    """Test t90_from_t68 with numpy input."""
    t68 = np.array([0.0, 10.0, 20.0])
    t90 = gsw_torch.t90_from_t68(t68)
    assert isinstance(t90, torch.Tensor)
    expected = torch.tensor(t68 / 1.00024, dtype=torch.float64)
    assert torch.allclose(t90, expected)


def test_grav_basic():
    """Test grav function with basic inputs."""
    lat = torch.tensor([0.0, 45.0, 90.0], dtype=torch.float64)
    p = torch.tensor([0.0, 1000.0, 5000.0], dtype=torch.float64)
    g = gsw_torch.grav(lat, p)

    # Check that gravity is reasonable (around 9.8 m/s^2)
    assert torch.all(g > 9.7)
    assert torch.all(g < 9.9)
    # Check that gravity increases with latitude (at poles)
    assert g[2] > g[0]  # Pole > equator


@pytest.mark.skip(reason="p_from_z and z_from_p need full implementation")
def test_p_z_roundtrip():
    """Test that p_from_z and z_from_p are inverse operations."""
    z = torch.tensor([-100.0, -500.0, -1000.0], dtype=torch.float64)
    lat = torch.tensor([45.0], dtype=torch.float64)
    p = gsw_torch.p_from_z(z, lat)
    z_recovered = gsw_torch.z_from_p(p, lat)
    # Should be approximately equal (within numerical precision)
    assert torch.allclose(z, z_recovered, rtol=1e-3)
