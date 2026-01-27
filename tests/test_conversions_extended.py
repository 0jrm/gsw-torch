"""
Extended tests for conversion functions.
"""

import numpy as np
import pytest
import torch

import gsw_torch


def test_CT_from_t_basic():
    """Test CT_from_t with basic inputs."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    t = torch.tensor([15.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)

    CT = gsw_torch.CT_from_t(SA, t, p)

    # Conservative Temperature should be close to in-situ temperature
    # at low pressures, but slightly different
    assert isinstance(CT, torch.Tensor)
    assert torch.abs(CT - t) < 1.0  # Should be within 1 degree


def test_CT_from_t_numpy_input():
    """Test CT_from_t with numpy array inputs."""
    SA = np.array([35.0])
    t = np.array([15.0])
    p = np.array([100.0])

    CT = gsw_torch.CT_from_t(SA, t, p)
    assert isinstance(CT, torch.Tensor)


def test_SA_from_SP_basic():
    """Test SA_from_SP with basic inputs."""
    SP = torch.tensor([35.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)
    lon = torch.tensor([0.0], dtype=torch.float64)
    lat = torch.tensor([0.0], dtype=torch.float64)

    SA = gsw_torch.SA_from_SP(SP, p, lon, lat)

    # Absolute Salinity should be close to Practical Salinity
    # but may differ slightly due to composition (typically 0.1-0.2 g/kg)
    assert isinstance(SA, torch.Tensor)
    assert torch.abs(SA - SP) < 0.5  # Should be within 0.5 g/kg


def test_SP_from_SA_basic():
    """Test SP_from_SA with basic inputs."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)
    lon = torch.tensor([0.0], dtype=torch.float64)
    lat = torch.tensor([0.0], dtype=torch.float64)

    SP = gsw_torch.SP_from_SA(SA, p, lon, lat)

    assert isinstance(SP, torch.Tensor)
    # Practical Salinity should be close to Absolute Salinity
    # (typically differs by 0.1-0.2 g/kg)
    assert torch.abs(SP - SA) < 0.5


def test_t_from_CT_basic():
    """Test t_from_CT with basic inputs."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([15.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)

    t = gsw_torch.t_from_CT(SA, CT, p)

    assert isinstance(t, torch.Tensor)
    # In-situ temperature should be close to Conservative Temperature
    assert torch.abs(t - CT) < 1.0


def test_CT_t_roundtrip():
    """Test that CT_from_t and t_from_CT are inverse operations."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    t_orig = torch.tensor([15.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)

    CT = gsw_torch.CT_from_t(SA, t_orig, p)
    t_recovered = gsw_torch.t_from_CT(SA, CT, p)

    # Should recover original temperature approximately
    assert torch.allclose(t_orig, t_recovered, rtol=1e-4, atol=1e-4)


def test_SA_SP_roundtrip():
    """Test that SA_from_SP and SP_from_SA are inverse operations."""
    SP_orig = torch.tensor([35.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)
    lon = torch.tensor([0.0], dtype=torch.float64)
    lat = torch.tensor([0.0], dtype=torch.float64)

    SA = gsw_torch.SA_from_SP(SP_orig, p, lon, lat)
    SP_recovered = gsw_torch.SP_from_SA(SA, p, lon, lat)

    # Should recover original Practical Salinity approximately
    assert torch.allclose(SP_orig, SP_recovered, rtol=1e-4, atol=1e-4)
