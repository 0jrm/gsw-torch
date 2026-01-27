"""
Tests for freezing functions.
"""

import numpy as np
import pytest
import torch

import gsw_torch


def test_CT_freezing_basic():
    """Test CT_freezing calculation."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)
    saturation_fraction = torch.tensor([1.0], dtype=torch.float64)

    CT_freeze = gsw_torch.CT_freezing(SA, p, saturation_fraction)

    # Freezing temperature should be negative for seawater
    assert CT_freeze.item() < 0
    # Should be around -1.9 to -2.0 degrees C for typical seawater
    assert CT_freeze.item() > -2.5
    assert CT_freeze.item() < -1.5


def test_t_freezing_basic():
    """Test t_freezing calculation."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)
    saturation_fraction = torch.tensor([1.0], dtype=torch.float64)

    t_freeze = gsw_torch.t_freezing(SA, p, saturation_fraction)

    # Freezing temperature should be negative
    assert t_freeze.item() < 0
    # Should be around -1.9 to -2.0 degrees C
    assert t_freeze.item() > -2.5
    assert t_freeze.item() < -1.5


def test_SA_freezing_from_CT():
    """Test SA_freezing_from_CT calculation."""
    CT = torch.tensor([-2.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)
    saturation_fraction = torch.tensor([1.0], dtype=torch.float64)

    SA_freeze = gsw_torch.SA_freezing_from_CT(CT, p, saturation_fraction)

    # Should be reasonable salinity value
    assert SA_freeze.item() > 0
    assert SA_freeze.item() < 50


def test_freezing_roundtrip():
    """Test that freezing functions are consistent."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)
    saturation_fraction = torch.tensor([1.0], dtype=torch.float64)

    CT_freeze = gsw_torch.CT_freezing(SA, p, saturation_fraction)
    SA_recovered = gsw_torch.SA_freezing_from_CT(CT_freeze, p, saturation_fraction)

    # Should recover original salinity approximately
    assert torch.allclose(SA, SA_recovered, rtol=1e-3, atol=1e-3)
