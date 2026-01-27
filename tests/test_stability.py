"""
Tests for stability functions.
"""

import numpy as np
import pytest
import torch

import gsw_torch


@pytest.mark.skip(reason="Requires specvol_alpha_beta implementation")
def test_Nsquared_basic():
    """Test Nsquared calculation with basic inputs."""
    SA = torch.tensor([35.0] * 10, dtype=torch.float64)
    CT = torch.linspace(20.0, 10.0, 10, dtype=torch.float64)  # Decreasing with depth
    p = torch.linspace(0, 1000, 10, dtype=torch.float64)

    N2, p_mid = gsw_torch.Nsquared(SA, CT, p)

    # N2 should be positive for stable stratification
    assert torch.all(N2 > 0)
    # p_mid should have one less element
    assert len(p_mid) == len(p) - 1


@pytest.mark.skip(reason="Requires specvol_alpha_beta implementation")
def test_Turner_Rsubrho():
    """Test Turner_Rsubrho calculation."""
    SA = torch.tensor([35.0] * 10, dtype=torch.float64)
    CT = torch.linspace(20.0, 10.0, 10, dtype=torch.float64)
    p = torch.linspace(0, 1000, 10, dtype=torch.float64)

    Tu, Rsubrho, p_mid = gsw_torch.Turner_Rsubrho(SA, CT, p)

    assert len(Tu) == len(p) - 1
    assert len(Rsubrho) == len(p) - 1
    assert len(p_mid) == len(p) - 1
