"""
Tests for density functions.
"""

import numpy as np
import pytest
import torch

import gsw_torch


@pytest.mark.skipif(
    not hasattr(gsw_torch, "specvol_alpha_beta"),
    reason="specvol_alpha_beta not available",
)
def test_specvol_alpha_beta_basic():
    """Test specvol_alpha_beta with basic inputs."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([15.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)

    specvol, alpha, beta = gsw_torch.specvol_alpha_beta(SA, CT, p)

    # Check that results are reasonable
    assert specvol.item() > 0.0009  # Specific volume ~0.00097 m^3/kg
    assert specvol.item() < 0.0010
    assert alpha.item() > 0  # Thermal expansion coefficient positive
    assert beta.item() > 0  # Saline contraction coefficient positive


@pytest.mark.skipif(
    not hasattr(gsw_torch, "specvol_alpha_beta"),
    reason="specvol_alpha_beta not available",
)
def test_specvol_alpha_beta_numpy_input():
    """Test specvol_alpha_beta with numpy array inputs."""
    SA = np.array([35.0])
    CT = np.array([15.0])
    p = np.array([100.0])

    specvol, alpha, beta = gsw_torch.specvol_alpha_beta(SA, CT, p)

    assert isinstance(specvol, torch.Tensor)
    assert isinstance(alpha, torch.Tensor)
    assert isinstance(beta, torch.Tensor)


@pytest.mark.skipif(
    not hasattr(gsw_torch, "specvol_alpha_beta"),
    reason="specvol_alpha_beta not available",
)
def test_specvol_alpha_beta_broadcasting():
    """Test specvol_alpha_beta with broadcasting."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([10.0, 15.0, 20.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)

    specvol, alpha, beta = gsw_torch.specvol_alpha_beta(SA, CT, p)

    assert specvol.shape == (3,)
    assert alpha.shape == (3,)
    assert beta.shape == (3,)
