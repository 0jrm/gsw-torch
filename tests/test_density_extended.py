"""
Extended tests for density functions.
"""

import numpy as np
import pytest
import torch

import gsw_torch


def test_rho_basic():
    """Test rho calculation."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([15.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)

    rho_val = gsw_torch.rho(SA, CT, p)

    # Density should be around 1025-1027 kg/m³ for typical seawater
    assert rho_val.item() > 1020
    assert rho_val.item() < 1030


def test_rho_from_specvol():
    """Test that rho = 1/specvol."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([15.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)

    specvol_val, _, _ = gsw_torch.specvol_alpha_beta(SA, CT, p)
    rho_val = gsw_torch.rho(SA, CT, p)

    # Should be inverse relationship
    assert torch.allclose(rho_val, 1.0 / specvol_val, rtol=1e-10)


def test_alpha_beta_basic():
    """Test alpha and beta calculations."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([15.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)

    alpha_val = gsw_torch.alpha(SA, CT, p)
    beta_val = gsw_torch.beta(SA, CT, p)

    # Both should be positive
    assert alpha_val.item() > 0
    assert beta_val.item() > 0

    # Check against specvol_alpha_beta
    _, alpha_check, beta_check = gsw_torch.specvol_alpha_beta(SA, CT, p)
    assert torch.allclose(alpha_val, alpha_check)
    assert torch.allclose(beta_val, beta_check)


def test_specvol_basic():
    """Test specvol calculation."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([15.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)

    specvol_val = gsw_torch.specvol(SA, CT, p)

    # Should match specvol_alpha_beta
    specvol_check, _, _ = gsw_torch.specvol_alpha_beta(SA, CT, p)
    assert torch.allclose(specvol_val, specvol_check)


def test_rho_alpha_beta():
    """Test rho_alpha_beta function."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([15.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)

    rho_val, alpha_val, beta_val = gsw_torch.rho_alpha_beta(SA, CT, p)

    # Check individual functions
    rho_check = gsw_torch.rho(SA, CT, p)
    alpha_check = gsw_torch.alpha(SA, CT, p)
    beta_check = gsw_torch.beta(SA, CT, p)

    assert torch.allclose(rho_val, rho_check)
    assert torch.allclose(alpha_val, alpha_check)
    assert torch.allclose(beta_val, beta_check)


def test_alpha_on_beta():
    """Test alpha_on_beta calculation."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([15.0], dtype=torch.float64)
    p = torch.tensor([100.0], dtype=torch.float64)

    ratio = gsw_torch.alpha_on_beta(SA, CT, p)
    alpha_val = gsw_torch.alpha(SA, CT, p)
    beta_val = gsw_torch.beta(SA, CT, p)

    assert torch.allclose(ratio, alpha_val / beta_val)


def test_sigma0():
    """Test sigma0 calculation."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([15.0], dtype=torch.float64)

    sigma0_val = gsw_torch.sigma0(SA, CT)

    # Should be rho at p=0 - 1000
    rho_0 = gsw_torch.rho(SA, CT, torch.tensor([0.0], dtype=torch.float64))
    assert torch.allclose(sigma0_val, rho_0 - 1000.0)


def test_sigma1():
    """Test sigma1 calculation."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([15.0], dtype=torch.float64)

    sigma1_val = gsw_torch.sigma1(SA, CT)

    # Should be rho at p=1000 - 1000
    rho_1000 = gsw_torch.rho(SA, CT, torch.tensor([1000.0], dtype=torch.float64))
    assert torch.allclose(sigma1_val, rho_1000 - 1000.0)


def test_sigma2():
    """Test sigma2 calculation."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([15.0], dtype=torch.float64)

    sigma2_val = gsw_torch.sigma2(SA, CT)
    rho_2000 = gsw_torch.rho(SA, CT, torch.tensor([2000.0], dtype=torch.float64))
    assert torch.allclose(sigma2_val, rho_2000 - 1000.0)


def test_sigma3():
    """Test sigma3 calculation."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([15.0], dtype=torch.float64)

    sigma3_val = gsw_torch.sigma3(SA, CT)
    rho_3000 = gsw_torch.rho(SA, CT, torch.tensor([3000.0], dtype=torch.float64))
    assert torch.allclose(sigma3_val, rho_3000 - 1000.0)


def test_sigma4():
    """Test sigma4 calculation."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([15.0], dtype=torch.float64)

    sigma4_val = gsw_torch.sigma4(SA, CT)
    rho_4000 = gsw_torch.rho(SA, CT, torch.tensor([4000.0], dtype=torch.float64))
    assert torch.allclose(sigma4_val, rho_4000 - 1000.0)


def test_sigma_ordering():
    """Test that sigma values increase with reference pressure."""
    SA = torch.tensor([35.0], dtype=torch.float64)
    CT = torch.tensor([15.0], dtype=torch.float64)

    sigma0_val = gsw_torch.sigma0(SA, CT)
    sigma1_val = gsw_torch.sigma1(SA, CT)
    sigma2_val = gsw_torch.sigma2(SA, CT)
    sigma3_val = gsw_torch.sigma3(SA, CT)
    sigma4_val = gsw_torch.sigma4(SA, CT)

    # Higher pressure should give higher density (more positive sigma)
    assert sigma1_val.item() > sigma0_val.item()
    assert sigma2_val.item() > sigma1_val.item()
    assert sigma3_val.item() > sigma2_val.item()
    assert sigma4_val.item() > sigma3_val.item()
