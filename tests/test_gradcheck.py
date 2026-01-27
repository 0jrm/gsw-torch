"""
Gradient checking tests using torch.autograd.gradcheck.

These tests verify that autograd gradients are computed correctly
for differentiable functions.
"""

import pytest
import torch

import gsw_torch


@pytest.mark.skip(reason="Requires differentiable implementations")
def test_t90_from_t68_gradcheck():
    """Test gradients for t90_from_t68."""
    t68 = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(gsw_torch.t90_from_t68, t68, eps=1e-6, atol=1e-5)


@pytest.mark.skip(reason="Requires differentiable implementations")
def test_grav_gradcheck():
    """Test gradients for grav function."""
    lat = torch.tensor([45.0], dtype=torch.float64, requires_grad=True)
    p = torch.tensor([1000.0], dtype=torch.float64, requires_grad=True)

    def grav_wrapper(lat, p):
        return gsw_torch.grav(lat, p)

    # Note: grav may not be differentiable in all inputs
    # This test will need to be adjusted based on actual implementation
    assert torch.autograd.gradcheck(
        grav_wrapper, (lat, p), eps=1e-6, atol=1e-5, check_undefined_grad=False
    )
