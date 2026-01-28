"""
Numerical parity tests against reference GSW implementation.

These tests compare the PyTorch implementation with the reference
numpy-based GSW implementation to ensure numerical parity.
"""

import os

import numpy as np
import pytest
import torch

import gsw_torch

from .conftest import (
    HAS_REFERENCE,
    assert_torch_allclose,
    check_values,
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "parity: mark test as parity test")


@pytest.mark.parity
@pytest.mark.skipif(not HAS_REFERENCE, reason="Reference gsw not available")
def test_t90_from_t68_parity(reference_gsw):
    """Test t90_from_t68 against reference implementation."""
    t68 = np.array([0.0, 10.0, 20.0, 30.0])
    result_torch = gsw_torch.t90_from_t68(t68)
    result_ref = reference_gsw.t90_from_t68(t68)
    assert_torch_allclose(result_torch, result_ref)


@pytest.mark.parity
@pytest.mark.skipif(not HAS_REFERENCE, reason="Reference gsw not available")
def test_grav_parity(reference_gsw, oceanographic_domain):
    """Test grav function against reference implementation."""
    lat = oceanographic_domain["lat"]
    p = oceanographic_domain["p"]

    result_torch = gsw_torch.grav(lat, p)
    result_ref = reference_gsw.grav(lat, p)

    assert_torch_allclose(result_torch, result_ref, rtol=1e-6, atol=1e-8)


def pytest_addoption(parser):
    """Add command-line options for pytest."""
    parser.addoption(
        "--run-parity",
        action="store_true",
        default=False,
        help="run parity tests against reference implementation",
    )


# Mark all tests in this file as parity tests
pytestmark = pytest.mark.parity
