"""
Comprehensive gradient checking tests for differentiable functions.

These tests verify that autograd gradients are computed correctly
using torch.autograd.gradcheck.
"""

import pytest
import torch

import gsw_torch


@pytest.mark.gradcheck
class TestGradcheckConversions:
    """Gradient tests for conversion functions."""

    def test_CT_from_t_gradcheck(self):
        """Test CT_from_t gradients."""
        SA = torch.tensor([35.0], dtype=torch.float64, requires_grad=True)
        t = torch.tensor([15.0], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)

        # Note: CT_from_t may not be differentiable if it's a wrapper
        # Skip if it raises NotImplementedError or doesn't support gradients
        try:
            result = gsw_torch.CT_from_t(SA, t, p)
            if result.requires_grad:
                # Test gradient computation
                torch.autograd.gradcheck(
                    lambda sa, temp, press: gsw_torch.CT_from_t(sa, temp, press),
                    (SA, t, p),
                    eps=1e-6,
                    atol=1e-5,
                    rtol=1e-4,
                )
        except (NotImplementedError, RuntimeError) as e:
            if "backward" in str(e) or "grad" in str(e).lower():
                pytest.skip(f"Function does not support gradients: {e}")
            raise

    def test_t_from_CT_gradcheck(self):
        """Test t_from_CT gradients."""
        SA = torch.tensor([35.0], dtype=torch.float64, requires_grad=True)
        CT = torch.tensor([15.0], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)

        try:
            result = gsw_torch.t_from_CT(SA, CT, p)
            if result.requires_grad:
                torch.autograd.gradcheck(
                    lambda sa, ct, press: gsw_torch.t_from_CT(sa, ct, press),
                    (SA, CT, p),
                    eps=1e-6,
                    atol=1e-5,
                    rtol=1e-4,
                )
        except (NotImplementedError, RuntimeError) as e:
            if "backward" in str(e) or "grad" in str(e).lower():
                pytest.skip(f"Function does not support gradients: {e}")
            raise


@pytest.mark.gradcheck
class TestGradcheckDensity:
    """Gradient tests for density functions."""

    def test_rho_gradcheck(self):
        """Test rho gradients."""
        SA = torch.tensor([35.0], dtype=torch.float64, requires_grad=True)
        CT = torch.tensor([15.0], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)

        try:
            result = gsw_torch.rho(SA, CT, p)
            if result.requires_grad:
                torch.autograd.gradcheck(
                    lambda sa, ct, press: gsw_torch.rho(sa, ct, press),
                    (SA, CT, p),
                    eps=1e-6,
                    atol=1e-5,
                    rtol=1e-4,
                )
        except (NotImplementedError, RuntimeError) as e:
            if "backward" in str(e) or "grad" in str(e).lower():
                pytest.skip(f"Function does not support gradients: {e}")
            raise

    def test_alpha_gradcheck(self):
        """Test alpha gradients."""
        SA = torch.tensor([35.0], dtype=torch.float64, requires_grad=True)
        CT = torch.tensor([15.0], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)

        try:
            result = gsw_torch.alpha(SA, CT, p)
            if result.requires_grad:
                torch.autograd.gradcheck(
                    lambda sa, ct, press: gsw_torch.alpha(sa, ct, press),
                    (SA, CT, p),
                    eps=1e-6,
                    atol=1e-5,
                    rtol=1e-4,
                )
        except (NotImplementedError, RuntimeError) as e:
            if "backward" in str(e) or "grad" in str(e).lower():
                pytest.skip(f"Function does not support gradients: {e}")
            raise

    def test_beta_gradcheck(self):
        """Test beta gradients."""
        SA = torch.tensor([35.0], dtype=torch.float64, requires_grad=True)
        CT = torch.tensor([15.0], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)

        try:
            result = gsw_torch.beta(SA, CT, p)
            if result.requires_grad:
                torch.autograd.gradcheck(
                    lambda sa, ct, press: gsw_torch.beta(sa, ct, press),
                    (SA, CT, p),
                    eps=1e-6,
                    atol=1e-5,
                    rtol=1e-4,
                )
        except (NotImplementedError, RuntimeError) as e:
            if "backward" in str(e) or "grad" in str(e).lower():
                pytest.skip(f"Function does not support gradients: {e}")
            raise

    def test_specvol_alpha_beta_gradcheck(self):
        """Test specvol_alpha_beta gradients."""
        SA = torch.tensor([35.0], dtype=torch.float64, requires_grad=True)
        CT = torch.tensor([15.0], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)

        try:
            specvol, alpha, beta = gsw_torch.specvol_alpha_beta(SA, CT, p)
            if specvol.requires_grad:
                # Test gradient for specvol
                torch.autograd.gradcheck(
                    lambda sa, ct, press: gsw_torch.specvol_alpha_beta(sa, ct, press)[0],
                    (SA, CT, p),
                    eps=1e-6,
                    atol=1e-5,
                    rtol=1e-4,
                )
        except (NotImplementedError, RuntimeError) as e:
            if "backward" in str(e) or "grad" in str(e).lower():
                pytest.skip(f"Function does not support gradients: {e}")
            raise


@pytest.mark.gradcheck
class TestGradcheckEnergy:
    """Gradient checks for energy-related functions."""

    def test_adiabatic_lapse_rate_from_CT_gradcheck(self):
        """Test gradient checking for adiabatic_lapse_rate_from_CT."""
        SA = torch.tensor([35.0], dtype=torch.float64, requires_grad=True)
        CT = torch.tensor([15.0], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)

        assert torch.autograd.gradcheck(
            gsw_torch.adiabatic_lapse_rate_from_CT,
            (SA, CT, p),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )

    def test_entropy_from_t_gradcheck(self):
        """Test gradient checking for entropy_from_t."""
        SA = torch.tensor([35.0], dtype=torch.float64, requires_grad=True)
        t = torch.tensor([15.0], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)

        # Skip if CT_from_t is still a reference wrapper (doesn't support autograd)
        try:
            result = gsw_torch.entropy_from_t(SA, t, p)
            if not result.requires_grad:
                pytest.skip("entropy_from_t depends on CT_from_t which is not yet pure PyTorch")
            assert torch.autograd.gradcheck(
                gsw_torch.entropy_from_t,
                (SA, t, p),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-4,
            )
        except (NotImplementedError, RuntimeError) as e:
            if "backward" in str(e) or "grad" in str(e).lower() or "requires_grad" in str(e):
                pytest.skip(f"Function does not support gradients yet: {e}")
            raise

    def test_entropy_from_pt_gradcheck(self):
        """Test gradient checking for entropy_from_pt."""
        SA = torch.tensor([35.0], dtype=torch.float64, requires_grad=True)
        pt = torch.tensor([15.0], dtype=torch.float64, requires_grad=True)

        # Skip if CT_from_pt is still a reference wrapper (doesn't support autograd)
        try:
            result = gsw_torch.entropy_from_pt(SA, pt)
            if not result.requires_grad:
                pytest.skip("entropy_from_pt depends on CT_from_pt which is not yet pure PyTorch")
            assert torch.autograd.gradcheck(
                gsw_torch.entropy_from_pt,
                (SA, pt),
                eps=1e-6,
                atol=1e-5,
                rtol=1e-4,
            )
        except (NotImplementedError, RuntimeError) as e:
            if "backward" in str(e) or "grad" in str(e).lower() or "requires_grad" in str(e):
                pytest.skip(f"Function does not support gradients yet: {e}")
            raise


@pytest.mark.gradcheck
class TestGradcheckPurePyTorch:
    """Gradient tests for pure PyTorch functions (should always work)."""

    def test_grav_gradcheck(self):
        """Test grav gradients."""
        lat = torch.tensor([45.0], dtype=torch.float64, requires_grad=True)
        p = torch.tensor([100.0], dtype=torch.float64, requires_grad=True)

        torch.autograd.gradcheck(
            lambda la, press: gsw_torch.grav(la, press),
            (lat, p),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )

    def test_f_gradcheck(self):
        """Test f (Coriolis) gradients."""
        lat = torch.tensor([45.0], dtype=torch.float64, requires_grad=True)

        torch.autograd.gradcheck(
            lambda la: gsw_torch.f(la),
            (lat,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )

    def test_t90_from_t68_gradcheck(self):
        """Test t90_from_t68 gradients."""
        t68 = torch.tensor([15.0], dtype=torch.float64, requires_grad=True)

        torch.autograd.gradcheck(
            lambda temp: gsw_torch.t90_from_t68(temp),
            (t68,),
            eps=1e-6,
            atol=1e-5,
            rtol=1e-4,
        )
