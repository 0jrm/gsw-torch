"""
Comprehensive parity tests comparing gsw_torch with reference gsw implementation.

These tests verify numerical parity between our PyTorch implementation
and the reference NumPy-based GSW implementation.
"""

import numpy as np
import pytest
import torch

import gsw_torch

from .conftest import HAS_REFERENCE, assert_torch_allclose


@pytest.mark.skipif(not HAS_REFERENCE, reason="Reference GSW not available")
@pytest.mark.parity
class TestParityConversions:
    """Parity tests for conversion functions."""

    def test_CT_from_t_parity(self, reference_gsw, oceanographic_domain):
        """Test CT_from_t parity."""
        domain = oceanographic_domain
        SA = torch.as_tensor(domain["SA"], dtype=torch.float64)
        t = torch.as_tensor(domain["CT"], dtype=torch.float64)  # Use CT as proxy for t
        p = torch.as_tensor(domain["p"], dtype=torch.float64)
        CT_torch = gsw_torch.CT_from_t(SA, t, p)
        CT_ref = reference_gsw.CT_from_t(
            SA.detach().cpu().numpy(),
            t.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )
        assert_torch_allclose(CT_torch, CT_ref, rtol=1e-6, atol=1e-8)

    def test_t_from_CT_parity(self, reference_gsw, oceanographic_domain):
        """Test t_from_CT parity."""
        domain = oceanographic_domain
        SA = torch.as_tensor(domain["SA"], dtype=torch.float64)
        CT = torch.as_tensor(domain["CT"], dtype=torch.float64)
        p = torch.as_tensor(domain["p"], dtype=torch.float64)
        t_torch = gsw_torch.t_from_CT(SA, CT, p)
        t_ref = reference_gsw.t_from_CT(
            SA.detach().cpu().numpy(),
            CT.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )
        assert_torch_allclose(t_torch, t_ref, rtol=1e-6, atol=1e-8)

    def test_SA_from_SP_parity(self, reference_gsw):
        """Test SA_from_SP parity."""
        SP = torch.tensor([35.0], dtype=torch.float64)
        p = torch.tensor([100.0], dtype=torch.float64)
        lon = torch.tensor([188.0], dtype=torch.float64)
        lat = torch.tensor([4.0], dtype=torch.float64)

        SA_torch = gsw_torch.SA_from_SP(SP, p, lon, lat)
        SA_ref = reference_gsw.SA_from_SP(
            SP.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
            lon.detach().cpu().numpy(),
            lat.detach().cpu().numpy(),
        )
        assert_torch_allclose(SA_torch, SA_ref, rtol=1e-6, atol=1e-8)

    def test_SP_from_SA_parity(self, reference_gsw):
        """Test SP_from_SA parity."""
        SA = torch.tensor([35.0], dtype=torch.float64)
        p = torch.tensor([100.0], dtype=torch.float64)
        lon = torch.tensor([188.0], dtype=torch.float64)
        lat = torch.tensor([4.0], dtype=torch.float64)

        SP_torch = gsw_torch.SP_from_SA(SA, p, lon, lat)
        SP_ref = reference_gsw.SP_from_SA(
            SA.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
            lon.detach().cpu().numpy(),
            lat.detach().cpu().numpy(),
        )
        assert_torch_allclose(SP_torch, SP_ref, rtol=1e-6, atol=1e-8)

    def test_CT_from_pt_parity(self, reference_gsw):
        """Test CT_from_pt parity."""
        SA = torch.tensor([35.0], dtype=torch.float64)
        pt = torch.tensor([15.0], dtype=torch.float64)

        CT_torch = gsw_torch.CT_from_pt(SA, pt)
        CT_ref = reference_gsw.CT_from_pt(SA.detach().cpu().numpy(), pt.detach().cpu().numpy())
        assert_torch_allclose(CT_torch, CT_ref, rtol=1e-6, atol=1e-8)

    def test_pt_from_CT_parity(self, reference_gsw):
        """Test pt_from_CT parity."""
        SA = torch.tensor([35.0], dtype=torch.float64)
        CT = torch.tensor([15.0], dtype=torch.float64)

        pt_torch = gsw_torch.pt_from_CT(SA, CT)
        pt_ref = reference_gsw.pt_from_CT(SA.detach().cpu().numpy(), CT.detach().cpu().numpy())
        assert_torch_allclose(pt_torch, pt_ref, rtol=1e-6, atol=1e-8)

    def test_pt_from_t_parity(self, reference_gsw, oceanographic_domain):
        """Test pt_from_t parity."""
        domain = oceanographic_domain
        SA = torch.as_tensor(domain["SA"], dtype=torch.float64)
        t = torch.as_tensor(domain["CT"], dtype=torch.float64)  # Use CT as proxy
        p = torch.as_tensor(domain["p"], dtype=torch.float64)
        p_ref = torch.zeros_like(p)

        pt_torch = gsw_torch.pt_from_t(SA, t, p, p_ref)
        pt_ref_np = reference_gsw.pt_from_t(
            SA.detach().cpu().numpy(),
            t.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
            p_ref.detach().cpu().numpy(),
        )
        assert_torch_allclose(pt_torch, pt_ref_np, rtol=1e-6, atol=1e-8)

    def test_pt0_from_t_parity(self, reference_gsw, oceanographic_domain):
        """Test pt0_from_t parity."""
        domain = oceanographic_domain
        SA = torch.as_tensor(domain["SA"], dtype=torch.float64)
        t = torch.as_tensor(domain["CT"], dtype=torch.float64)  # Use CT as proxy
        p = torch.as_tensor(domain["p"], dtype=torch.float64)

        pt0_torch = gsw_torch.pt0_from_t(SA, t, p)
        pt0_ref = reference_gsw.pt0_from_t(
            SA.detach().cpu().numpy(),
            t.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )
        assert_torch_allclose(pt0_torch, pt0_ref, rtol=1e-6, atol=1e-8)

    def test_entropy_from_CT_parity(self, reference_gsw):
        """Test entropy_from_CT parity."""
        SA = torch.tensor([35.0], dtype=torch.float64)
        CT = torch.tensor([15.0], dtype=torch.float64)

        entropy_torch = gsw_torch.entropy_from_CT(SA, CT)
        entropy_ref = reference_gsw.entropy_from_CT(
            SA.detach().cpu().numpy(), CT.detach().cpu().numpy()
        )
        assert_torch_allclose(entropy_torch, entropy_ref, rtol=1e-6, atol=1e-8)

    def test_entropy_from_pt_parity(self, reference_gsw):
        """Test entropy_from_pt parity."""
        SA = torch.tensor([35.0], dtype=torch.float64)
        pt = torch.tensor([15.0], dtype=torch.float64)

        entropy_torch = gsw_torch.entropy_from_pt(SA, pt)
        entropy_ref = reference_gsw.entropy_from_pt(
            SA.detach().cpu().numpy(), pt.detach().cpu().numpy()
        )
        assert_torch_allclose(entropy_torch, entropy_ref, rtol=1e-6, atol=1e-8)

    def test_entropy_from_t_parity(self, reference_gsw, oceanographic_domain):
        """Test entropy_from_t parity."""
        domain = oceanographic_domain
        SA = torch.as_tensor(domain["SA"], dtype=torch.float64)
        t = torch.as_tensor(domain["CT"], dtype=torch.float64)  # Use CT as proxy
        p = torch.as_tensor(domain["p"], dtype=torch.float64)

        entropy_torch = gsw_torch.entropy_from_t(SA, t, p)
        entropy_ref = reference_gsw.entropy_from_t(
            SA.detach().cpu().numpy(),
            t.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )
        assert_torch_allclose(entropy_torch, entropy_ref, rtol=2e-6, atol=1e-8)

    def test_CT_from_enthalpy_parity(self, reference_gsw):
        """Test CT_from_enthalpy parity."""
        SA = torch.tensor([35.0], dtype=torch.float64)
        h = torch.tensor([40000.0], dtype=torch.float64)
        p = torch.tensor([100.0], dtype=torch.float64)

        CT_torch = gsw_torch.CT_from_enthalpy(SA, h, p)
        CT_ref = reference_gsw.CT_from_enthalpy(
            SA.detach().cpu().numpy(),
            h.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )
        assert_torch_allclose(CT_torch, CT_ref, rtol=1e-6, atol=1e-8)

    def test_CT_from_entropy_parity(self, reference_gsw):
        """Test CT_from_entropy parity."""
        SA = torch.tensor([35.0], dtype=torch.float64)
        entropy = torch.tensor([400.0], dtype=torch.float64)

        CT_torch = gsw_torch.CT_from_entropy(SA, entropy)
        CT_ref = reference_gsw.CT_from_entropy(
            SA.detach().cpu().numpy(), entropy.detach().cpu().numpy()
        )
        assert_torch_allclose(CT_torch, CT_ref, rtol=1e-6, atol=1e-8)

    def test_CT_from_rho_parity(self, reference_gsw):
        """Test CT_from_rho parity."""
        rho = torch.tensor([1026.0], dtype=torch.float64)
        SA = torch.tensor([35.0], dtype=torch.float64)
        p = torch.tensor([100.0], dtype=torch.float64)

        CT_torch = gsw_torch.CT_from_rho(rho, SA, p)
        CT_ref = reference_gsw.CT_from_rho(
            rho.detach().cpu().numpy(),
            SA.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )
        # Handle tuple return
        if isinstance(CT_ref, tuple):
            CT_ref = CT_ref[0]
        assert_torch_allclose(CT_torch, CT_ref, rtol=1e-6, atol=1e-8)

    def test_adiabatic_lapse_rate_from_CT_parity(self, reference_gsw, oceanographic_domain):
        """Test adiabatic_lapse_rate_from_CT parity."""
        domain = oceanographic_domain
        SA = torch.as_tensor(domain["SA"], dtype=torch.float64)
        CT = torch.as_tensor(domain["CT"], dtype=torch.float64)
        p = torch.as_tensor(domain["p"], dtype=torch.float64)

        lapse_torch = gsw_torch.adiabatic_lapse_rate_from_CT(SA, CT, p)
        lapse_ref = reference_gsw.adiabatic_lapse_rate_from_CT(
            SA.detach().cpu().numpy(),
            CT.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )
        assert_torch_allclose(lapse_torch, lapse_ref, rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(not HAS_REFERENCE, reason="Reference GSW not available")
@pytest.mark.parity
class TestParityDensity:
    """Parity tests for density functions."""

    def test_specvol_alpha_beta_parity(self, reference_gsw, oceanographic_domain):
        """Test specvol_alpha_beta parity."""
        domain = oceanographic_domain
        SA = torch.as_tensor(domain["SA"], dtype=torch.float64)
        CT = torch.as_tensor(domain["CT"], dtype=torch.float64)
        p = torch.as_tensor(domain["p"], dtype=torch.float64)

        specvol_torch, alpha_torch, beta_torch = gsw_torch.specvol_alpha_beta(SA, CT, p)
        specvol_ref, alpha_ref, beta_ref = reference_gsw.specvol_alpha_beta(
            SA.detach().cpu().numpy(),
            CT.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )

        assert_torch_allclose(specvol_torch, specvol_ref, rtol=1e-6, atol=1e-8)
        assert_torch_allclose(alpha_torch, alpha_ref, rtol=1e-6, atol=1e-8)
        assert_torch_allclose(beta_torch, beta_ref, rtol=1e-6, atol=1e-8)

    def test_rho_parity(self, reference_gsw, oceanographic_domain):
        """Test rho parity."""
        domain = oceanographic_domain
        SA = torch.as_tensor(domain["SA"], dtype=torch.float64)
        CT = torch.as_tensor(domain["CT"], dtype=torch.float64)
        p = torch.as_tensor(domain["p"], dtype=torch.float64)

        rho_torch = gsw_torch.rho(SA, CT, p)
        rho_ref = reference_gsw.rho(
            SA.detach().cpu().numpy(),
            CT.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )
        assert_torch_allclose(rho_torch, rho_ref, rtol=1e-6, atol=1e-8)

    def test_alpha_parity(self, reference_gsw, oceanographic_domain):
        """Test alpha parity."""
        domain = oceanographic_domain
        SA = torch.as_tensor(domain["SA"], dtype=torch.float64)
        CT = torch.as_tensor(domain["CT"], dtype=torch.float64)
        p = torch.as_tensor(domain["p"], dtype=torch.float64)

        alpha_torch = gsw_torch.alpha(SA, CT, p)
        alpha_ref = reference_gsw.alpha(
            SA.detach().cpu().numpy(),
            CT.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )
        assert_torch_allclose(alpha_torch, alpha_ref, rtol=1e-6, atol=1e-8)

    def test_beta_parity(self, reference_gsw, oceanographic_domain):
        """Test beta parity."""
        domain = oceanographic_domain
        SA = torch.as_tensor(domain["SA"], dtype=torch.float64)
        CT = torch.as_tensor(domain["CT"], dtype=torch.float64)
        p = torch.as_tensor(domain["p"], dtype=torch.float64)

        beta_torch = gsw_torch.beta(SA, CT, p)
        beta_ref = reference_gsw.beta(
            SA.detach().cpu().numpy(),
            CT.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )
        assert_torch_allclose(beta_torch, beta_ref, rtol=1e-6, atol=1e-8)

    def test_sigma0_parity(self, reference_gsw):
        """Test sigma0 parity."""
        SA = torch.tensor([35.0], dtype=torch.float64)
        CT = torch.tensor([15.0], dtype=torch.float64)

        sigma0_torch = gsw_torch.sigma0(SA, CT)
        sigma0_ref = reference_gsw.sigma0(SA.detach().cpu().numpy(), CT.detach().cpu().numpy())
        assert_torch_allclose(sigma0_torch, sigma0_ref, rtol=1e-6, atol=1e-8)

    def test_sigma1_parity(self, reference_gsw):
        """Test sigma1 parity."""
        SA = torch.tensor([35.0], dtype=torch.float64)
        CT = torch.tensor([15.0], dtype=torch.float64)

        sigma1_torch = gsw_torch.sigma1(SA, CT)
        sigma1_ref = reference_gsw.sigma1(SA.detach().cpu().numpy(), CT.detach().cpu().numpy())
        assert_torch_allclose(sigma1_torch, sigma1_ref, rtol=1e-6, atol=1e-8)

    def test_kappa_parity(self, reference_gsw, oceanographic_domain):
        """Test kappa parity."""
        domain = oceanographic_domain
        SA = torch.as_tensor(domain["SA"], dtype=torch.float64)
        CT = torch.as_tensor(domain["CT"], dtype=torch.float64)
        p = torch.as_tensor(domain["p"], dtype=torch.float64)

        kappa_torch = gsw_torch.kappa(SA, CT, p)
        kappa_ref = reference_gsw.kappa(
            SA.detach().cpu().numpy(),
            CT.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )
        assert_torch_allclose(kappa_torch, kappa_ref, rtol=1e-6, atol=1e-8)

    def test_sound_speed_parity(self, reference_gsw, oceanographic_domain):
        """Test sound_speed parity."""
        domain = oceanographic_domain
        SA = torch.as_tensor(domain["SA"], dtype=torch.float64)
        CT = torch.as_tensor(domain["CT"], dtype=torch.float64)
        p = torch.as_tensor(domain["p"], dtype=torch.float64)

        sound_speed_torch = gsw_torch.sound_speed(SA, CT, p)
        sound_speed_ref = reference_gsw.sound_speed(
            SA.detach().cpu().numpy(),
            CT.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )
        assert_torch_allclose(sound_speed_torch, sound_speed_ref, rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(not HAS_REFERENCE, reason="Reference GSW not available")
@pytest.mark.parity
class TestParityEnergy:
    """Parity tests for energy functions."""

    def test_enthalpy_parity(self, reference_gsw, oceanographic_domain):
        """Test enthalpy parity."""
        domain = oceanographic_domain
        SA = torch.as_tensor(domain["SA"], dtype=torch.float64)
        CT = torch.as_tensor(domain["CT"], dtype=torch.float64)
        p = torch.as_tensor(domain["p"], dtype=torch.float64)

        enthalpy_torch = gsw_torch.enthalpy(SA, CT, p)
        enthalpy_ref = reference_gsw.enthalpy(
            SA.detach().cpu().numpy(),
            CT.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )
        assert_torch_allclose(enthalpy_torch, enthalpy_ref, rtol=1e-6, atol=1e-8)

    def test_internal_energy_parity(self, reference_gsw, oceanographic_domain):
        """Test internal_energy parity."""
        domain = oceanographic_domain
        SA = torch.as_tensor(domain["SA"], dtype=torch.float64)
        CT = torch.as_tensor(domain["CT"], dtype=torch.float64)
        p = torch.as_tensor(domain["p"], dtype=torch.float64)

        internal_energy_torch = gsw_torch.internal_energy(SA, CT, p)
        internal_energy_ref = reference_gsw.internal_energy(
            SA.detach().cpu().numpy(),
            CT.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
        )
        assert_torch_allclose(internal_energy_torch, internal_energy_ref, rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(not HAS_REFERENCE, reason="Reference GSW not available")
@pytest.mark.parity
class TestParityFreezing:
    """Parity tests for freezing functions."""

    def test_CT_freezing_parity(self, reference_gsw):
        """Test CT_freezing parity."""
        SA = torch.tensor([35.0], dtype=torch.float64)
        p = torch.tensor([100.0], dtype=torch.float64)
        saturation_fraction = torch.tensor([1.0], dtype=torch.float64)

        CT_freeze_torch = gsw_torch.CT_freezing(SA, p, saturation_fraction)
        CT_freeze_ref = reference_gsw.CT_freezing(
            SA.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
            saturation_fraction.detach().cpu().numpy(),
        )
        assert_torch_allclose(CT_freeze_torch, CT_freeze_ref, rtol=1e-6, atol=1e-8)

    def test_t_freezing_parity(self, reference_gsw):
        """Test t_freezing parity."""
        SA = torch.tensor([35.0], dtype=torch.float64)
        p = torch.tensor([100.0], dtype=torch.float64)
        saturation_fraction = torch.tensor([1.0], dtype=torch.float64)

        t_freeze_torch = gsw_torch.t_freezing(SA, p, saturation_fraction)
        t_freeze_ref = reference_gsw.t_freezing(
            SA.detach().cpu().numpy(),
            p.detach().cpu().numpy(),
            saturation_fraction.detach().cpu().numpy(),
        )
        assert_torch_allclose(t_freeze_torch, t_freeze_ref, rtol=1e-6, atol=1e-8)
