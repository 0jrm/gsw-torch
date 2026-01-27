# GSW PyTorch Comprehensive Test Report

This report contains comprehensive test results comparing the PyTorch
implementation against the reference GSW implementation.

## Executive Summary

- **Total Functions Tested**: 117
- **Functions with Errors**: 57

### Numerical Parity Tests
- **Total Tests**: 240
- **Passed**: 171
- **Failed** (error > 1e-8): 25
- **Errors**: 44

### Autograd Compatibility Tests
- **Total Tests**: 60
- **Passed**: 46
- **Failed**: 3

### Performance Benchmarks
- **Total Benchmarks**: 0

### Functions Needing Attention

**Numerical Parity Issues:**
- CT_first_derivatives
- CT_second_derivatives
- enthalpy_second_derivatives
- entropy_from_CT
- spiciness0
- spiciness1
- spiciness2

**Autograd Issues:**
- CT_freezing
- O2sol_SP_pt
- t_freezing

**Functions with Errors:**
- CT_from_enthalpy
- CT_from_entropy
- CT_from_pt
- CT_from_rho
- CT_from_t
- C_from_SP
- O2sol
- SAAR
- SA_freezing_from_CT
- SA_from_SP
- SA_from_Sstar
- SA_from_rho
- SP_from_C
- SP_from_SA
- SP_from_SK
- SP_from_SR
- SP_from_Sstar
- Sstar_from_SA
- Sstar_from_SP
- Turner_Rsubrho
- adiabatic_lapse_rate_ice
- alpha_wrt_t_ice
- chem_potential_water_ice
- chem_potential_water_t_exact
- cp_ice
- deltaSA_from_SP
- enthalpy_ice
- enthalpy_t_exact
- entropy_from_pt
- entropy_from_t
- entropy_ice
- ice_fraction_to_freeze_seawater
- internal_energy_ice
- kappa_const_t_ice
- kappa_ice
- latentheat_evap_t
- melting_ice_SA_CT_ratio
- melting_ice_equilibrium_SA_CT_ratio
- melting_ice_into_seawater
- melting_seaice_SA_CT_ratio
- melting_seaice_into_seawater
- p_from_z
- pchip_interp
- pot_enthalpy_ice_freezing
- pot_rho_t_exact
- pressure_coefficient_ice
- pressure_freezing_CT
- pt0_from_t
- pt_from_CT
- pt_from_entropy
- pt_from_t
- rho_t_exact
- seaice_fraction_to_freeze_seawater
- specvol_t_exact
- t_deriv_chem_potential_water_t_exact
- t_from_CT
- z_from_p

## Detailed Results

### Numerical Parity Results

| Function | Sample Size | Max Abs Error | Max Rel Error | Status |
|----------|-------------|---------------|---------------|--------|
| CT_first_derivatives | 10 | 0.00e+00 | 0.00e+00 | PASS |
| CT_first_derivatives | 100 | 8.94e-08 | 4.46e-16 | FAIL |
| CT_first_derivatives | 1000 | 1.49e-08 | 3.32e-16 | FAIL |
| CT_first_derivatives | 10000 | 8.94e-08 | 4.46e-16 | FAIL |
| CT_freezing | 10 | 1.08e-12 | 1.32e-11 | PASS |
| CT_freezing | 100 | 1.13e-12 | 1.32e-11 | PASS |
| CT_freezing | 1000 | 1.20e-12 | 4.98e-11 | PASS |
| CT_freezing | 10000 | 1.30e-12 | 1.98e-10 | PASS |
| CT_maxdensity | 10 | 3.35e-09 | 1.26e-09 | PASS |
| CT_maxdensity | 100 | 3.35e-09 | 1.26e-09 | PASS |
| CT_maxdensity | 1000 | 3.35e-09 | 1.26e-09 | PASS |
| CT_maxdensity | 10000 | 3.35e-09 | 1.26e-09 | PASS |
| CT_second_derivatives | 10 | 0.00e+00 | 0.00e+00 | PASS |
| CT_second_derivatives | 100 | 0.00e+00 | 0.00e+00 | PASS |
| CT_second_derivatives | 1000 | 7.45e-06 | 1.91e-10 | FAIL |
| CT_second_derivatives | 10000 | 1.49e-05 | 2.90e-10 | FAIL |
| IPV_vs_fNsquared_ratio | 10 | N/A | N/A | ERROR |
| IPV_vs_fNsquared_ratio | 100 | N/A | N/A | ERROR |
| IPV_vs_fNsquared_ratio | 1000 | N/A | N/A | ERROR |
| IPV_vs_fNsquared_ratio | 10000 | N/A | N/A | ERROR |
| Nsquared | 10 | N/A | N/A | ERROR |
| Nsquared | 100 | N/A | N/A | ERROR |
| Nsquared | 1000 | N/A | N/A | ERROR |
| Nsquared | 10000 | N/A | N/A | ERROR |
| O2sol_SP_pt | 10 | 0.00e+00 | 0.00e+00 | PASS |
| O2sol_SP_pt | 100 | 1.07e-14 | 4.05e-16 | PASS |
| O2sol_SP_pt | 1000 | 3.55e-14 | 1.40e-15 | PASS |
| O2sol_SP_pt | 10000 | 3.98e-13 | 1.33e-15 | PASS |
| SR_from_SP | 10 | 0.00e+00 | 0.00e+00 | PASS |
| SR_from_SP | 100 | 0.00e+00 | 0.00e+00 | PASS |
| SR_from_SP | 1000 | 0.00e+00 | 0.00e+00 | PASS |
| SR_from_SP | 10000 | 0.00e+00 | 0.00e+00 | PASS |
| adiabatic_lapse_rate_from_CT | 10 | 1.38e-10 | 3.89e-03 | PASS |
| adiabatic_lapse_rate_from_CT | 100 | 1.38e-10 | 3.89e-03 | PASS |
| adiabatic_lapse_rate_from_CT | 1000 | 1.38e-10 | 3.89e-03 | PASS |
| adiabatic_lapse_rate_from_CT | 10000 | 1.38e-10 | 3.89e-03 | PASS |
| alpha | 10 | 8.30e-14 | 1.37e-09 | PASS |
| alpha | 100 | 8.30e-14 | 1.37e-09 | PASS |
| alpha | 1000 | 8.30e-14 | 1.37e-09 | PASS |
| alpha | 10000 | 8.30e-14 | 1.37e-09 | PASS |
| alpha_on_beta | 10 | 7.61e-10 | 1.46e-09 | PASS |
| alpha_on_beta | 100 | 7.61e-10 | 1.46e-09 | PASS |
| alpha_on_beta | 1000 | 7.61e-10 | 1.46e-09 | PASS |
| alpha_on_beta | 10000 | 7.61e-10 | 1.46e-09 | PASS |
| beta | 10 | 1.15e-12 | 1.69e-09 | PASS |
| beta | 100 | 1.15e-12 | 1.69e-09 | PASS |
| beta | 1000 | 1.15e-12 | 1.69e-09 | PASS |
| beta | 10000 | 1.15e-12 | 1.69e-09 | PASS |
| cabbeling | 10 | N/A | N/A | ERROR |
| cabbeling | 100 | N/A | N/A | ERROR |
| cabbeling | 1000 | N/A | N/A | ERROR |
| cabbeling | 10000 | N/A | N/A | ERROR |
| distance | 10 | N/A | N/A | ERROR |
| distance | 100 | N/A | N/A | ERROR |
| distance | 1000 | N/A | N/A | ERROR |
| distance | 10000 | N/A | N/A | ERROR |
| dynamic_enthalpy | 10 | 7.28e-12 | 3.38e-16 | PASS |
| dynamic_enthalpy | 100 | 1.46e-11 | 3.92e-16 | PASS |
| dynamic_enthalpy | 1000 | 2.18e-11 | 5.83e-16 | PASS |
| dynamic_enthalpy | 10000 | 2.91e-11 | 7.02e-16 | PASS |
| enthalpy | 10 | 1.46e-11 | 2.21e-16 | PASS |
| enthalpy | 100 | 2.91e-11 | 2.56e-16 | PASS |
| enthalpy | 1000 | 2.91e-11 | 2.80e-16 | PASS |
| enthalpy | 10000 | 2.91e-11 | 3.58e-16 | PASS |
| enthalpy_SSO_0 | 10 | 7.28e-12 | 1.94e-16 | PASS |
| enthalpy_SSO_0 | 100 | 7.28e-12 | 2.98e-16 | PASS |
| enthalpy_SSO_0 | 1000 | 1.46e-11 | 3.56e-16 | PASS |
| enthalpy_SSO_0 | 10000 | 2.18e-11 | 5.36e-16 | PASS |
| enthalpy_diff | 10 | 0.00e+00 | 0.00e+00 | PASS |
| enthalpy_diff | 100 | 0.00e+00 | 0.00e+00 | PASS |
| enthalpy_diff | 1000 | 0.00e+00 | 0.00e+00 | PASS |
| enthalpy_diff | 10000 | 0.00e+00 | 0.00e+00 | PASS |
| enthalpy_first_derivatives | 10 | 1.78e-13 | 6.33e-15 | PASS |
| enthalpy_first_derivatives | 100 | 3.41e-13 | 1.40e-14 | PASS |
| enthalpy_first_derivatives | 1000 | 4.97e-13 | 1.54e-14 | PASS |
| enthalpy_first_derivatives | 10000 | 5.65e-13 | 1.91e-14 | PASS |
| enthalpy_second_derivatives | 10 | 5.93e-03 | 7.53e-02 | FAIL |
| enthalpy_second_derivatives | 100 | 5.96e-03 | 7.53e-02 | FAIL |
| enthalpy_second_derivatives | 1000 | 5.96e-03 | 7.53e-02 | FAIL |
| enthalpy_second_derivatives | 10000 | 5.96e-03 | 7.53e-02 | FAIL |
| entropy_first_derivatives | 10 | 1.78e-15 | 3.89e-16 | PASS |
| entropy_first_derivatives | 100 | 3.55e-15 | 9.79e-16 | PASS |
| entropy_first_derivatives | 1000 | 3.55e-15 | 1.03e-15 | PASS |
| entropy_first_derivatives | 10000 | 5.33e-15 | 1.07e-15 | PASS |
| entropy_from_CT | 10 | 2.57e-06 | 1.79e-06 | FAIL |
| entropy_from_CT | 100 | 2.66e-06 | 1.79e-06 | FAIL |
| entropy_from_CT | 1000 | 2.67e-06 | 1.79e-06 | FAIL |
| entropy_from_CT | 10000 | 2.67e-06 | 1.79e-06 | FAIL |
| entropy_second_derivatives | 10 | 6.94e-18 | 9.64e-16 | PASS |
| entropy_second_derivatives | 100 | 3.47e-17 | 9.89e-16 | PASS |
| entropy_second_derivatives | 1000 | 4.86e-17 | 2.60e-15 | PASS |
| entropy_second_derivatives | 10000 | 4.86e-17 | 3.13e-15 | PASS |
| f | 10 | 0.00e+00 | 0.00e+00 | PASS |
| f | 100 | 0.00e+00 | 0.00e+00 | PASS |
| f | 1000 | 1.36e-20 | 1.59e-16 | PASS |
| f | 10000 | 1.36e-20 | 2.42e-16 | PASS |
| geostrophic_velocity | 10 | 0.00e+00 | 0.00e+00 | PASS |
| geostrophic_velocity | 100 | 1.39e-17 | 2.02e-16 | PASS |
| geostrophic_velocity | 1000 | 8.88e-16 | 3.12e-16 | PASS |
| geostrophic_velocity | 10000 | 4.44e-16 | 4.33e-16 | PASS |
| grav | 10 | 0.00e+00 | 0.00e+00 | PASS |
| grav | 100 | 0.00e+00 | 0.00e+00 | PASS |
| grav | 1000 | 0.00e+00 | 0.00e+00 | PASS |
| grav | 10000 | 1.78e-15 | 1.81e-16 | PASS |
| infunnel | 10 | 0.00e+00 | 0.00e+00 | PASS |
| infunnel | 100 | 0.00e+00 | 0.00e+00 | PASS |
| infunnel | 1000 | 0.00e+00 | 0.00e+00 | PASS |
| infunnel | 10000 | 0.00e+00 | 0.00e+00 | PASS |
| internal_energy | 10 | 1.31e-10 | 1.31e-15 | PASS |
| internal_energy | 100 | 1.46e-10 | 1.47e-15 | PASS |
| internal_energy | 1000 | 1.46e-10 | 1.52e-15 | PASS |
| internal_energy | 10000 | 1.75e-10 | 6.64e-14 | PASS |
| kappa | 10 | 1.68e-22 | 1.63e-14 | PASS |
| kappa | 100 | 1.68e-22 | 1.63e-14 | PASS |
| kappa | 1000 | 1.68e-22 | 1.63e-14 | PASS |
| kappa | 10000 | 1.68e-22 | 1.63e-14 | PASS |
| latentheat_evap_CT | 10 | 0.00e+00 | 0.00e+00 | PASS |
| latentheat_evap_CT | 100 | 0.00e+00 | 0.00e+00 | PASS |
| latentheat_evap_CT | 1000 | 0.00e+00 | 0.00e+00 | PASS |
| latentheat_evap_CT | 10000 | 0.00e+00 | 0.00e+00 | PASS |
| latentheat_melting | 10 | 4.66e-10 | 1.44e-15 | PASS |
| latentheat_melting | 100 | 6.40e-10 | 2.00e-15 | PASS |
| latentheat_melting | 1000 | 6.40e-10 | 2.00e-15 | PASS |
| latentheat_melting | 10000 | 9.31e-10 | 2.88e-15 | PASS |
| pt_first_derivatives | 10 | 5.55e-16 | 2.48e-15 | PASS |
| pt_first_derivatives | 100 | 6.66e-16 | 2.73e-15 | PASS |
| pt_first_derivatives | 1000 | 8.88e-16 | 4.32e-15 | PASS |
| pt_first_derivatives | 10000 | 8.88e-16 | 5.24e-15 | PASS |
| pt_second_derivatives | 10 | 4.54e-11 | 5.49e-06 | PASS |
| pt_second_derivatives | 100 | 4.54e-11 | 3.73e-04 | PASS |
| pt_second_derivatives | 1000 | 4.54e-11 | 2.33e-04 | PASS |
| pt_second_derivatives | 10000 | 4.54e-11 | 1.11e-03 | PASS |
| rho | 10 | 2.96e-12 | 2.82e-15 | PASS |
| rho | 100 | 2.96e-12 | 2.82e-15 | PASS |
| rho | 1000 | 3.18e-12 | 3.04e-15 | PASS |
| rho | 10000 | 3.18e-12 | 3.04e-15 | PASS |
| rho_alpha_beta | 10 | 2.96e-12 | 1.69e-09 | PASS |
| rho_alpha_beta | 100 | 2.96e-12 | 1.69e-09 | PASS |
| rho_alpha_beta | 1000 | 3.18e-12 | 1.69e-09 | PASS |
| rho_alpha_beta | 10000 | 3.18e-12 | 1.69e-09 | PASS |
| rho_first_derivatives | 10 | 1.20e-09 | 1.69e-09 | PASS |
| rho_first_derivatives | 100 | 1.20e-09 | 1.69e-09 | PASS |
| rho_first_derivatives | 1000 | 1.20e-09 | 1.69e-09 | PASS |
| rho_first_derivatives | 10000 | 1.20e-09 | 1.69e-09 | PASS |
| rho_first_derivatives_wrt_enthalpy | 10 | 1.20e-09 | 1.70e-09 | PASS |
| rho_first_derivatives_wrt_enthalpy | 100 | 1.20e-09 | 1.70e-09 | PASS |
| rho_first_derivatives_wrt_enthalpy | 1000 | 1.20e-09 | 1.70e-09 | PASS |
| rho_first_derivatives_wrt_enthalpy | 10000 | 1.20e-09 | 1.70e-09 | PASS |
| rho_second_derivatives | 10 | N/A | N/A | ERROR |
| rho_second_derivatives | 100 | N/A | N/A | ERROR |
| rho_second_derivatives | 1000 | N/A | N/A | ERROR |
| rho_second_derivatives | 10000 | N/A | N/A | ERROR |
| rho_second_derivatives_wrt_enthalpy | 10 | N/A | N/A | ERROR |
| rho_second_derivatives_wrt_enthalpy | 100 | N/A | N/A | ERROR |
| rho_second_derivatives_wrt_enthalpy | 1000 | N/A | N/A | ERROR |
| rho_second_derivatives_wrt_enthalpy | 10000 | N/A | N/A | ERROR |
| sa_ct_interp | 10 | N/A | N/A | ERROR |
| sa_ct_interp | 100 | N/A | N/A | ERROR |
| sa_ct_interp | 1000 | N/A | N/A | ERROR |
| sa_ct_interp | 10000 | N/A | N/A | ERROR |
| sigma0 | 10 | 4.55e-13 | 1.72e-14 | PASS |
| sigma0 | 100 | 4.55e-13 | 1.72e-14 | PASS |
| sigma0 | 1000 | 4.55e-13 | 1.81e-14 | PASS |
| sigma0 | 10000 | 4.55e-13 | 1.89e-14 | PASS |
| sigma1 | 10 | 2.27e-13 | 7.92e-15 | PASS |
| sigma1 | 100 | 2.27e-13 | 7.92e-15 | PASS |
| sigma1 | 1000 | 4.55e-13 | 1.56e-14 | PASS |
| sigma1 | 10000 | 4.55e-13 | 1.58e-14 | PASS |
| sigma2 | 10 | 4.55e-13 | 1.36e-14 | PASS |
| sigma2 | 100 | 4.55e-13 | 1.36e-14 | PASS |
| sigma2 | 1000 | 6.82e-13 | 1.94e-14 | PASS |
| sigma2 | 10000 | 6.82e-13 | 2.02e-14 | PASS |
| sigma3 | 10 | 9.09e-13 | 2.40e-14 | PASS |
| sigma3 | 100 | 9.09e-13 | 2.40e-14 | PASS |
| sigma3 | 1000 | 1.14e-12 | 2.90e-14 | PASS |
| sigma3 | 10000 | 1.14e-12 | 3.00e-14 | PASS |
| sigma4 | 10 | 1.82e-12 | 4.19e-14 | PASS |
| sigma4 | 100 | 1.82e-12 | 4.27e-14 | PASS |
| sigma4 | 1000 | 2.05e-12 | 4.74e-14 | PASS |
| sigma4 | 10000 | 2.05e-12 | 4.82e-14 | PASS |
| sound_speed | 10 | 3.80e-10 | 2.34e-13 | PASS |
| sound_speed | 100 | 3.80e-10 | 2.34e-13 | PASS |
| sound_speed | 1000 | 3.80e-10 | 2.34e-13 | PASS |
| sound_speed | 10000 | 3.80e-10 | 2.34e-13 | PASS |
| specvol | 10 | 2.71e-18 | 2.84e-15 | PASS |
| specvol | 100 | 2.71e-18 | 2.84e-15 | PASS |
| specvol | 1000 | 2.93e-18 | 3.06e-15 | PASS |
| specvol | 10000 | 2.93e-18 | 3.06e-15 | PASS |
| specvol_alpha_beta | 10 | 1.15e-12 | 1.69e-09 | PASS |
| specvol_alpha_beta | 100 | 1.15e-12 | 1.69e-09 | PASS |
| specvol_alpha_beta | 1000 | 1.15e-12 | 1.69e-09 | PASS |
| specvol_alpha_beta | 10000 | 1.15e-12 | 1.69e-09 | PASS |
| specvol_anom_standard | 10 | 3.25e-19 | 1.22e-13 | PASS |
| specvol_anom_standard | 100 | 4.34e-19 | 1.46e-13 | PASS |
| specvol_anom_standard | 1000 | 4.34e-19 | 1.80e-13 | PASS |
| specvol_anom_standard | 10000 | 5.42e-19 | 2.16e-13 | PASS |
| specvol_first_derivatives | 10 | 1.10e-15 | 1.66e-09 | PASS |
| specvol_first_derivatives | 100 | 1.10e-15 | 1.66e-09 | PASS |
| specvol_first_derivatives | 1000 | 1.10e-15 | 1.66e-09 | PASS |
| specvol_first_derivatives | 10000 | 1.10e-15 | 1.66e-09 | PASS |
| specvol_first_derivatives_wrt_enthalpy | 10 | 1.10e-15 | 1.67e-09 | PASS |
| specvol_first_derivatives_wrt_enthalpy | 100 | 1.10e-15 | 1.67e-09 | PASS |
| specvol_first_derivatives_wrt_enthalpy | 1000 | 1.10e-15 | 1.67e-09 | PASS |
| specvol_first_derivatives_wrt_enthalpy | 10000 | 1.10e-15 | 1.67e-09 | PASS |
| specvol_second_derivatives | 10 | N/A | N/A | ERROR |
| specvol_second_derivatives | 100 | N/A | N/A | ERROR |
| specvol_second_derivatives | 1000 | N/A | N/A | ERROR |
| specvol_second_derivatives | 10000 | N/A | N/A | ERROR |
| specvol_second_derivatives_wrt_enthalpy | 10 | N/A | N/A | ERROR |
| specvol_second_derivatives_wrt_enthalpy | 100 | N/A | N/A | ERROR |
| specvol_second_derivatives_wrt_enthalpy | 1000 | N/A | N/A | ERROR |
| specvol_second_derivatives_wrt_enthalpy | 10000 | N/A | N/A | ERROR |
| spiciness0 | 10 | 9.74e-06 | 6.08e-06 | FAIL |
| spiciness0 | 100 | 9.74e-06 | 9.93e-05 | FAIL |
| spiciness0 | 1000 | 9.74e-06 | 8.78e-03 | FAIL |
| spiciness0 | 10000 | 9.74e-06 | 1.96e-03 | FAIL |
| spiciness1 | 10 | 9.75e-06 | 2.04e-05 | FAIL |
| spiciness1 | 100 | 9.75e-06 | 1.19e-04 | FAIL |
| spiciness1 | 1000 | 9.75e-06 | 1.14e-03 | FAIL |
| spiciness1 | 10000 | 9.75e-06 | 6.52e-03 | FAIL |
| spiciness2 | 10 | 9.75e-06 | 1.60e-05 | FAIL |
| spiciness2 | 100 | 9.75e-06 | 2.57e-04 | FAIL |
| spiciness2 | 1000 | 9.75e-06 | 2.24e-03 | FAIL |
| spiciness2 | 10000 | 9.75e-06 | 1.41e-02 | FAIL |
| t90_from_t68 | 10 | 0.00e+00 | 0.00e+00 | PASS |
| t90_from_t68 | 100 | 0.00e+00 | 0.00e+00 | PASS |
| t90_from_t68 | 1000 | 0.00e+00 | 0.00e+00 | PASS |
| t90_from_t68 | 10000 | 0.00e+00 | 0.00e+00 | PASS |
| t_freezing | 10 | 1.19e-12 | 8.90e-11 | PASS |
| t_freezing | 100 | 1.24e-12 | 8.90e-11 | PASS |
| t_freezing | 1000 | 1.32e-12 | 8.90e-11 | PASS |
| t_freezing | 10000 | 1.42e-12 | 3.12e-10 | PASS |
| thermobaric | 10 | N/A | N/A | ERROR |
| thermobaric | 100 | N/A | N/A | ERROR |
| thermobaric | 1000 | N/A | N/A | ERROR |
| thermobaric | 10000 | N/A | N/A | ERROR |
| tracer_ct_interp | 10 | N/A | N/A | ERROR |
| tracer_ct_interp | 100 | N/A | N/A | ERROR |
| tracer_ct_interp | 1000 | N/A | N/A | ERROR |
| tracer_ct_interp | 10000 | N/A | N/A | ERROR |

### Autograd Compatibility Results

| Function | Gradients Work | Num Differentiable Inputs | Issues |
|----------|----------------|---------------------------|--------|
| CT_first_derivatives | Yes | 2 | None |
| CT_freezing | No | 3 | Non-finite gradients for input 0; Non-finite gradients for input 1; Non-finite gradients for input 2 |
| CT_maxdensity | Yes | 2 | None |
| CT_second_derivatives | Yes | 2 | None |
| IPV_vs_fNsquared_ratio | No | 0 | None |
| Nsquared | No | 0 | None |
| O2sol_SP_pt | No | 2 | Non-finite gradients for input 0; Non-finite gradients for input 1 |
| SR_from_SP | Yes | 1 | None |
| adiabatic_lapse_rate_from_CT | Yes | 3 | None |
| alpha | Yes | 3 | None |
| alpha_on_beta | Yes | 3 | None |
| beta | Yes | 3 | None |
| cabbeling | No | 0 | None |
| distance | No | 0 | None |
| dynamic_enthalpy | Yes | 3 | None |
| enthalpy | Yes | 3 | None |
| enthalpy_SSO_0 | Yes | 1 | None |
| enthalpy_diff | Yes | 4 | None |
| enthalpy_first_derivatives | Yes | 0 | None |
| enthalpy_second_derivatives | Yes | 3 | None |
| entropy_first_derivatives | Yes | 2 | None |
| entropy_from_CT | Yes | 2 | None |
| entropy_second_derivatives | Yes | 2 | None |
| f | Yes | 1 | None |
| geostrophic_velocity | Yes | 3 | None |
| grav | Yes | 2 | None |
| infunnel | Yes | 0 | None |
| internal_energy | Yes | 3 | None |
| kappa | Yes | 3 | None |
| latentheat_evap_CT | Yes | 2 | None |
| latentheat_melting | Yes | 2 | None |
| pt_first_derivatives | Yes | 0 | None |
| pt_second_derivatives | Yes | 0 | None |
| rho | Yes | 3 | None |
| rho_alpha_beta | Yes | 3 | None |
| rho_first_derivatives | Yes | 3 | None |
| rho_first_derivatives_wrt_enthalpy | Yes | 3 | None |
| rho_second_derivatives | No | 0 | None |
| rho_second_derivatives_wrt_enthalpy | No | 0 | None |
| sa_ct_interp | No | 0 | None |
| sigma0 | Yes | 2 | None |
| sigma1 | Yes | 2 | None |
| sigma2 | Yes | 2 | None |
| sigma3 | Yes | 2 | None |
| sigma4 | Yes | 2 | None |
| sound_speed | Yes | 3 | None |
| specvol | Yes | 3 | None |
| specvol_alpha_beta | Yes | 3 | None |
| specvol_anom_standard | Yes | 3 | None |
| specvol_first_derivatives | Yes | 3 | None |
| specvol_first_derivatives_wrt_enthalpy | Yes | 3 | None |
| specvol_second_derivatives | No | 0 | None |
| specvol_second_derivatives_wrt_enthalpy | No | 0 | None |
| spiciness0 | Yes | 2 | None |
| spiciness1 | Yes | 2 | None |
| spiciness2 | Yes | 2 | None |
| t90_from_t68 | Yes | 1 | None |
| t_freezing | No | 3 | Non-finite gradients for input 0 |
| thermobaric | No | 0 | None |
| tracer_ct_interp | No | 0 | None |

### Performance Benchmarks

| Function | Sample Size | PyTorch Time (s) | Reference Time (s) | Speedup |
|----------|-------------|------------------|-------------------|---------|

## Per-Module Breakdown

| Module | Total Functions | Parity Pass | Parity Fail | Autograd Pass | Autograd Fail |
|--------|----------------|-------------|-------------|--------------|---------------|
| conversions | 22 | 63 | 9 | 15 | 3 |
| density | 15 | 52 | 0 | 13 | 0 |
| energy | 6 | 20 | 4 | 6 | 0 |
| geostrophy | 2 | 4 | 0 | 1 | 0 |
| stability | 2 | 0 | 0 | 0 | 0 |
| utility | 13 | 32 | 12 | 11 | 0 |
