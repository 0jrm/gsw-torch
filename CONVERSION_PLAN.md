# Conversion Plan: Reference Wrappers → Pure PyTorch

## Current Status

**Total Functions**: 114 implemented
- **Pure PyTorch**: ~10 functions (grav, t90_from_t68, f, distance, unwrap, geostrophic_velocity, and derived functions)
- **Reference Wrappers**: ~75 functions (call NumPy GSW via `_call_reference()`)
- **Missing**: ~66 functions

## Critical Issue

**ALL reference wrappers must be converted to pure PyTorch** because:
1. ❌ No GPU support (converts to CPU NumPy)
2. ❌ No proper autograd (can't backprop through NumPy)
3. ❌ Not a true PyTorch implementation

## Implementation Priority

### Phase 1: Foundation (CRITICAL - Blocks everything else)
1. ✅ **specvol_alpha_beta** - 75-term polynomial (Roquet et al. 2015) - **COMPLETED**
   - Pure PyTorch implementation with all 75 coefficients
   - Perfect numerical parity with reference
   - Autograd support verified
   - GPU compatible

### Phase 2: Core Conversions (High Priority)
2. **CT_from_t** / **t_from_CT** - Conservative Temperature conversions
3. **SA_from_SP** / **SP_from_SA** - Salinity conversions
4. ✅ **pt0_from_t** - Potential temperature (simple wrapper) - **COMPLETED**
5. **pt_from_t** / **pt_from_CT** - Potential temperature
6. ✅ **entropy_from_t** / **entropy_from_pt** - Entropy (simple wrappers) - **COMPLETED**
7. **entropy_from_CT** - Entropy (core function)

### Phase 3: Derived Functions (Medium Priority)
6. ✅ **adiabatic_lapse_rate_from_CT** - Advanced conversions - **COMPLETED**
7. **CT_from_enthalpy** / **CT_from_entropy** / **CT_from_rho** - Advanced conversions
8. **CT_first_derivatives** / **CT_second_derivatives** - CT derivatives
9. **entropy_first_derivatives** / **entropy_second_derivatives** - Entropy derivatives
10. **pt_first_derivatives** / **pt_second_derivatives** - PT derivatives

### Phase 4: Density Functions (Medium Priority)
10. ✅ **kappa** / **sound_speed** - Derived from specvol_alpha_beta - **COMPLETED**
11. ✅ **specvol_first_derivatives** / **specvol_second_derivatives** - Specvol derivatives - **COMPLETED**
12. ✅ **rho_first_derivatives** / **rho_second_derivatives** - Density derivatives - **COMPLETED**
13. ✅ **rho_first_derivatives_wrt_enthalpy** / **rho_second_derivatives_wrt_enthalpy** - Enthalpy derivatives - **COMPLETED**
14. ✅ **specvol_first_derivatives_wrt_enthalpy** / **specvol_second_derivatives_wrt_enthalpy** - Enthalpy derivatives - **COMPLETED**
15. ✅ **specvol_anom_standard** / **pot_rho_t_exact** - Other density functions - **COMPLETED** (pot_rho_t_exact)
16. **rho_t_exact** / **infunnel** - Other density functions
17. **spiciness0** / **spiciness1** / **spiciness2** - Spiciness functions

### Phase 5: Energy Functions (Medium Priority)
17. **enthalpy** / **enthalpy_diff** / **internal_energy** - Core energy functions
18. **latentheat_evap_CT** / **latentheat_evap_t** / **latentheat_melting** - Latent heat
19. **enthalpy_first_derivatives** / **enthalpy_second_derivatives** - Enthalpy derivatives
20. **enthalpy_t_exact** / **enthalpy_SSO_0** - Exact enthalpy functions

### Phase 6: Freezing Functions (Lower Priority)
21. **CT_freezing** / **t_freezing** / **SA_freezing_from_CT** / **pressure_freezing_CT** - Core freezing
22. **CT_freezing_first_derivatives** / **t_freezing_first_derivatives** / **SA_freezing_from_t** - Freezing derivatives
23. **CT_freezing_poly** / **t_freezing_poly** / **SA_freezing_from_CT_poly** / **SA_freezing_from_t_poly** - Polynomial versions
24. **CT_freezing_first_derivatives_poly** / **t_freezing_first_derivatives_poly** - Polynomial derivatives

### Phase 7: Ice Functions (Lower Priority)
25. **enthalpy_ice** / **entropy_ice** / **internal_energy_ice** / **cp_ice** / **adiabatic_lapse_rate_ice** - Core ice
26. **alpha_wrt_t_ice** / **chem_potential_water_ice** / **kappa_const_t_ice** / **kappa_ice** / **pressure_coefficient_ice** - Ice properties
27. **ice_fraction_to_freeze_seawater** / **melting_ice_into_seawater** / **melting_ice_SA_CT_ratio** / **melting_ice_equilibrium_SA_CT_ratio** - Ice melting
28. **melting_seaice_into_seawater** / **melting_seaice_SA_CT_ratio** / **seaice_fraction_to_freeze_seawater** - Sea ice melting
29. **pot_enthalpy_ice_freezing** / **Helmholtz_energy_ice** / **gibbs_ice** / **gibbs_ice_part_t** / **gibbs_ice_pt0** / **gibbs_ice_pt0_pt0** - Advanced ice

### Phase 8: Stability & Utility (Lower Priority)
30. ✅ **thermobaric** / **cabbeling** - Stability coefficients - **COMPLETED**
31. **O2sol** / **O2sol_SP_pt** - Oxygen solubility

### Phase 9: Salinity & Other Conversions (Lower Priority)
32. **SAAR** / **deltaSA_from_SP** - Salinity anomalies
33. **C_from_SP** / **SP_from_C** - Conductivity conversions
34. **SA_from_Sstar** / **Sstar_from_SA** / **Sstar_from_SP** / **SP_from_Sstar** - Preformed salinity
35. **SP_from_SK** / **SP_from_SR** / **SR_from_SP** - Other salinity scales
36. **CT_maxdensity** / **adiabatic_lapse_rate_from_CT** - Other conversions

### Phase 10: Pressure/Depth (Lower Priority)
37. **p_from_z** / **z_from_p** - Complete iterative/integration solutions

### Phase 11: Missing Functions (Future)
38. All remaining 66 missing functions (exact versions, Baltic functions, frazil ice, etc.)

## Implementation Strategy

### For Each Function:

1. **Find Reference Implementation**
   - Check GSW-C source code
   - Check GSW-Matlab code
   - Check polyTEOS repository
   - Check Roquet et al. (2015) paper

2. **Extract Algorithm**
   - Identify polynomial coefficients
   - Identify iterative procedures
   - Identify numerical methods

3. **Implement in Pure PyTorch**
   - Use torch operations only
   - Ensure GPU compatibility
   - Ensure autograd compatibility
   - Handle broadcasting correctly

4. **Test**
   - Numerical parity tests (rtol=1e-6, atol=1e-8)
   - Gradient checks (gradcheck)
   - Edge cases

5. **Remove Reference Wrapper**
   - Delete `_call_reference()` call
   - Remove NumPy dependencies
   - Update documentation

## Key Resources

- **Roquet et al. (2015)**: "Accurate polynomial expressions for the density and specific volume of seawater using the TEOS-10 standard" - Contains 75-term polynomial coefficients
- **polyTEOS Repository**: https://github.com/fabien-roquet/polyTEOS
- **GSW-C Source**: C implementation with coefficients
- **GSW-Matlab**: MATLAB reference implementation
- **TEOS-10 Manual**: https://www.teos-10.org/

## Notes

- All functions must work on GPU
- All functions must support autograd
- All functions must maintain numerical parity with reference GSW
- Priority is on functions that other functions depend on
