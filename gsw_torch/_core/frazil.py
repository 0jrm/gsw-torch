"""
Frazil ice functions - exact reimplementations from GSW-C.

Frazil ice forms when seawater is supercooled and ice crystals form.
These functions calculate the properties of seawater-ice mixtures.
"""

import torch
from .._utilities import as_tensor
from ._saar_data import GSW_INVALID_VALUE


def frazil_properties(SA_bulk, h_bulk, p):
    """
    Calculates the mass fraction of ice (mass of ice divided by mass of ice
    plus seawater), w_Ih_final, which results from given values of the bulk
    Absolute Salinity, SA_bulk, bulk enthalpy, h_bulk, occurring at pressure
    p.  The final values of Absolute Salinity, SA_final, and Conservative
    Temperature, CT_final, of the interstitial seawater phase are also
    returned.  This code assumes that there is no dissolved air in the
    seawater (that is, saturation_fraction is assumed to be zero
    throughout the code).

    Parameters
    ----------
    SA_bulk : array-like
        bulk Absolute Salinity of the seawater and ice mixture, g/kg
    h_bulk : array-like
        bulk enthalpy of the seawater and ice mixture, J/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    SA_final : torch.Tensor, g/kg
        Absolute Salinity of the seawater in the final state,
        whether or not any ice is present.
    CT_final : torch.Tensor, deg C
        Conservative Temperature of the seawater in the final
        state, whether or not any ice is present.
    w_Ih_final : torch.Tensor, unitless
        mass fraction of ice in the final seawater-ice mixture.
        If this ice mass fraction is positive, the system is at
        thermodynamic equilibrium.  If this ice mass fraction is
        zero there is no ice in the final state which consists
        only of seawater which is warmer than the freezing
        temperature.

    Notes
    -----
    This is an exact reimplementation from GSW-C using modified Newton-Raphson
    iteration (McDougall and Wotherspoon, 2014).
    """
    from .freezing import CT_freezing, t_freezing, CT_freezing_first_derivatives, t_freezing_first_derivatives
    from .energy import enthalpy_CT_exact, enthalpy_first_derivatives_CT_exact
    from .conversions import CT_from_enthalpy_exact
    from .ice import enthalpy_ice, cp_ice
    
    SA_bulk = as_tensor(SA_bulk, dtype=torch.float64)
    h_bulk = as_tensor(h_bulk, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA_bulk, h_bulk, p = torch.broadcast_tensors(SA_bulk, h_bulk, p)
    
    # Constants
    saturation_fraction = torch.tensor(0.0, dtype=torch.float64, device=SA_bulk.device)
    num_f = 5.0e-2
    num_f2 = 6.9e-7
    num_p = 2.21
    
    # Initialize outputs
    SA_final = torch.zeros_like(SA_bulk)
    CT_final = torch.zeros_like(SA_bulk)
    w_Ih_final = torch.zeros_like(SA_bulk)
    
    # Calculate func0 = h_bulk - enthalpy_CT_exact(SA_bulk, CT_freezing(SA_bulk, p, 0), p)
    ctf_bulk = CT_freezing(SA_bulk, p, saturation_fraction)
    hf_bulk = enthalpy_CT_exact(SA_bulk, ctf_bulk, p)
    func0 = h_bulk - hf_bulk
    
    # Case 1: func0 >= 0 (no ice, warm seawater)
    no_ice_mask = func0 >= 0.0
    SA_final[no_ice_mask] = SA_bulk[no_ice_mask]
    CT_final[no_ice_mask] = CT_from_enthalpy_exact(SA_bulk[no_ice_mask], h_bulk[no_ice_mask], p[no_ice_mask])
    w_Ih_final[no_ice_mask] = 0.0
    
    # Case 2: func0 < 0 (ice present, need to solve)
    ice_mask = ~no_ice_mask
    
    if torch.any(ice_mask):
        SA_bulk_ice = SA_bulk[ice_mask]
        h_bulk_ice = h_bulk[ice_mask]
        p_ice = p[ice_mask]
        func0_ice = func0[ice_mask]
        
        # Initial estimate using polynomial
        dfunc_dw_ih_mean_poly = 3.347814e+05 - num_f * func0_ice * (1.0 + num_f2 * func0_ice) - num_p * p_ice
        w_ih = torch.clamp(-func0_ice / dfunc_dw_ih_mean_poly, max=0.95)
        sa = SA_bulk_ice / (1.0 - w_ih)
        
        # Check bounds
        invalid_mask = (sa < 0.0) | (sa > 120.0)
        if torch.any(invalid_mask):
            SA_final[ice_mask] = torch.where(invalid_mask, torch.tensor(GSW_INVALID_VALUE, dtype=torch.float64, device=SA_bulk.device), SA_final[ice_mask])
            CT_final[ice_mask] = torch.where(invalid_mask, torch.tensor(GSW_INVALID_VALUE, dtype=torch.float64, device=SA_bulk.device), CT_final[ice_mask])
            w_Ih_final[ice_mask] = torch.where(invalid_mask, torch.tensor(GSW_INVALID_VALUE, dtype=torch.float64, device=SA_bulk.device), w_Ih_final[ice_mask])
            # Continue with valid points
            valid_mask = ~invalid_mask
            if not torch.any(valid_mask):
                return SA_final, CT_final, w_Ih_final
            SA_bulk_ice = SA_bulk_ice[valid_mask]
            h_bulk_ice = h_bulk_ice[valid_mask]
            p_ice = p_ice[valid_mask]
            func0_ice = func0_ice[valid_mask]
            w_ih = w_ih[valid_mask]
            sa = sa[valid_mask]
            ice_mask_valid = torch.zeros(len(SA_bulk), dtype=torch.bool, device=SA_bulk.device)
            ice_mask_valid[ice_mask] = valid_mask
        else:
            ice_mask_valid = ice_mask
        
        # Calculate initial derivative
        ctf = CT_freezing(sa, p_ice, saturation_fraction)
        hf = enthalpy_CT_exact(sa, ctf, p_ice)
        tf = t_freezing(sa, p_ice, saturation_fraction)
        h_ihf = enthalpy_ice(tf, p_ice)
        cp_ih = cp_ice(tf, p_ice)
        h_hat_sa, h_hat_ct = enthalpy_first_derivatives_CT_exact(sa, ctf, p_ice)
        ctf_sa, _ = CT_freezing_first_derivatives(sa, p_ice, saturation_fraction)
        tf_sa, _ = t_freezing_first_derivatives(sa, p_ice, saturation_fraction)
        
        dfunc_dw_ih = hf - h_ihf - sa * (h_hat_sa + h_hat_ct * ctf_sa + w_ih * cp_ih * tf_sa / (1.0 - w_ih))
        
        # Modified Newton-Raphson iteration (3 iterations as in GSW-C)
        for iteration in range(3):
            # Calculate function value
            ctf = CT_freezing(sa, p_ice, saturation_fraction)
            hf = enthalpy_CT_exact(sa, ctf, p_ice)
            tf = t_freezing(sa, p_ice, saturation_fraction)
            h_ihf = enthalpy_ice(tf, p_ice)
            
            func = h_bulk_ice - (1.0 - w_ih) * hf - w_ih * h_ihf
            
            w_ih_old = w_ih
            w_ih = w_ih_old - func / dfunc_dw_ih
            w_ih_mean = 0.5 * (w_ih + w_ih_old)
            
            # Check bounds
            if torch.any(w_ih_mean > 0.9):
                invalid_iter_mask = w_ih_mean > 0.9
                # Mark as invalid
                sa[invalid_iter_mask] = GSW_INVALID_VALUE
                w_ih[invalid_iter_mask] = GSW_INVALID_VALUE
            
            sa_mean = SA_bulk_ice / (1.0 - w_ih_mean)
            ctf_mean = CT_freezing(sa_mean, p_ice, saturation_fraction)
            hf_mean = enthalpy_CT_exact(sa_mean, ctf_mean, p_ice)
            tf_mean = t_freezing(sa_mean, p_ice, saturation_fraction)
            h_ihf_mean = enthalpy_ice(tf_mean, p_ice)
            cp_ih_mean = cp_ice(tf_mean, p_ice)
            h_hat_sa_mean, h_hat_ct_mean = enthalpy_first_derivatives_CT_exact(sa_mean, ctf_mean, p_ice)
            ctf_sa_mean, _ = CT_freezing_first_derivatives(sa_mean, p_ice, saturation_fraction)
            tf_sa_mean, _ = t_freezing_first_derivatives(sa_mean, p_ice, saturation_fraction)
            
            dfunc_dw_ih = hf_mean - h_ihf_mean - sa_mean * (h_hat_sa_mean + h_hat_ct_mean * ctf_sa_mean + 
                                                                 w_ih_mean * cp_ih_mean * tf_sa_mean / (1.0 - w_ih_mean))
            
            w_ih = w_ih_old - func / dfunc_dw_ih
            
            # Check bounds again
            if torch.any(w_ih > 0.9):
                invalid_iter_mask = w_ih > 0.9
                # Mark as invalid
                sa[invalid_iter_mask] = GSW_INVALID_VALUE
                w_ih[invalid_iter_mask] = GSW_INVALID_VALUE
            
            sa = SA_bulk_ice / (1.0 - w_ih)
        
        # Final values
        sa_final_ice = sa
        ctf_final_ice = CT_freezing(sa_final_ice, p_ice, saturation_fraction)
        w_ih_final_ice = w_ih
        
        # Handle negative w_Ih (within machine precision)
        negative_mask = w_ih_final_ice < 0.0
        if torch.any(negative_mask):
            sa_final_ice[negative_mask] = SA_bulk_ice[negative_mask]
            ctf_final_ice[negative_mask] = CT_from_enthalpy_exact(SA_bulk_ice[negative_mask], h_bulk_ice[negative_mask], p_ice[negative_mask])
            w_ih_final_ice[negative_mask] = 0.0
        
        # Store results
        SA_final[ice_mask_valid] = sa_final_ice
        CT_final[ice_mask_valid] = ctf_final_ice
        w_Ih_final[ice_mask_valid] = w_ih_final_ice
    
    return SA_final, CT_final, w_Ih_final


def frazil_ratios_adiabatic(SA, p, w_Ih):
    """
    Calculates the ratios of SA, CT and P changes when frazil ice forms or
    melts in response to an adiabatic change in pressure of a mixture of
    seawater and frazil ice crystals.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    w_Ih : array-like
        mass fraction of ice: the mass of ice divided by the
        sum of the masses of ice and seawater. 0 <= wIh <= 1. unitless.

    Returns
    -------
    dSA_dCT_frazil : torch.Tensor, g/(kg K)
        the ratio of the changes in Absolute Salinity
        to that of Conservative Temperature
    dSA_dP_frazil : torch.Tensor, g/(kg Pa)
        the ratio of the changes in Absolute Salinity
        to that of pressure (in Pa)
    dCT_dP_frazil : torch.Tensor, K/Pa
        the ratio of the changes in Conservative Temperature
        to that of pressure (in Pa)

    Notes
    -----
    This is an exact reimplementation from GSW-C.
    Note that dSA_dP_frazil and dCT_dP_frazil use pressure in Pa, not dbar.
    """
    from .freezing import CT_freezing, t_freezing, CT_freezing_first_derivatives, t_freezing_first_derivatives
    from .energy import enthalpy_CT_exact, enthalpy_first_derivatives_CT_exact
    from .ice import enthalpy_ice, cp_ice, adiabatic_lapse_rate_ice
    
    SA = as_tensor(SA, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    w_Ih = as_tensor(w_Ih, dtype=torch.float64)
    
    # Broadcast inputs
    SA, p, w_Ih = torch.broadcast_tensors(SA, p, w_Ih)
    
    # Constants
    saturation_fraction = torch.tensor(0.0, dtype=torch.float64, device=SA.device)
    db2pa = 1e4  # dbar to Pa conversion
    
    # Calculate freezing temperatures and enthalpies
    ctf = CT_freezing(SA, p, saturation_fraction)
    tf = t_freezing(SA, p, saturation_fraction)
    h = enthalpy_CT_exact(SA, ctf, p)
    h_ih = enthalpy_ice(tf, p)
    cp_ih = cp_ice(tf, p)
    gamma_ih_pa = adiabatic_lapse_rate_ice(tf, p)  # Returns K/Pa
    
    # Calculate derivatives
    # Note: freezing_first_derivatives return K/Pa, but GSW-C uses K/dbar
    # So we need to convert back to K/dbar for the calculation
    h_hat_sa, h_hat_ct = enthalpy_first_derivatives_CT_exact(SA, ctf, p)
    tf_sa, tf_p_pa = t_freezing_first_derivatives(SA, p, saturation_fraction)
    ctf_sa, ctf_p_pa = CT_freezing_first_derivatives(SA, p, saturation_fraction)
    
    # Convert pressure derivatives from Pa to dbar (GSW-C uses K/dbar internally)
    # Also convert gamma_ih from K/Pa to K/dbar
    tf_p = tf_p_pa * db2pa
    ctf_p = ctf_p_pa * db2pa
    gamma_ih = gamma_ih_pa * db2pa  # Convert to K/dbar
    
    # Calculate intermediate quantities (exact match to GSW-C)
    wcp = cp_ih * w_Ih / (1.0 - w_Ih)
    part = (tf_p - gamma_ih) / ctf_p
    
    bracket1 = h_hat_ct + wcp * part
    bracket2 = h - h_ih - SA * (h_hat_sa + wcp * (tf_sa - part * ctf_sa))
    rec_bracket3 = 1.0 / (h - h_ih - SA * (h_hat_sa + h_hat_ct * ctf_sa + wcp * tf_sa))
    
    # Calculate outputs (exact match to GSW-C)
    # Note: ctf_p is in K/dbar, but outputs need to be in Pa, so convert to K/Pa
    ctf_p_pa = ctf_p / db2pa
    dSA_dCT_frazil = SA * (bracket1 / bracket2)
    dSA_dP_frazil = SA * ctf_p_pa * bracket1 * rec_bracket3
    dCT_dP_frazil = ctf_p_pa * bracket2 * rec_bracket3
    
    return dSA_dCT_frazil, dSA_dP_frazil, dCT_dP_frazil


def frazil_ratios_adiabatic_poly(SA, p, w_Ih):
    """
    Calculates the ratios of SA, CT and P changes when frazil ice forms or
    melts in response to an adiabatic change in pressure of a mixture of
    seawater and frazil ice crystals, using polynomial approximations.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    w_Ih : array-like
        mass fraction of ice: the mass of ice divided by the
        sum of the masses of ice and seawater. 0 <= wIh <= 1. unitless.

    Returns
    -------
    dSA_dCT_frazil : torch.Tensor, g/(kg K)
        the ratio of the changes in Absolute Salinity
        to that of Conservative Temperature
    dSA_dP_frazil : torch.Tensor, g/(kg Pa)
        the ratio of the changes in Absolute Salinity
        to that of pressure (in Pa)
    dCT_dP_frazil : torch.Tensor, K/Pa
        the ratio of the changes in Conservative Temperature
        to that of pressure (in Pa)

    Notes
    -----
    This is an exact reimplementation from GSW-C using polynomial approximations.
    Uses CT_freezing_poly and t_freezing_poly instead of exact versions.
    """
    from .freezing import CT_freezing_poly, t_freezing_poly, CT_freezing_first_derivatives_poly, t_freezing_first_derivatives_poly
    from .energy import enthalpy, enthalpy_first_derivatives
    from .ice import enthalpy_ice, cp_ice, adiabatic_lapse_rate_ice
    
    SA = as_tensor(SA, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    w_Ih = as_tensor(w_Ih, dtype=torch.float64)
    
    # Broadcast inputs
    SA, p, w_Ih = torch.broadcast_tensors(SA, p, w_Ih)
    
    # Constants
    saturation_fraction = torch.tensor(0.0, dtype=torch.float64, device=SA.device)
    db2pa = 1e4  # dbar to Pa conversion
    
    # Calculate freezing temperatures and enthalpies (using polynomial versions)
    tf = t_freezing_poly(SA, p, saturation_fraction)
    ctf = CT_freezing_poly(SA, p, saturation_fraction)
    h = enthalpy(SA, ctf, p)  # Polynomial version
    h_ih = enthalpy_ice(tf, p)
    cp_ih = cp_ice(tf, p)
    gamma_ih_pa = adiabatic_lapse_rate_ice(tf, p)  # Returns K/Pa
    
    # Calculate derivatives (using polynomial versions)
    # Note: freezing_first_derivatives_poly return K/Pa, but GSW-C uses K/dbar
    # So we need to convert back to K/dbar for the calculation
    h_hat_sa, h_hat_ct = enthalpy_first_derivatives(SA, ctf, p)  # Polynomial version
    tf_sa, tf_p_pa = t_freezing_first_derivatives_poly(SA, p, saturation_fraction)
    ctf_sa, ctf_p_pa = CT_freezing_first_derivatives_poly(SA, p, saturation_fraction)
    
    # Convert pressure derivatives from Pa to dbar (GSW-C uses K/dbar internally)
    # Also convert gamma_ih from K/Pa to K/dbar
    tf_p = tf_p_pa * db2pa
    ctf_p = ctf_p_pa * db2pa
    gamma_ih = gamma_ih_pa * db2pa  # Convert to K/dbar
    
    # Calculate intermediate quantities (exact match to GSW-C)
    wcp = cp_ih * w_Ih / (1.0 - w_Ih)
    part = (tf_p - gamma_ih) / ctf_p
    
    bracket1 = h_hat_ct + wcp * part
    bracket2 = h - h_ih - SA * (h_hat_sa + wcp * (tf_sa - part * ctf_sa))
    rec_bracket3 = 1.0 / (h - h_ih - SA * (h_hat_sa + h_hat_ct * ctf_sa + wcp * tf_sa))
    
    # Calculate outputs (exact match to GSW-C)
    # Note: ctf_p is in K/dbar, but outputs need to be in Pa, so convert to K/Pa
    ctf_p_pa = ctf_p / db2pa
    dSA_dCT_frazil = SA * (bracket1 / bracket2)
    dSA_dP_frazil = SA * ctf_p_pa * bracket1 * rec_bracket3
    dCT_dP_frazil = ctf_p_pa * bracket2 * rec_bracket3
    
    return dSA_dCT_frazil, dSA_dP_frazil, dCT_dP_frazil
