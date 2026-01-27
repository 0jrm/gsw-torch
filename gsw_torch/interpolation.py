"""
Functions for vertical interpolation.
"""

import torch

from ._utilities import indexer, match_args_return

__all__ = [
    "sa_ct_interp",
    "tracer_ct_interp",
]


def _util_intersect(x, nx, y, ny):
    """
    Find intersection of two sorted arrays.
    
    Returns indices into x and y arrays for matching values.
    Similar to gsw_util_intersect in GSW-C.
    
    Parameters
    ----------
    x : torch.Tensor
        First sorted array
    nx : int
        Length of x
    y : torch.Tensor
        Second sorted array
    ny : int
        Length of y
        
    Returns
    -------
    i_out : list
        Indices into x for matching values
    i_1 : list
        Indices into y for matching values
    """
    i_out = []
    i_1 = []
    i = 0
    j = 0
    
    while i < nx and j < ny:
        if abs(x[i].item() - y[j].item()) < 1e-10:
            i_out.append(i)
            i_1.append(j)
            i += 1
            j += 1
        elif x[i].item() < y[j].item():
            i += 1
        else:
            j += 1
    
    return i_out, i_1


def _linear_interp_sa_ct(sa_obs, ct_obs, indep_var_obs, n_obs, indep_var_i, n_i):
    """
    Linear interpolation of SA and CT.
    
    Parameters
    ----------
    sa_obs : torch.Tensor
        Observed SA values
    ct_obs : torch.Tensor
        Observed CT values
    indep_var_obs : torch.Tensor
        Independent variable at observations
    n_obs : int
        Number of observations
    indep_var_i : torch.Tensor
        Independent variable at interpolation points
    n_i : int
        Number of interpolation points
        
    Returns
    -------
    sa_i : torch.Tensor
        Linearly interpolated SA
    ct_i : torch.Tensor
        Linearly interpolated CT
    """
    sa_i = torch.zeros((n_i,), dtype=torch.float64, device=sa_obs.device)
    ct_i = torch.zeros((n_i,), dtype=torch.float64, device=ct_obs.device)
    
    for i in range(n_i):
        iv = indep_var_i[i].item()
        
        # Find interval
        if iv <= indep_var_obs[0].item():
            sa_i[i] = sa_obs[0]
            ct_i[i] = ct_obs[0]
        elif iv >= indep_var_obs[n_obs-1].item():
            sa_i[i] = sa_obs[n_obs-1]
            ct_i[i] = ct_obs[n_obs-1]
        else:
            # Binary search for interval
            j = 0
            k = n_obs - 1
            while k - j > 1:
                m = (j + k) // 2
                if iv > indep_var_obs[m].item():
                    j = m
                else:
                    k = m
            
            # Linear interpolation
            iv0 = indep_var_obs[j].item()
            iv1 = indep_var_obs[k].item()
            t = (iv - iv0) / (iv1 - iv0) if iv1 != iv0 else 0.0
            sa_i[i] = sa_obs[j] + t * (sa_obs[k] - sa_obs[j])
            ct_i[i] = ct_obs[j] + t * (ct_obs[k] - ct_obs[j])
    
    return sa_i, ct_i


def _sa_ct_interp_core(sa, ct, pgood, pi):
    """
    Core SA/CT interpolation function using full Reiniger-Ross method.
    
    This implements the complete Reiniger-Ross interpolation method as
    specified in GSW-C. The method uses rotated coordinate systems and
    multiple interpolation directions to achieve accurate results.
    
    Parameters
    ----------
    sa : torch.Tensor
        Absolute Salinity values (monotonically increasing pressure)
    ct : torch.Tensor
        Conservative Temperature values
    pgood : torch.Tensor
        Pressure values (monotonically increasing)
    pi : torch.Tensor
        Pressure values to interpolate at
        
    Returns
    -------
    sai : torch.Tensor
        Interpolated SA values at pi
    cti : torch.Tensor
        Interpolated CT values at pi
    """
    from .utility import _pchip_interp_core
    from .freezing import CT_freezing_poly
    
    # Constants
    factor = 9.0
    rec_factor = 1.0 / factor
    
    sin_kpi_on_16 = torch.tensor([
        1.950903220161283e-1,
        3.826834323650898e-1,
        5.555702330196022e-1,
        7.071067811865475e-1,
        8.314696123025452e-1,
        9.238795325112867e-1,
        9.807852804032304e-1
    ], dtype=torch.float64, device=sa.device)
    
    cos_kpi_on_16 = torch.tensor([
        9.807852804032304e-1,
        9.238795325112867e-1,
        8.314696123025452e-1,
        7.071067811865476e-1,
        5.555702330196023e-1,
        3.826834323650898e-1,
        1.950903220161283e-1
    ], dtype=torch.float64, device=sa.device)
    
    # Check minimum length
    if len(sa) < 4:
        raise ValueError("sa_ct_interp requires at least 4 bottles")
    
    # Round pressures to 0.001 dbar precision
    p_obs = torch.round(pgood * 1000.0) / 1000.0
    p_i_tmp = torch.round(pi * 1000.0) / 1000.0
    
    # Apply freezing point corrections to observations
    prof_len = len(sa)
    ct_corrected = ct.clone()
    for i in range(prof_len):
        ct_f = CT_freezing_poly(sa[i:i+1], p_obs[i:i+1], torch.zeros(1, dtype=torch.float64, device=sa.device))
        if ct_corrected[i] < (ct_f[0] - 0.1):
            ct_corrected[i] = ct_f[0]
    ct = ct_corrected
    
    # Combine observed and interpolation pressures, sort, remove duplicates
    p_all = torch.cat([p_obs, p_i_tmp])
    p_all_sorted, p_all_idx = torch.sort(p_all)
    
    # Remove duplicates
    unique_mask = torch.ones(len(p_all_sorted), dtype=torch.bool, device=sa.device)
    for i in range(1, len(p_all_sorted)):
        if abs(p_all_sorted[i].item() - p_all_sorted[i-1].item()) < 1e-10:
            unique_mask[i] = False
    p_all = p_all_sorted[unique_mask]
    p_all_len = len(p_all)
    
    # Find min/max observed pressures
    min_p_obs = p_obs[0].item()
    max_p_obs = p_obs[-1].item()
    i_min_p_obs = 0
    
    # Create indices for various subsets
    i_obs_plus_interp = []
    i_surf_and_obs_plus_interp = []
    p_obs_plus_interp = []
    p_surf_and_obs_plus_interp = []
    
    for i in range(p_all_len):
        p_val = p_all[i].item()
        if min_p_obs <= p_val <= max_p_obs:
            i_obs_plus_interp.append(i)
            p_obs_plus_interp.append(p_val)
        if p_val <= max_p_obs:
            i_surf_and_obs_plus_interp.append(i)
            p_surf_and_obs_plus_interp.append(p_val)
    
    i_obs_plus_interp_len = len(i_obs_plus_interp)
    i_surf_and_obs_plus_interp_len = len(i_surf_and_obs_plus_interp)
    
    if i_obs_plus_interp_len == 0:
        # No valid interpolation points
        sai = torch.full((len(pi),), float('nan'), dtype=torch.float64, device=sa.device)
        cti = torch.full((len(pi),), float('nan'), dtype=torch.float64, device=sa.device)
        return sai, cti
    
    # Find intersection of p_i_tmp with p_surf_and_obs_plus_interp
    p_surf_tensor = torch.tensor(p_surf_and_obs_plus_interp, dtype=torch.float64, device=sa.device)
    i_out, i_1 = _util_intersect(p_i_tmp, len(p_i_tmp), p_surf_tensor, len(p_surf_tensor))
    i_out_len = len(i_out)
    
    # Find intersection of p_obs with p_obs_plus_interp
    p_obs_plus_tensor = torch.tensor(p_obs_plus_interp, dtype=torch.float64, device=sa.device)
    i_2, i_3 = _util_intersect(p_obs, prof_len, p_obs_plus_tensor, len(p_obs_plus_interp))
    i_2_len = len(i_2)
    
    # Create independent variable (0, 1, 2, ..., prof_len-1)
    independent_variable = torch.arange(prof_len, dtype=torch.float64, device=sa.device)
    
    # Interpolate independent variable to p_obs_plus_interp using PCHIP
    independent_variable_obs_plus_interp = _pchip_interp_core(
        p_obs, independent_variable, p_obs_plus_tensor
    )
    
    # Scale SA
    scaled_sa_obs = factor * sa
    
    # Initial interpolation: CT and scaled SA using PCHIP on independent variable
    ct_i_obs_plus_interp = _pchip_interp_core(
        independent_variable, ct, independent_variable_obs_plus_interp
    )
    q_i = _pchip_interp_core(
        independent_variable, scaled_sa_obs, independent_variable_obs_plus_interp
    )
    
    # Initial SA interpolation
    sa_i_obs_plus_interp = rec_factor * q_i
    
    # Reiniger-Ross rotations: 7 rotations at different angles
    for k in range(7):
        # Rotate coordinates
        v_tmp = scaled_sa_obs * sin_kpi_on_16[k] + ct * cos_kpi_on_16[k]
        q_tmp = scaled_sa_obs * cos_kpi_on_16[k] - ct * sin_kpi_on_16[k]
        
        # Interpolate rotated coordinates
        v_i = _pchip_interp_core(independent_variable, v_tmp, independent_variable_obs_plus_interp)
        q_i = _pchip_interp_core(independent_variable, q_tmp, independent_variable_obs_plus_interp)
        
        # Rotate back and accumulate
        ct_i_obs_plus_interp = ct_i_obs_plus_interp + (-q_i * sin_kpi_on_16[k] + v_i * cos_kpi_on_16[k])
        sa_i_obs_plus_interp = sa_i_obs_plus_interp + rec_factor * (q_i * cos_kpi_on_16[k] + v_i * sin_kpi_on_16[k])
    
    # Average (multiply by 0.125 = 1/8, since we have 8 directions including initial)
    sa_i_obs_plus_interp = sa_i_obs_plus_interp * 0.125
    ct_i_obs_plus_interp = ct_i_obs_plus_interp * 0.125
    
    # Calculate limiting values using linear interpolation
    sa_i_limiting_obs_plus_interp, ct_i_limiting_obs_plus_interp = _linear_interp_sa_ct(
        sa, ct, independent_variable, prof_len,
        independent_variable_obs_plus_interp, i_obs_plus_interp_len
    )
    
    # Freezing point corrections - iterative process
    p_obs_plus_tensor_for_freeze = torch.tensor(p_obs_plus_interp, dtype=torch.float64, device=sa.device)
    ctf_i_tointerp = torch.zeros((i_obs_plus_interp_len,), dtype=torch.float64, device=sa.device)
    
    max_iterations = 100  # Safety limit
    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        
        # Calculate freezing temperatures
        for i in range(i_obs_plus_interp_len):
            ct_freeze = CT_freezing_poly(
                sa_i_obs_plus_interp[i:i+1],
                p_obs_plus_tensor_for_freeze[i:i+1],
                torch.zeros(1, dtype=torch.float64, device=sa.device)
            )
            if ct_i_limiting_obs_plus_interp[i] < (ct_freeze[0] - 0.1):
                ctf_i_tointerp[i] = ct_i_limiting_obs_plus_interp[i]
            else:
                ctf_i_tointerp[i] = ct_freeze[0]
        
        # Find first frozen point
        i_frozen = i_obs_plus_interp_len
        for i in range(i_obs_plus_interp_len):
            if ct_i_obs_plus_interp[i] < (ctf_i_tointerp[i] - 0.1):
                i_frozen = i
                break
        
        if i_frozen == i_obs_plus_interp_len:
            break  # No frozen points
        
        # Find shallower observation
        i_shallower = -1
        for j in range(i_2_len - 1, -1, -1):
            if i_3[j] <= i_frozen:
                i_shallower = j
                break
        
        if i_shallower < 0:
            break
        
        i_above = i_2[i_shallower]
        i_above_i = i_3[i_shallower]
        
        if i_above + 1 >= i_2_len:
            i_below_i = i_3[i_2_len - 1]
        else:
            i_below_i = i_3[i_above + 1]
        
        # Replace frozen points with limiting values
        for i in range(i_above_i, min(i_below_i + 1, i_obs_plus_interp_len)):
            sa_i_obs_plus_interp[i] = sa_i_limiting_obs_plus_interp[i]
            ct_i_obs_plus_interp[i] = ct_i_limiting_obs_plus_interp[i]
            ct_freeze = CT_freezing_poly(
                sa_i_obs_plus_interp[i:i+1],
                p_all[i_obs_plus_interp[i]:i_obs_plus_interp[i]+1],
                torch.zeros(1, dtype=torch.float64, device=sa.device)
            )
            ctf_i_tointerp[i] = ct_freeze[0]
    
    # Map to output
    sa_i_tooutput = torch.full((i_surf_and_obs_plus_interp_len,), float('nan'), dtype=torch.float64, device=sa.device)
    ct_i_tooutput = torch.full((i_surf_and_obs_plus_interp_len,), float('nan'), dtype=torch.float64, device=sa.device)
    
    if abs(min_p_obs) > 1e-10:
        # Handle surface extrapolation
        p_i_tmp_tensor = torch.tensor([p_surf_and_obs_plus_interp[i] for i in range(i_surf_and_obs_plus_interp_len)], 
                                      dtype=torch.float64, device=sa.device)
        for i in range(i_surf_and_obs_plus_interp_len):
            if p_i_tmp_tensor[i].item() < min_p_obs:
                # Use value at minimum pressure
                if i_min_p_obs < len(i_3):
                    sa_i_tooutput[i] = sa_i_obs_plus_interp[i_3[i_min_p_obs]]
                    ct_i_tooutput[i] = ct_i_obs_plus_interp[i_3[i_min_p_obs]]
        
        # Map observed+interpolated values
        for i in range(i_obs_plus_interp_len):
            # Find index in surf array
            for j in range(i_surf_and_obs_plus_interp_len):
                if abs(p_surf_and_obs_plus_interp[j] - p_obs_plus_interp[i]) < 1e-10:
                    sa_i_tooutput[j] = sa_i_obs_plus_interp[i]
                    ct_i_tooutput[j] = ct_i_obs_plus_interp[i]
                    break
    else:
        for i in range(i_obs_plus_interp_len):
            sa_i_tooutput[i] = sa_i_obs_plus_interp[i]
            ct_i_tooutput[i] = ct_i_obs_plus_interp[i]
    
    # Extract final output for requested pressures
    sai = torch.full((len(pi),), float('nan'), dtype=torch.float64, device=sa.device)
    cti = torch.full((len(pi),), float('nan'), dtype=torch.float64, device=sa.device)
    
    for i in range(i_out_len):
        if i_out[i] < len(sai) and i_1[i] < len(sa_i_tooutput):
            sai[i_out[i]] = sa_i_tooutput[i_1[i]]
            cti[i_out[i]] = ct_i_tooutput[i_1[i]]
    
    return sai, cti


def _tracer_ct_interp_core(tr, ct, pgood, pi, factor):
    """
    Core tracer/CT interpolation function using full Reiniger-Ross method.
    
    Similar to sa_ct_interp but without freezing point corrections.
    
    Parameters
    ----------
    tr : torch.Tensor
        Tracer values (monotonically increasing pressure)
    ct : torch.Tensor
        Conservative Temperature values
    pgood : torch.Tensor
        Pressure values (monotonically increasing)
    pi : torch.Tensor
        Pressure values to interpolate at
    factor : float
        Scaling factor (typically 9.0)
        
    Returns
    -------
    tri : torch.Tensor
        Interpolated tracer values at pi
    cti : torch.Tensor
        Interpolated CT values at pi
    """
    from .utility import _pchip_interp_core
    
    # Constants
    rec_factor = 1.0 / factor
    
    sin_kpi_on_16 = torch.tensor([
        1.950903220161283e-1,
        3.826834323650898e-1,
        5.555702330196022e-1,
        7.071067811865475e-1,
        8.314696123025452e-1,
        9.238795325112867e-1,
        9.807852804032304e-1
    ], dtype=torch.float64, device=tr.device)
    
    cos_kpi_on_16 = torch.tensor([
        9.807852804032304e-1,
        9.238795325112867e-1,
        8.314696123025452e-1,
        7.071067811865476e-1,
        5.555702330196023e-1,
        3.826834323650898e-1,
        1.950903220161283e-1
    ], dtype=torch.float64, device=tr.device)
    
    # Check minimum length
    if len(tr) < 4:
        raise ValueError("tracer_ct_interp requires at least 4 bottles")
    
    # Round pressures to 0.001 dbar precision
    p_obs = torch.round(pgood * 1000.0) / 1000.0
    p_i_tmp = torch.round(pi * 1000.0) / 1000.0
    
    # Combine observed and interpolation pressures, sort, remove duplicates
    p_all = torch.cat([p_obs, p_i_tmp])
    p_all_sorted, p_all_idx = torch.sort(p_all)
    
    # Remove duplicates
    unique_mask = torch.ones(len(p_all_sorted), dtype=torch.bool, device=tr.device)
    for i in range(1, len(p_all_sorted)):
        if abs(p_all_sorted[i].item() - p_all_sorted[i-1].item()) < 1e-10:
            unique_mask[i] = False
    p_all = p_all_sorted[unique_mask]
    p_all_len = len(p_all)
    
    # Find min/max observed pressures
    min_p_obs = p_obs[0].item()
    max_p_obs = p_obs[-1].item()
    i_min_p_obs = 0
    
    # Create indices for various subsets
    i_obs_plus_interp = []
    i_surf_and_obs_plus_interp = []
    p_obs_plus_interp = []
    p_surf_and_obs_plus_interp = []
    
    for i in range(p_all_len):
        p_val = p_all[i].item()
        if min_p_obs <= p_val <= max_p_obs:
            i_obs_plus_interp.append(i)
            p_obs_plus_interp.append(p_val)
        if p_val <= max_p_obs:
            i_surf_and_obs_plus_interp.append(i)
            p_surf_and_obs_plus_interp.append(p_val)
    
    i_obs_plus_interp_len = len(i_obs_plus_interp)
    i_surf_and_obs_plus_interp_len = len(i_surf_and_obs_plus_interp)
    
    if i_obs_plus_interp_len == 0:
        # No valid interpolation points
        tri = torch.full((len(pi),), float('nan'), dtype=torch.float64, device=tr.device)
        cti = torch.full((len(pi),), float('nan'), dtype=torch.float64, device=tr.device)
        return tri, cti
    
    # Find intersection of p_i_tmp with p_surf_and_obs_plus_interp
    p_surf_tensor = torch.tensor(p_surf_and_obs_plus_interp, dtype=torch.float64, device=tr.device)
    i_out, i_1 = _util_intersect(p_i_tmp, len(p_i_tmp), p_surf_tensor, len(p_surf_tensor))
    i_out_len = len(i_out)
    
    # Create independent variable (0, 1, 2, ..., prof_len-1)
    prof_len = len(tr)
    independent_variable = torch.arange(prof_len, dtype=torch.float64, device=tr.device)
    
    # Interpolate independent variable to p_obs_plus_interp using PCHIP
    p_obs_plus_tensor = torch.tensor(p_obs_plus_interp, dtype=torch.float64, device=tr.device)
    independent_variable_obs_plus_interp = _pchip_interp_core(
        p_obs, independent_variable, p_obs_plus_tensor
    )
    
    # Scale tracer
    scaled_tracer_obs = factor * tr
    
    # Initial interpolation: CT and scaled tracer using PCHIP on independent variable
    ct_i_obs_plus_interp = _pchip_interp_core(
        independent_variable, ct, independent_variable_obs_plus_interp
    )
    q_i = _pchip_interp_core(
        independent_variable, scaled_tracer_obs, independent_variable_obs_plus_interp
    )
    
    # Initial tracer interpolation
    tracer_i_obs_plus_interp = rec_factor * q_i
    
    # Reiniger-Ross rotations: 7 rotations at different angles
    for k in range(7):
        # Rotate coordinates
        v_tmp = scaled_tracer_obs * sin_kpi_on_16[k] + ct * cos_kpi_on_16[k]
        q_tmp = scaled_tracer_obs * cos_kpi_on_16[k] - ct * sin_kpi_on_16[k]
        
        # Interpolate rotated coordinates
        v_i = _pchip_interp_core(independent_variable, v_tmp, independent_variable_obs_plus_interp)
        q_i = _pchip_interp_core(independent_variable, q_tmp, independent_variable_obs_plus_interp)
        
        # Rotate back and accumulate
        ct_i_obs_plus_interp = ct_i_obs_plus_interp + (-q_i * sin_kpi_on_16[k] + v_i * cos_kpi_on_16[k])
        tracer_i_obs_plus_interp = tracer_i_obs_plus_interp + rec_factor * (q_i * cos_kpi_on_16[k] + v_i * sin_kpi_on_16[k])
    
    # Average (multiply by 0.125 = 1/8, since we have 8 directions including initial)
    tracer_i_obs_plus_interp = tracer_i_obs_plus_interp * 0.125
    ct_i_obs_plus_interp = ct_i_obs_plus_interp * 0.125
    
    # Map to output
    tracer_i_tooutput = torch.full((i_surf_and_obs_plus_interp_len,), float('nan'), dtype=torch.float64, device=tr.device)
    ct_i_tooutput = torch.full((i_surf_and_obs_plus_interp_len,), float('nan'), dtype=torch.float64, device=tr.device)
    
    if abs(min_p_obs) > 1e-10:
        # Handle surface extrapolation
        p_i_tmp_tensor = torch.tensor([p_surf_and_obs_plus_interp[i] for i in range(i_surf_and_obs_plus_interp_len)], 
                                      dtype=torch.float64, device=tr.device)
        for i in range(i_surf_and_obs_plus_interp_len):
            if p_i_tmp_tensor[i].item() < min_p_obs:
                # Use value at minimum pressure
                tracer_i_tooutput[i] = tracer_i_obs_plus_interp[0]
                ct_i_tooutput[i] = ct_i_obs_plus_interp[0]
        
        # Map observed+interpolated values
        for i in range(i_obs_plus_interp_len):
            # Find index in surf array
            for j in range(i_surf_and_obs_plus_interp_len):
                if abs(p_surf_and_obs_plus_interp[j] - p_obs_plus_interp[i]) < 1e-10:
                    tracer_i_tooutput[j] = tracer_i_obs_plus_interp[i]
                    ct_i_tooutput[j] = ct_i_obs_plus_interp[i]
                    break
    else:
        for i in range(i_obs_plus_interp_len):
            tracer_i_tooutput[i] = tracer_i_obs_plus_interp[i]
            ct_i_tooutput[i] = ct_i_obs_plus_interp[i]
    
    # Extract final output for requested pressures
    tri = torch.full((len(pi),), float('nan'), dtype=torch.float64, device=tr.device)
    cti = torch.full((len(pi),), float('nan'), dtype=torch.float64, device=tr.device)
    
    for i in range(i_out_len):
        if i_out[i] < len(tri) and i_1[i] < len(tracer_i_tooutput):
            tri[i_out[i]] = tracer_i_tooutput[i_1[i]]
            cti[i_out[i]] = ct_i_tooutput[i_1[i]]
    
    return tri, cti


@match_args_return
def sa_ct_interp(SA, CT, p, p_i, axis=0):
    """
    Interpolates vertical casts of values of Absolute Salinity
    and Conservative Temperature to the arbitrary pressures p_i.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    p_i : array-like
        Sea pressure to interpolate on, dbar
    axis : int, optional, default is 0
        The index of the pressure dimension in SA and CT.

    Returns
    -------
    SA_i : torch.Tensor
        Values of SA interpolated to p_i along the specified axis.
    CT_i : torch.Tensor
        Values of CT interpolated to p_i along the specified axis.
    """
    SA = torch.as_tensor(SA, dtype=torch.float64)
    CT = torch.as_tensor(CT, dtype=torch.float64)
    p = torch.as_tensor(p, dtype=torch.float64)
    p_i = torch.as_tensor(p_i, dtype=torch.float64)

    if SA.shape != CT.shape:
        raise ValueError(
            f"Shapes of SA and CT must match; found {SA.shape} and {CT.shape}"
        )
    if p.ndim != p_i.ndim:
        raise ValueError(
            f"p and p_i must have the same number of dimensions;\n"
            f" found {p.ndim} versus {p_i.ndim}"
        )
    if p.ndim == 1 and SA.ndim > 1:
        if len(p) != SA.shape[axis]:
            raise ValueError(
                f"With 1-D p, len(p) must be SA.shape[axis];\n"
                f" found {len(p)} versus {SA.shape[axis]} on specified axis, {axis}"
            )
        ind = [None] * SA.ndim
        ind[axis] = slice(None)
        p = p[tuple(ind)]
        p_i = p_i[tuple(ind)]
    elif p.ndim > 1:
        if p.shape != SA.shape:
            raise ValueError(
                f"With {p.ndim}-D p, shapes of p and SA must match;\n"
                f"found {p.shape} and {SA.shape}"
            )
        # Check that p and p_i have same dimensions except on the interpolation axis
        for i in range(p.ndim):
            if i != axis:
                if p.shape[i] != p_i.shape[i]:
                    raise ValueError(
                        f"With {p.ndim}-D p, p and p_i must have the same dimensions outside of axis {axis};\n"
                        f" found {p.shape} versus {p_i.shape}"
                    )

    # Check for increasing pressure
    if torch.any(torch.diff(p_i, dim=axis) <= 0) or torch.any(torch.diff(p, dim=axis) <= 0):
        raise ValueError("p and p_i must be increasing along the specified axis")

    p = torch.broadcast_to(p, SA.shape)
    goodmask = ~(torch.isnan(SA) | torch.isnan(CT) | torch.isnan(p))
    SA_i = torch.full(p_i.shape, float("nan"), dtype=torch.float64, device=SA.device)
    CT_i = torch.full(p_i.shape, float("nan"), dtype=torch.float64, device=SA.device)

    order = "C"  # Default order
    for ind in indexer(SA.shape, axis, order=order):
        igood = goodmask[ind]
        pgood = p[ind][igood]
        pi = p_i[ind]
        # There must be at least 4 non-NaN values for interpolation
        if len(pgood) >= 4:
            sa = SA[ind][igood]
            ct = CT[ind][igood]
            try:
                sai, cti = _sa_ct_interp_core(sa, ct, pgood, pi)
                SA_i[ind] = sai
                CT_i[ind] = cti
            except Exception:
                # If interpolation fails, leave as NaN
                pass

    return (SA_i, CT_i)


@match_args_return
def tracer_ct_interp(tracer, CT, p, p_i, factor=9.0, axis=0):
    """
    Interpolates vertical casts of values of a tracer
    and Conservative Temperature to the arbitrary pressures p_i.

    Parameters
    ----------
    tracer : array-like
        tracer
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    p_i : array-like
        Sea pressure to interpolate on, dbar
    factor: float, optional, default is 9.
        Ratio between the ranges of Conservative Temperature
        and tracer in the world ocean.
    axis : int, optional, default is 0
        The index of the pressure dimension in tracer and CT.

    Returns
    -------
    tracer_i : torch.Tensor
        Values of tracer interpolated to p_i along the specified axis.
    CT_i : torch.Tensor
        Values of CT interpolated to p_i along the specified axis.
    """
    tracer = torch.as_tensor(tracer, dtype=torch.float64)
    CT = torch.as_tensor(CT, dtype=torch.float64)
    p = torch.as_tensor(p, dtype=torch.float64)
    p_i = torch.as_tensor(p_i, dtype=torch.float64)

    if tracer.shape != CT.shape:
        raise ValueError(
            f"Shapes of tracer and CT must match; found {tracer.shape} and {CT.shape}"
        )
    if p.ndim != p_i.ndim:
        raise ValueError(
            f"p and p_i must have the same number of dimensions;\n"
            f" found {p.ndim} versus {p_i.ndim}"
        )
    if p.ndim == 1 and tracer.ndim > 1:
        if len(p) != tracer.shape[axis]:
            raise ValueError(
                f"With 1-D p, len(p) must be tracer.shape[axis];\n"
                f" found {len(p)} versus {tracer.shape[axis]} on specified axis, {axis}"
            )
        ind = [None] * tracer.ndim
        ind[axis] = slice(None)
        p = p[tuple(ind)]
        p_i = p_i[tuple(ind)]
    elif p.ndim > 1:
        if p.shape != tracer.shape:
            raise ValueError(
                f"With {p.ndim}-D p, shapes of p and tracer must match;\n"
                f"found {p.shape} and {tracer.shape}"
            )
        # Check that p and p_i have same dimensions except on the interpolation axis
        for i in range(p.ndim):
            if i != axis:
                if p.shape[i] != p_i.shape[i]:
                    raise ValueError(
                        f"With {p.ndim}-D p, p and p_i must have the same dimensions outside of axis {axis};\n"
                        f" found {p.shape} versus {p_i.shape}"
                    )

    # Check for increasing pressure
    if torch.any(torch.diff(p_i, dim=axis) <= 0) or torch.any(torch.diff(p, dim=axis) <= 0):
        raise ValueError("p and p_i must be increasing along the specified axis")

    p = torch.broadcast_to(p, tracer.shape)
    goodmask = ~(torch.isnan(tracer) | torch.isnan(CT) | torch.isnan(p))
    tracer_i = torch.full(p_i.shape, float("nan"), dtype=torch.float64, device=tracer.device)
    CT_i = torch.full(p_i.shape, float("nan"), dtype=torch.float64, device=tracer.device)

    order = "C"  # Default order
    for ind in indexer(tracer.shape, axis, order=order):
        igood = goodmask[ind]
        pgood = p[ind][igood]
        pi = p_i[ind]
        # There must be at least 4 non-NaN values for interpolation
        if len(pgood) >= 4:
            tr = tracer[ind][igood]
            ct = CT[ind][igood]
            try:
                tri, cti = _tracer_ct_interp_core(tr, ct, pgood, pi, factor)
                tracer_i[ind] = tri
                CT_i[ind] = cti
            except Exception:
                # If interpolation fails, leave as NaN
                pass

    return (tracer_i, CT_i)
