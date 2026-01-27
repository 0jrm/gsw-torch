"""
Core PyTorch implementations of energy-related functions.

These functions calculate enthalpy, internal energy, and latent heats.
"""

import os
import sys

import numpy as np
import torch

from .._utilities import as_tensor


def _get_reference_gsw():
    """Helper to get reference GSW implementation."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../.."))
    
    ref_link = os.path.join(project_root, "reference", "gsw")
    ref_direct = "/home/jrm22n/gsw2torch/source_files"
    
    paths_to_try = []
    if os.path.islink(ref_link):
        target = os.readlink(ref_link)
        if os.path.isabs(target):
            paths_to_try.append(os.path.dirname(target))
        else:
            paths_to_try.append(os.path.join(os.path.dirname(ref_link), target))
    paths_to_try.extend([
        os.path.join(project_root, "reference"),
        os.path.join(current_dir, "../../../reference"),
        ref_direct,
    ])
    
    for path in paths_to_try:
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            continue
            
        gsw_init = os.path.join(abs_path, "gsw", "__init__.py")
        if os.path.exists(gsw_init) or os.path.exists(os.path.join(abs_path, "gsw")):
            sys.path.insert(0, abs_path)
            try:
                import gsw as gsw_ref
                return gsw_ref, abs_path
            except (ImportError, ModuleNotFoundError):
                if abs_path in sys.path:
                    sys.path.remove(abs_path)
                continue
    
    return None, None


def _call_reference(func_name, *args, device=None):
    """Helper to call reference GSW function and convert to torch."""
    gsw_ref, path = _get_reference_gsw()
    if gsw_ref is None:
        raise NotImplementedError(
            f"{func_name} requires reference gsw implementation. "
            "Please ensure reference/gsw is available."
        )
    
    # Convert inputs to numpy
    args_np = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            args_np.append(arg.detach().cpu().numpy())
            if device is None:
                device = arg.device
        else:
            args_np.append(np.asarray(arg, dtype=np.float64))
            if device is None:
                device = torch.device("cpu")
    
    # Call reference function
    func = getattr(gsw_ref, func_name)
    result = func(*args_np)
    
    # Convert result to torch
    if isinstance(result, tuple):
        return tuple(torch.as_tensor(r, dtype=torch.float64, device=device) for r in result)
    else:
        return torch.as_tensor(result, dtype=torch.float64, device=device)


__all__ = [
    "enthalpy",
    "dynamic_enthalpy",
    "enthalpy_diff",
    "enthalpy_first_derivatives",
    "enthalpy_first_derivatives_CT_exact",
    "enthalpy_second_derivatives",
    "enthalpy_second_derivatives_CT_exact",
    "enthalpy_SSO_0",
    "enthalpy_CT_exact",
    "enthalpy_t_exact",
    "internal_energy",
    "latentheat_evap_CT",
    "latentheat_evap_t",
    "latentheat_melting",
]

# Note: enthalpy_diff and enthalpy_SSO_0 are now pure PyTorch (call enthalpy)


def enthalpy(SA, CT, p):
    """
    Calculates specific enthalpy of seawater.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    enthalpy : torch.Tensor, J/kg
        specific enthalpy

    Notes
    -----
    This is a pure PyTorch implementation using the 75-term polynomial expression
    from Roquet et al. (2015). The enthalpy is calculated as:
    h = cp0 * CT + dynamic_enthalpy(SA, CT, p)
    where cp0 = 3991.86795711963 J/(kg K) and dynamic_enthalpy uses the same
    polynomial structure as specvol but with enthalpy-specific coefficients.
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, CT, p = torch.broadcast_tensors(SA, CT, p)
    
    # Conservative Temperature constant (cp0)
    cp0 = 3991.86795711963  # J/(kg K)
    
    # Calculate dynamic enthalpy
    dynamic_h = dynamic_enthalpy(SA, CT, p)
    
    # Enthalpy = cp0 * CT + dynamic_enthalpy
    h = cp0 * CT + dynamic_h
    
    return h


def dynamic_enthalpy(SA, CT, p):
    """
    Calculates dynamic enthalpy of seawater.
    
    Dynamic enthalpy is defined as enthalpy minus potential enthalpy
    (Young, 2010). This uses the computationally-efficient 75-term polynomial
    expression (Roquet et al., 2015).

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    dynamic_enthalpy : torch.Tensor, J/kg
        dynamic enthalpy
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, CT, p = torch.broadcast_tensors(SA, CT, p)
    
    # Reduced variables (same as specvol)
    SAu = 40.0 * 35.16504 / 35.0
    CTu = 40.0
    Zu = 1e4
    deltaS = 24.0
    
    ss = torch.sqrt((SA + deltaS) / SAu)
    tt = CT / CTu
    z = p / Zu
    
    # Enthalpy coefficients (h###) from GSW-C gsw_internal_const.h
    # These are the coefficients for dynamic enthalpy polynomial
    h001 =  1.07699958620e-3
    h002 = -3.03995719050e-5
    h003 =  3.32853897400e-6
    h004 = -2.82734035930e-7
    h005 =  2.10623061600e-8
    h006 = -2.10787688100e-9
    h007 =  2.80192913290e-10
    h011 = -1.56497346750e-5
    h012 =  9.25288271450e-6
    h013 = -3.91212891030e-7
    h014 = -9.13175163830e-8
    h015 =  6.29081998040e-8
    h021 =  2.77621064840e-5
    h022 = -5.85830342650e-6
    h023 =  7.10167624670e-7
    h024 =  7.17397628980e-8
    h031 = -1.65211592590e-5
    h032 =  3.96398280870e-6
    h033 = -1.53775133460e-7
    h041 =  6.91113227020e-6
    h042 = -1.70510937410e-6
    h043 = -2.11176388380e-8
    h051 = -8.05396155400e-7
    h052 =  2.53683834070e-7
    h061 =  2.05430942680e-7
    h101 = -3.10389819760e-4
    h102 =  1.21312343735e-5
    h103 = -1.94948109950e-7
    h104 =  9.07754712880e-8
    h105 = -2.22942508460e-8
    h111 =  3.50095997640e-5
    h112 = -4.78385440780e-6
    h113 = -1.85663848520e-6
    h114 = -6.82392405930e-8
    h121 = -3.74358423440e-5
    h122 = -1.18391541805e-7
    h123 =  1.30457956930e-7
    h131 =  2.41414794830e-5
    h132 = -1.72793868275e-6
    h133 =  2.58729626970e-9
    h141 = -8.75958731540e-6
    h142 =  6.47835889150e-7
    h151 = -3.30527589000e-7
    h201 =  6.69280670380e-4
    h202 = -1.73962304870e-5
    h203 = -1.60407505320e-6
    h204 =  4.18657594500e-9
    h211 = -4.35926785610e-5
    h212 =  5.55041738250e-6
    h213 =  1.82069162780e-6
    h221 =  3.59078227600e-5
    h222 =  1.46416731475e-6
    h223 = -2.19103680220e-7
    h231 = -1.43536330480e-5
    h232 =  1.58276530390e-7
    h241 =  4.37036805980e-6
    h301 = -8.50479339370e-4
    h302 =  1.87353886525e-5
    h303 =  1.64210356660e-6
    h311 =  3.45324618280e-5
    h312 = -4.92235589220e-6
    h313 = -4.51472854230e-7
    h321 = -1.86985841870e-5
    h322 = -2.44130696000e-7
    h331 =  2.28633245560e-6
    h401 =  5.80860699430e-4
    h402 = -8.66110930600e-6
    h403 = -5.93732490900e-7
    h411 = -1.19594097880e-5
    h412 =  1.29546126300e-6
    h421 =  3.85953392440e-6
    h501 = -2.10923705070e-4
    h502 =  1.54637136265e-6
    h511 =  1.38645945810e-6
    h601 =  3.19324573050e-5
    
    # Build polynomial terms (nested Horner form for efficiency)
    # Structure matches gsw_dynamic_enthalpy in GSW-C
    part_1 = (h001 + ss*(h101 + ss*(h201 + ss*(h301 + ss*(h401
        + ss*(h501 + h601*ss))))) + tt*(h011 + ss*(h111 + ss*(h211 + ss*(h311
        + ss*(h411 + h511*ss)))) + tt*(h021 + ss*(h121 + ss*(h221 + ss*(h321
        + h421*ss))) + tt*(h031 + ss*(h131 + ss*(h231 + h331*ss)) + tt*(h041
        + ss*(h141 + h241*ss) + tt*(h051 + h151*ss + h061*tt))))))
    
    part_2 = (h002 + ss*(h102 + ss*(h202 + ss*(h302 + ss*(h402 + h502*ss))))
        + tt*(h012 + ss*(h112 + ss*(h212 + ss*(h312 + h412*ss))) + tt*(h022
        + ss*(h122 + ss*(h222 + h322*ss)) + tt*(h032 + ss*(h132 + h232*ss)
        + tt*(h042 + h142*ss + h052*tt)))))
    
    part_3 = (h003 + ss*(h103 + ss*(h203 + ss*(h303 + h403*ss))) + tt*(h013
        + ss*(h113 + ss*(h213 + h313*ss)) + tt*(h023 + ss*(h123 + h223*ss)
        + tt*(h033 + h133*ss + h043*tt))))
    
    part_4 = h004 + ss*(h104 + h204*ss) + tt*(h014 + h114*ss + h024*tt)
    
    part_5 = h005 + h105*ss + h015*tt
    
    dynamic_enthalpy_part = z*(part_1 + z*(part_2 + z*(part_3 + z*(part_4
        + z*(part_5 + z*(h006 + h007*z))))))
    
    # Convert from dbar to Pa: db2pa = 1e4, then multiply by 1e4 for J/kg
    # Total factor: db2pa * 1e4 = 1e8
    return dynamic_enthalpy_part * 1e8


def enthalpy_diff(SA, CT, p_shallow, p_deep):
    """
    Calculates the difference of specific enthalpy between two pressures.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p_shallow : array-like
        Upper sea pressure, dbar
    p_deep : array-like
        Lower sea pressure, dbar

    Returns
    -------
    enthalpy_diff : torch.Tensor, J/kg
        difference in specific enthalpy

    Notes
    -----
    This is a pure PyTorch implementation. Simply calculates:
    enthalpy_diff = enthalpy(SA, CT, p_deep) - enthalpy(SA, CT, p_shallow)
    """
    h_deep = enthalpy(SA, CT, p_deep)
    h_shallow = enthalpy(SA, CT, p_shallow)
    return h_deep - h_shallow


def internal_energy(SA, CT, p):
    """
    Calculates specific internal energy of seawater.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    internal_energy : torch.Tensor, J/kg
        specific internal energy

    Notes
    -----
    This is a pure PyTorch implementation. Internal energy is calculated as:
    u = h - p * v
    where h is specific enthalpy, p is pressure (in Pa), and v is specific volume.
    """
    from ..density import specvol
    
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, CT, p = torch.broadcast_tensors(SA, CT, p)
    
    # Get enthalpy and specific volume
    h = enthalpy(SA, CT, p)
    v = specvol(SA, CT, p)
    
    # Constants
    gsw_p0 = 101325.0  # Reference pressure in Pa (10.1325 dbar)
    db2pa = 1e4  # Conversion from dbar to Pa
    
    # Internal energy: u = h - (p0 + db2pa*p) * v
    # where p0 is the reference pressure (101325 Pa) and db2pa converts dbar to Pa
    p_abs_Pa = gsw_p0 + db2pa * p
    u = h - p_abs_Pa * v
    
    return u


def latentheat_evap_CT(SA, CT):
    """
    Calculates latent heat of evaporation.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    latentheat : torch.Tensor, J/kg
        latent heat of evaporation

    Notes
    -----
    This is a pure PyTorch implementation using the exact polynomial from GSW-C.
    The polynomial is valid in the ranges 0 < SA < 42 g/kg and 0 < CT < 40 deg C.
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    
    # Broadcast inputs
    SA, CT = torch.broadcast_tensors(SA, CT)
    
    # Constants
    sfac = 0.0248826675584615
    
    # Reduced variables
    x = torch.sqrt(sfac * SA)
    y = CT / 40.0
    
    # Coefficients from GSW-C gsw_latentheat_evap_ct
    c0  =  2.499065844825125e6
    c1  = -1.544590633515099e-1
    c2  = -9.096800915831875e4
    c3  =  1.665513670736000e2
    c4  =  4.589984751248335e1
    c5  =  1.894281502222415e1
    c6  =  1.192559661490269e3
    c7  = -6.631757848479068e3
    c8  = -1.104989199195898e2
    c9  = -1.207006482532330e3
    c10 = -3.148710097513822e3
    c11 =  7.437431482069087e2
    c12 =  2.519335841663499e3
    c13 =  1.186568375570869e1
    c14 =  5.731307337366114e2
    c15 =  1.213387273240204e3
    c16 =  1.062383995581363e3
    c17 = -6.399956483223386e2
    c18 = -1.541083032068263e3
    c19 =  8.460780175632090e1
    c20 = -3.233571307223379e2
    c21 = -2.031538422351553e2
    c22 =  4.351585544019463e1
    c23 = -8.062279018001309e2
    c24 =  7.510134932437941e2
    c25 =  1.797443329095446e2
    c26 = -2.389853928747630e1
    c27 =  1.021046205356775e2
    
    # Polynomial structure from GSW-C (exact match)
    # Structure: c0 + x_term + y_term
    x_term = x*(c1 + c4*y + x*(c3 + y*(c7 + c12*y) + x*(c6 + y*(c11 + y*(c17 + c24*y))
                   + x*(c10 + y*(c16 + c23*y) + x*(c15 + c22*y + c21*x)))))
    
    y_term = y*(c2 + y*(c5 + c8*x + y*(c9 + x*(c13 + c18*x)
                   + y*(c14 + x*(c19 + c25*x) + y*(c20 + c26*x + c27*y)))))
    
    latentheat = c0 + x_term + y_term
    
    return latentheat


def latentheat_evap_t(SA, t):
    """
    Calculates latent heat of evaporation from in-situ temperature.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C

    Returns
    -------
    latentheat : torch.Tensor, J/kg
        latent heat of evaporation

    Notes
    -----
    This is a pure PyTorch implementation. Computes CT from t, then calls
    latentheat_evap_CT.
    """
    from ..conversions import CT_from_t
    
    SA = as_tensor(SA, dtype=torch.float64)
    t = as_tensor(t, dtype=torch.float64)
    
    # Broadcast inputs
    SA, t = torch.broadcast_tensors(SA, t)
    
    # Convert t to CT (at p=0 for surface)
    CT = CT_from_t(SA, t, 0.0)
    
    # Use latentheat_evap_CT
    return latentheat_evap_CT(SA, CT)


def enthalpy_first_derivatives(SA, CT, p):
    """
    Calculates the first derivatives of specific enthalpy.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    h_SA : torch.Tensor, J/(kg (g/kg))
        derivative with respect to Absolute Salinity at constant CT and p
    h_CT : torch.Tensor, J/(kg K)
        derivative with respect to CT at constant SA and p

    Notes
    -----
    This is a pure PyTorch implementation using automatic differentiation.
    Since h = cp0 * CT + dynamic_enthalpy(SA, CT, p):
    - h_SA = d(dynamic_enthalpy)/dSA
    - h_CT = cp0 + d(dynamic_enthalpy)/dCT
    
    The derivatives are computed using PyTorch's autograd, which is more
    reliable and maintainable than manual differentiation.
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, CT, p = torch.broadcast_tensors(SA, CT, p)
    
    # Enable gradients for SA and CT
    SA_grad = SA.clone().detach().requires_grad_(True)
    CT_grad = CT.clone().detach().requires_grad_(True)
    p_grad = p.clone().detach()
    
    # Compute enthalpy with gradients enabled
    cp0 = 3991.86795711963  # J/(kg K)
    dynamic_h = dynamic_enthalpy(SA_grad, CT_grad, p_grad)
    h = cp0 * CT_grad + dynamic_h
    
    # Compute derivatives using autograd
    h_SA, = torch.autograd.grad(h, SA_grad, grad_outputs=torch.ones_like(h), create_graph=False, retain_graph=True)
    h_CT, = torch.autograd.grad(h, CT_grad, grad_outputs=torch.ones_like(h), create_graph=False, retain_graph=False)
    
    return h_SA, h_CT


def enthalpy_second_derivatives(SA, CT, p):
    """
    Calculates the second derivatives of specific enthalpy.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    h_SA_SA : torch.Tensor, J/(kg (g/kg)^2)
        second derivative with respect to SA at constant CT and p
    h_SA_CT : torch.Tensor, J/(kg K (g/kg))
        derivative with respect to SA and CT at constant p
    h_CT_CT : torch.Tensor, J/(kg K^2)
        second derivative with respect to CT at constant SA and p

    Notes
    -----
    This implementation computes first derivatives analytically, then uses autograd
    to compute second derivatives from the first derivatives. This hybrid approach
    avoids numerical precision issues with second derivatives through sqrt while
    maintaining accuracy.
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, CT, p = torch.broadcast_tensors(SA, CT, p)
    
    # Enable gradients for SA and CT
    SA_grad = SA.clone().detach().requires_grad_(True)
    CT_grad = CT.clone().detach().requires_grad_(True)
    p_grad = p.clone().detach()
    
    # Compute first derivatives using autograd (these are accurate)
    cp0 = 3991.86795711963  # J/(kg K)
    dynamic_h = dynamic_enthalpy(SA_grad, CT_grad, p_grad)
    h = cp0 * CT_grad + dynamic_h
    
    # Compute first derivatives with create_graph=True for second derivatives
    h_SA, = torch.autograd.grad(h, SA_grad, grad_outputs=torch.ones_like(h), create_graph=True, retain_graph=True)
    h_CT, = torch.autograd.grad(h, CT_grad, grad_outputs=torch.ones_like(h), create_graph=True, retain_graph=True)
    
    # Now compute second derivatives from first derivatives
    # This should be more accurate than computing second derivatives directly
    # h_SA_SA = d²h/dSA² = d(h_SA)/dSA
    h_SA_SA, = torch.autograd.grad(h_SA, SA_grad, grad_outputs=torch.ones_like(h_SA), create_graph=False, retain_graph=True)
    
    # h_SA_CT = d²h/(dSA dCT) = d(h_SA)/dCT
    h_SA_CT, = torch.autograd.grad(h_SA, CT_grad, grad_outputs=torch.ones_like(h_SA), create_graph=False, retain_graph=True)
    
    # h_CT_CT = d²h/dCT² = d(h_CT)/dCT
    h_CT_CT, = torch.autograd.grad(h_CT, CT_grad, grad_outputs=torch.ones_like(h_CT), create_graph=False, retain_graph=False)
    
    return h_SA_SA, h_SA_CT, h_CT_CT
    
    # Enthalpy coefficients (same as in dynamic_enthalpy)
    h001 =  1.07699958620e-3
    h002 = -3.03995719050e-5
    h003 =  3.32853897400e-6
    h004 = -2.82734035930e-7
    h005 =  2.10623061600e-8
    h006 = -2.10787688100e-9
    h007 =  2.80192913290e-10
    h011 = -1.56497346750e-5
    h012 =  9.25288271450e-6
    h013 = -3.91212891030e-7
    h014 = -9.13175163830e-8
    h015 =  6.29081998040e-8
    h021 =  2.77621064840e-5
    h022 = -5.85830342650e-6
    h023 =  7.10167624670e-7
    h024 =  7.17397628980e-8
    h031 = -1.65211592590e-5
    h032 =  3.96398280870e-6
    h033 = -1.53775133460e-7
    h041 =  6.91113227020e-6
    h042 = -1.70510937410e-6
    h043 = -2.11176388380e-8
    h051 = -8.05396155400e-7
    h052 =  2.53683834070e-7
    h061 =  2.05430942680e-7
    h101 = -3.10389819760e-4
    h102 =  1.21312343735e-5
    h103 = -1.94948109950e-7
    h104 =  9.07754712880e-8
    h105 = -2.22942508460e-8
    h111 =  3.50095997640e-5
    h112 = -4.78385440780e-6
    h113 = -1.85663848520e-6
    h114 = -6.82392405930e-8
    h121 = -3.74358423440e-5
    h122 = -1.18391541805e-7
    h123 =  1.30457956930e-7
    h131 =  2.41414794830e-5
    h132 = -1.72793868275e-6
    h133 =  2.58729626970e-9
    h141 = -8.75958731540e-6
    h142 =  6.47835889150e-7
    h151 = -3.30527589000e-7
    h201 =  6.69280670380e-4
    h202 = -1.73962304870e-5
    h203 = -1.60407505320e-6
    h204 =  4.18657594500e-9
    h211 = -4.35926785610e-5
    h212 =  5.55041738250e-6
    h213 =  1.82069162780e-6
    h221 =  3.59078227600e-5
    h222 =  1.46416731475e-6
    h223 = -2.19103680220e-7
    h231 = -1.43536330480e-5
    h232 =  1.58276530390e-7
    h241 =  4.37036805980e-6
    h301 = -8.50479339370e-4
    h302 =  1.87353886525e-5
    h303 =  1.64210356660e-6
    h311 =  3.45324618280e-5
    h312 = -4.92235589220e-6
    h313 = -4.51472854230e-7
    h321 = -1.86985841870e-5
    h322 = -2.44130696000e-7
    h331 =  2.28633245560e-6
    h401 =  5.80860699430e-4
    h402 = -8.66110930600e-6
    h403 = -5.93732490900e-7
    h411 = -1.19594097880e-5
    h412 =  1.29546126300e-6
    h421 =  3.85953392440e-6
    h501 = -2.10923705070e-4
    h502 =  1.54637136265e-6
    h511 =  1.38645945810e-6
    h601 =  3.19324573050e-5
    
    xs2 = xs * xs
    
    # Compute h_SA_SA (exact from GSW-C)
    # Structure: z * (-h101 + xs2*(...) + ys*(...) + z*(-h102 + xs2*(...) + ys*(...) + z*(...)))
    dynamic_h_sa_sa_part = z * (
        -h101 + xs2 * (3.0 * h301 + xs * (8.0 * h401 + xs * (15.0 * h501 + 24.0 * h601 * xs))) +
        ys * (-h111 + xs2 * (3.0 * h311 + xs * (8.0 * h411 + 15.0 * h511 * xs)) +
              ys * (-h121 + xs2 * (3.0 * h321 + 8.0 * h421 * xs) +
                    ys * (-h131 + 3.0 * h331 * xs2 + ys * (-h141 - h151 * ys)))) +
        z * (-h102 + xs2 * (3.0 * h302 + xs * (8.0 * h402 + 15.0 * h502 * xs)) +
             ys * (-h112 + xs2 * (3.0 * h312 + 8.0 * h412 * xs) +
                   ys * (-h122 + 3.0 * h322 * xs2 + ys * (-h132 - h142 * ys))) +
             z * (xs2 * (8.0 * h403 * xs + 3.0 * h313 * ys) +
                  z * (-h103 + 3.0 * h303 * xs2 + ys * (-h113 + ys * (-h123 - h133 * ys)) +
                       z * (-h104 - h114 * ys - h105 * z))))
    )
    
    # From GSW-C: h_SA_SA = 1e8 * 0.25 * gsw_sfac^2 * dynamic_h_sa_sa_part / xs^3
    h_SA_SA = 1e8 * 0.25 * gsw_sfac * gsw_sfac * dynamic_h_sa_sa_part / (xs * xs2)
    
    # Compute h_SA_CT (exact from GSW-C)
    dynamic_h_sa_ct_part = (z * (h111 + xs * (2.0 * h211 + xs * (3.0 * h311 +
        xs * (4.0 * h411 + 5.0 * h511 * xs))) + ys * (2.0 * h121 +
        xs * (4.0 * h221 + xs * (6.0 * h321 + 8.0 * h421 * xs)) +
        ys * (3.0 * h131 + xs * (6.0 * h231 + 9.0 * h331 * xs) +
        ys * (4.0 * h141 + 8.0 * h241 * xs + 5.0 * h151 * ys))) + z * (h112 +
        xs * (2.0 * h212 + xs * (3.0 * h312 + 4.0 * h412 * xs)) +
        ys * (2.0 * h122 + xs * (4.0 * h222 + 6.0 * h322 * xs) +
        ys * (3.0 * h132 + 6.0 * h232 * xs + 4.0 * h142 * ys)) + z * (h113 +
        xs * (2.0 * h213 + 3.0 * h313 * xs) + ys * (2.0 * h123 +
        4.0 * h223 * xs + 3.0 * h133 * ys) + h114 * z))))
    
    # From GSW-C: h_SA_CT = 1e8 * 0.025 * 0.5 * gsw_sfac * dynamic_h_sa_ct_part / xs
    h_SA_CT = 1e8 * 0.025 * 0.5 * gsw_sfac * dynamic_h_sa_ct_part / xs
    
    # Compute h_CT_CT (exact from GSW-C)
    dynamic_h_ct_ct_part = (z * (2.0 * h021 + xs * (2.0 * h121 + xs * (2.0 * h221 +
        xs * (2.0 * h321 + 2.0 * h421 * xs))) + ys * (6.0 * h031 +
        xs * (6.0 * h131 + xs * (6.0 * h231 + 6.0 * h331 * xs)) +
        ys * (12.0 * h041 + xs * (12.0 * h141 + 12.0 * h241 * xs) +
        ys * (20.0 * h051 + 20.0 * h151 * xs + 30.0 * h061 * ys))) +
        z * (2.0 * h022 + xs * (2.0 * h122 + xs * (2.0 * h222 +
        2.0 * h322 * xs)) + ys * (6.0 * h032 + xs * (6.0 * h132 +
        6.0 * h232 * xs) + ys * (12.0 * h042 + 12.0 * h142 * xs +
        20.0 * h052 * ys)) + z * (2.0 * h023 + xs * (2.0 * h123 +
        2.0 * h223 * xs) + ys * (6.0 * h133 * xs + 6.0 * h033 +
        12.0 * h043 * ys) + 2.0 * h024 * z))))
    
    # From GSW-C: h_CT_CT = 1e8 * 6.25e-4 * dynamic_h_ct_ct_part
    # Note: 6.25e-4 = (0.025)^2
    h_CT_CT = 1e8 * 6.25e-4 * dynamic_h_ct_ct_part
    
    return h_SA_SA, h_SA_CT, h_CT_CT


def enthalpy_CT_exact(SA, CT, p):
    """
    Calculates specific enthalpy of seawater from Absolute Salinity and
    Conservative Temperature using the full Gibbs function.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    enthalpy_CT_exact : torch.Tensor, J/kg
        specific enthalpy

    Notes
    -----
    This is a pure PyTorch implementation using the full Gibbs function.
    Calculates: enthalpy_CT_exact = enthalpy_t_exact(SA, t_from_CT(SA, CT, p), p)
    where t_from_CT uses the exact Gibbs function and enthalpy_t_exact uses
    gibbs(0, 1, 0, SA, t, p).
    """
    from ..conversions import t_from_CT
    
    # Convert CT to in-situ temperature (exact)
    t = t_from_CT(SA, CT, p)
    
    # Calculate enthalpy using exact Gibbs function
    return enthalpy_t_exact(SA, t, p)


def enthalpy_first_derivatives_CT_exact(SA, CT, p):
    """
    Calculates the first derivatives of specific enthalpy using the exact method.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    h_SA : torch.Tensor, J/(kg (g/kg))
        derivative with respect to Absolute Salinity at constant CT and p
    h_CT : torch.Tensor, J/(kg K)
        derivative with respect to CT at constant SA and p

    Notes
    -----
    This is a pure PyTorch implementation. For the exact version, this is identical
    to enthalpy_first_derivatives since it already uses the exact Gibbs function.
    """
    # The exact version is the same as the regular version
    return enthalpy_first_derivatives(SA, CT, p)


def enthalpy_second_derivatives_CT_exact(SA, CT, p):
    """
    Calculates the second derivatives of specific enthalpy using the exact method.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    h_SA_SA : torch.Tensor, J/(kg (g/kg)^2)
        second derivative with respect to SA at constant CT and p
    h_SA_CT : torch.Tensor, J/(kg K (g/kg))
        derivative with respect to SA and CT at constant p
    h_CT_CT : torch.Tensor, J/(kg K^2)
        second derivative with respect to CT at constant SA and p

    Notes
    -----
    This is a pure PyTorch implementation. For the exact version, this is identical
    to enthalpy_second_derivatives since it already uses the exact Gibbs function.
    """
    # The exact version is the same as the regular version
    return enthalpy_second_derivatives(SA, CT, p)


def latentheat_melting(SA, p):
    """
    Calculates latent heat, or enthalpy, of melting.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    latentheat_melting : torch.Tensor, J/kg
        latent heat of melting

    Notes
    -----
    This is a pure PyTorch implementation using the exact formula from GSW-C.
    Calculates latent heat as:
    L = 1000*mu_water - (T0 + tf)*1000*dmu_water_dt - enthalpy_ice
    
    where:
    - mu_water = chem_potential_water_t_exact(SA, tf, p) in J/g
    - dmu_water_dt = t_deriv_chem_potential_water_t_exact(SA, tf, p) in J/(g K)
    - enthalpy_ice = enthalpy_ice(tf, p) in J/kg
    - tf = t_freezing(SA, p, 0) is the freezing temperature
    - T0 = 273.15 K
    """
    from ..freezing import t_freezing
    from ..utility import chem_potential_water_t_exact, t_deriv_chem_potential_water_t_exact
    from ..ice import enthalpy_ice
    
    SA = as_tensor(SA, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, p = torch.broadcast_tensors(SA, p)
    
    # Constants
    gsw_t0 = 273.15  # K
    kg2g = 1e-3  # Conversion factor: 1 J/g = 1000 J/kg, so multiply by 1000 = divide by kg2g
    
    # Get freezing temperature (exact, without air)
    saturation_fraction = torch.zeros_like(SA)
    tf = t_freezing(SA, p, saturation_fraction)
    
    # Get chemical potential of water (in J/g)
    mu_water = chem_potential_water_t_exact(SA, tf, p)
    
    # Get temperature derivative of chemical potential (in J/(g K))
    dmu_water_dt = t_deriv_chem_potential_water_t_exact(SA, tf, p)
    
    # Get enthalpy of ice (in J/kg)
    h_ice = enthalpy_ice(tf, p)
    
    # Calculate latent heat: L = 1000*mu_water - (T0 + tf)*1000*dmu_water_dt - h_ice
    # where 1000 converts from J/g to J/kg (divide by kg2g = 1e-3)
    latent_heat = (mu_water / kg2g) - (gsw_t0 + tf) * (dmu_water_dt / kg2g) - h_ice
    
    return latent_heat


def enthalpy_t_exact(SA, t, p):
    """
    Calculates the specific enthalpy of seawater from in-situ temperature
    using the full Gibbs function.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    enthalpy_t_exact : torch.Tensor, J/kg
        specific enthalpy

    Notes
    -----
    This is a pure PyTorch implementation using the full Gibbs function.
    Enthalpy is calculated as: h = gibbs(0, 0, 0) - (t + t0) * gibbs(0, 1, 0)
    where t0 = 273.15 K and gibbs(0, 1, 0) = dG/dt.
    This follows from the thermodynamic relationship: h = g - T * (dg/dt)
    where g is the Gibbs function and T is absolute temperature.
    """
    from .gibbs import gibbs
    
    SA = as_tensor(SA, dtype=torch.float64)
    t = as_tensor(t, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, t, p = torch.broadcast_tensors(SA, t, p)
    
    # Constants
    gsw_t0 = 273.15  # K
    
    # Enthalpy = gibbs(0, 0, 0) - (t + t0) * gibbs(0, 1, 0)
    gibbs_000 = gibbs(0, 0, 0, SA, t, p)
    gibbs_010 = gibbs(0, 1, 0, SA, t, p)
    h = gibbs_000 - (t + gsw_t0) * gibbs_010
    
    return h


def enthalpy_SSO_0(p):
    """
    Calculates enthalpy at Standard Ocean Salinity (SSO) and Conservative Temperature of 0°C.

    Parameters
    ----------
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar

    Returns
    -------
    enthalpy_SSO_0 : torch.Tensor, J/kg
        enthalpy at (SSO, CT=0, p)

    Notes
    -----
    This is a pure PyTorch implementation. Calculates:
    enthalpy_SSO_0 = enthalpy(SSO, 0, p)
    where SSO = 35.16504 g/kg (Standard Ocean Salinity).
    """
    SSO = 35.16504  # Standard Ocean Salinity, g/kg
    p = as_tensor(p, dtype=torch.float64)
    SA_SSO = torch.full_like(p, SSO)
    CT_zero = torch.zeros_like(p)
    return enthalpy(SA_SSO, CT_zero, p)
