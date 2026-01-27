"""
Helper functions for computing specific Gibbs function derivatives at p=0.

These functions implement gibbs(0,0,0,SA,pt,0) and gibbs(0,1,0,SA,pt,0)
which are needed for exact CT_from_pt calculation.
"""

import torch
from .._utilities import as_tensor

__all__ = [
    "gibbs_000_zerop",
    "gibbs_010_zerop",
    "gibbs_100_zerop",
    "gibbs_200_zerop",
    "gibbs_110_zerop",
]


def gibbs_000_zerop(SA, pt):
    """
    Computes gibbs(0,0,0,SA,pt,0) - the Gibbs function at p=0.
    
    This is the exact implementation from GSW-C source code.
    For p=0, z=0, so many terms simplify.
    
    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    pt : array-like
        Potential temperature (ITS-90), degrees C
    
    Returns
    -------
    gibbs_000 : torch.Tensor, J/kg
        Gibbs function value at p=0
    """
    SA = as_tensor(SA, dtype=torch.float64)
    pt = as_tensor(pt, dtype=torch.float64)
    
    SA, pt = torch.broadcast_tensors(SA, pt)
    
    # Constants
    sfac = 0.0248826675584615
    gsw_t0 = 273.15
    
    # Reduced variables
    x2 = sfac * SA
    x = torch.sqrt(x2)
    y = pt * 0.025
    z = 0.0  # p = 0, so z = p * 1e-4 = 0
    
    # g03 polynomial (from GSW-C, with z=0)
    g03 = (101.342743139674 + 
        y * (5.90578347909402 +
            y * (-12357.785933039 +
                y * (736.741204151612 +
                    y * (-148.185936433658 +
                        y * (58.0259125842571 +
                            y * (-18.9843846514172 + 
                                y * 3.05081646487967)))))))
    
    # g08 polynomial (from GSW-C, with z=0)
    g08 = (x2 * (1416.27648484197 +
        x * (-2432.14662381794 +
            x * (2025.80115603697 +
                y * (543.835333000098 +
                    y * (-68.5572509204491 +
                        y * (49.3667694856254 +
                            y * (-17.1397577419788 +
                                2.49697009569508 * y)))) +
                x * (-1091.66841042967 - 196.028306689776 * y +
                    x * (374.60123787784 - 48.5891069025409 * x +
                        36.7571622995805 * y))) +
            y * (-493.407510141682 +
                y * (-43.0664675978042 +
                    y * (-10.0227370861875 +
                        y * 0.875600661808945)))) +
        y * (168.072408311545 +
            y * (880.031352997204 +
                y * (-225.267649263401 +
                    y * (91.4260447751259 +
                        y * (-21.6603240875311 +
                            2.13016970847183 * y)))))))
    
    # Add logarithmic term for SA > 0
    # g08 = g08 + x2*(5812.81456626732 + 851.226734946706*y)*log(x)
    # But we need to handle SA=0 case
    log_term = torch.where(
        SA > 0.0,
        x2 * (5812.81456626732 + 851.226734946706 * y) * torch.log(x),
        torch.zeros_like(x2)
    )
    g08 = g08 + log_term
    
    return g03 + g08


def gibbs_010_zerop(SA, pt):
    """
    Computes gibbs(0,1,0,SA,pt,0) - first derivative of Gibbs function 
    with respect to temperature at p=0.
    
    This is the exact implementation from GSW-C source code.
    For p=0, z=0, so many terms simplify.
    
    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    pt : array-like
        Potential temperature (ITS-90), degrees C
    
    Returns
    -------
    gibbs_010 : torch.Tensor, J/(kg K)
        First derivative of Gibbs function with respect to temperature at p=0
    """
    SA = as_tensor(SA, dtype=torch.float64)
    pt = as_tensor(pt, dtype=torch.float64)
    
    SA, pt = torch.broadcast_tensors(SA, pt)
    
    # Constants
    sfac = 0.0248826675584615
    
    # Reduced variables
    x2 = sfac * SA
    x = torch.sqrt(x2)
    y = pt * 0.025
    z = 0.0  # p = 0
    
    # g03 polynomial for (0,1,0) at z=0 (from GSW-C, setting z=0)
    # When z=0, only constant terms remain (terms not multiplied by z)
    g03 = (5.90578347909402 +
        y * (-24715.571866078 +
            y * (2210.2236124548363 +
                y * (-592.743745734632 +
                    y * (290.12956292128547 +
                        y * (-113.90630790850321 +
                            y * 21.35571525415769))))))
    
    # g08 polynomial for (0,1,0) at z=0 (from GSW-C, setting z=0)
    # Structure: g08 = x2 * (constant + x_term + y_term)
    # When z=0, all z terms vanish
    g08 = x2 * (168.072408311545 +
        x * (-493.407510141682 +
            x * (543.835333000098 +
                x * (-196.028306689776 + 36.7571622995805 * x) +
                y * (-137.1145018408982 +
                    y * (148.10030845687618 +
                        y * (-68.5590309679152 +
                            12.4848504784754 * y)))) +
            y * (-86.1329351956084 +
                y * (-30.0682112585625 +
                    y * 3.50240264723578))) +
        y * (1760.062705994408 +
            y * (-675.802947790203 +
                y * (365.7041791005036 +
                    y * (-108.30162043765552 +
                        12.78101825083098 * y)))))
    
    # Add logarithmic term for SA > 0
    # g08 = g08 + 851.226734946706*x2*log(x)
    log_term = torch.where(
        SA > 0.0,
        851.226734946706 * x2 * torch.log(x),
        torch.zeros_like(x2)
    )
    g08 = g08 + log_term
    
    # In GSW-C: return_value = (g03 + g08) * 0.025
    return (g03 + g08) * 0.025


def gibbs_100_zerop(SA, pt):
    """
    Computes gibbs(1,0,0,SA,pt,0) - first derivative of Gibbs function 
    with respect to Absolute Salinity at p=0.
    
    Uses autograd to compute the derivative from gibbs(0,0,0) for accuracy.
    
    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    pt : array-like
        Potential temperature (ITS-90), degrees C
    
    Returns
    -------
    gibbs_100 : torch.Tensor, J/(kg (g/kg))
        First derivative of Gibbs function with respect to SA at p=0
    """
    SA = as_tensor(SA, dtype=torch.float64)
    pt = as_tensor(pt, dtype=torch.float64)
    
    SA, pt = torch.broadcast_tensors(SA, pt)
    
    # Use autograd to compute derivative from gibbs(0,0,0)
    SA_grad = SA.clone().detach().requires_grad_(True)
    pt_grad = pt.clone().detach()
    
    # Compute gibbs(0,0,0) with gradients enabled
    gibbs_000 = gibbs_000_zerop(SA_grad, pt_grad)
    
    # Compute derivative with respect to SA
    gibbs_100, = torch.autograd.grad(
        gibbs_000, 
        SA_grad, 
        grad_outputs=torch.ones_like(gibbs_000),
        create_graph=False,
        retain_graph=False
    )
    
    return gibbs_100


def gibbs_200_zerop(SA, pt):
    """
    Computes gibbs(2,0,0,SA,pt,0) - second derivative of Gibbs function 
    with respect to Absolute Salinity at p=0.
    
    Uses autograd to compute the second derivative from gibbs(0,0,0) for accuracy.
    
    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    pt : array-like
        Potential temperature (ITS-90), degrees C
    
    Returns
    -------
    gibbs_200 : torch.Tensor, J/(kg (g/kg)^2)
        Second derivative of Gibbs function with respect to SA at p=0
    """
    SA = as_tensor(SA, dtype=torch.float64)
    pt = as_tensor(pt, dtype=torch.float64)
    
    SA, pt = torch.broadcast_tensors(SA, pt)
    
    # Use autograd to compute second derivative from gibbs(0,0,0)
    SA_grad = SA.clone().detach().requires_grad_(True)
    pt_grad = pt.clone().detach()
    
    # Compute gibbs(0,0,0) with gradients enabled
    gibbs_000 = gibbs_000_zerop(SA_grad, pt_grad)
    
    # Compute first derivative
    gibbs_100, = torch.autograd.grad(
        gibbs_000,
        SA_grad,
        grad_outputs=torch.ones_like(gibbs_000),
        create_graph=True,
        retain_graph=True
    )
    
    # Compute second derivative
    gibbs_200, = torch.autograd.grad(
        gibbs_100,
        SA_grad,
        grad_outputs=torch.ones_like(gibbs_100),
        create_graph=False,
        retain_graph=False
    )
    
    return gibbs_200


def gibbs_110_zerop(SA, pt):
    """
    Computes gibbs(1,1,0,SA,pt,0) - mixed derivative of Gibbs function 
    with respect to Absolute Salinity and temperature at p=0.
    
    Uses autograd to compute the mixed derivative from gibbs(0,0,0) for accuracy.
    
    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    pt : array-like
        Potential temperature (ITS-90), degrees C
    
    Returns
    -------
    gibbs_110 : torch.Tensor, J/(kg K (g/kg))
        Mixed derivative of Gibbs function with respect to SA and T at p=0
    """
    SA = as_tensor(SA, dtype=torch.float64)
    pt = as_tensor(pt, dtype=torch.float64)
    
    SA, pt = torch.broadcast_tensors(SA, pt)
    
    # Use autograd to compute mixed derivative from gibbs(0,0,0)
    SA_grad = SA.clone().detach().requires_grad_(True)
    pt_grad = pt.clone().detach().requires_grad_(True)
    
    # Compute gibbs(0,0,0) with gradients enabled
    gibbs_000 = gibbs_000_zerop(SA_grad, pt_grad)
    
    # Compute first derivative with respect to SA
    gibbs_100, = torch.autograd.grad(
        gibbs_000,
        SA_grad,
        grad_outputs=torch.ones_like(gibbs_000),
        create_graph=True,
        retain_graph=True
    )
    
    # Compute mixed derivative: d²/dSA dT = d(gibbs_100)/dT
    gibbs_110, = torch.autograd.grad(
        gibbs_100,
        pt_grad,
        grad_outputs=torch.ones_like(gibbs_100),
        create_graph=False,
        retain_graph=False
    )
    
    return gibbs_110
