"""
Functions not specific to the TEOS-10 realm of variables.
"""

import torch

from ._core.utility import O2sol, O2sol_SP_pt, chem_potential_water_t_exact, t_deriv_chem_potential_water_t_exact
from ._utilities import indexer, match_args_return

__all__ = ["pchip_interp", "O2sol", "O2sol_SP_pt", "chem_potential_water_t_exact", "t_deriv_chem_potential_water_t_exact"]


def _pchip_interp_core(xgood, ygood, xi):
    """
    Core PCHIP interpolation function.
    
    Implements Piecewise Cubic Hermite Interpolating Polynomial (PCHIP)
    based on Fritsch & Butland (1984) algorithm. This is a shape-preserving
    interpolation method that maintains monotonicity.
    
    Parameters
    ----------
    xgood : torch.Tensor
        Sorted x values (monotonically increasing)
    ygood : torch.Tensor
        Corresponding y values
    xi : torch.Tensor
        Points to interpolate at
        
    Returns
    -------
    yi : torch.Tensor
        Interpolated y values at xi
    """
    import torch
    
    # Handle edge cases
    if len(xgood) < 2:
        # Not enough points for interpolation
        return torch.full_like(xi, float('nan'), dtype=torch.float64, device=xgood.device)
    
    if len(xgood) == 2:
        # Linear interpolation for 2 points
        x0, x1 = xgood[0], xgood[1]
        y0, y1 = ygood[0], ygood[1]
        t = (xi - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)
    
    # Calculate slopes between consecutive points
    h = xgood[1:] - xgood[:-1]  # Differences in x
    delta = (ygood[1:] - ygood[:-1]) / h  # Slopes
    
    # Calculate derivatives at interior points using weighted harmonic mean
    # This is the key part of PCHIP algorithm
    n = len(xgood)
    d = torch.zeros(n, dtype=torch.float64, device=xgood.device)
    
    # Handle interior points
    for i in range(1, n - 1):
        if delta[i-1] * delta[i] <= 0:
            # Slopes have opposite signs or one is zero - set derivative to zero
            d[i] = 0.0
        else:
            # Weighted harmonic mean
            w1 = 2 * h[i] + h[i-1]
            w2 = h[i] + 2 * h[i-1]
            d[i] = (w1 + w2) / (w1 / delta[i-1] + w2 / delta[i])
    
    # Handle endpoints
    # Left endpoint: one-sided difference
    if n > 2:
        h0 = h[0]
        h1 = h[1]
        delta0 = delta[0]
        delta1 = delta[1]
        
        if (delta0 * delta1 <= 0) or (abs(delta0) > 2 * abs(delta1)) or (abs(delta1) > 2 * abs(delta0)):
            d[0] = 0.0
        else:
            d[0] = (3 * delta0 - d[1]) / 2.0
            # Clamp to reasonable range
            if (d[0] * delta0 < 0):
                d[0] = 0.0
            elif (abs(d[0]) > 3 * abs(delta0)):
                d[0] = 3 * delta0
        
        # Right endpoint: one-sided difference
        hn_2 = h[-2]
        hn_1 = h[-1]
        deltan_2 = delta[-2]
        deltan_1 = delta[-1]
        
        if (deltan_2 * deltan_1 <= 0) or (abs(deltan_1) > 2 * abs(deltan_2)) or (abs(deltan_2) > 2 * abs(deltan_1)):
            d[-1] = 0.0
        else:
            d[-1] = (3 * deltan_1 - d[-2]) / 2.0
            # Clamp to reasonable range
            if (d[-1] * deltan_1 < 0):
                d[-1] = 0.0
            elif (abs(d[-1]) > 3 * abs(deltan_1)):
                d[-1] = 3 * deltan_1
    else:
        # Only 2 points - use linear interpolation derivatives
        d[0] = delta[0]
        d[-1] = delta[0]
    
    # Now interpolate at xi using Hermite cubic polynomials
    yi = torch.zeros_like(xi, dtype=torch.float64)
    
    # Find which interval each xi falls into
    for j, x_val in enumerate(xi):
        # Handle extrapolation (use end values)
        if x_val <= xgood[0]:
            yi[j] = ygood[0]
        elif x_val >= xgood[-1]:
            yi[j] = ygood[-1]
        else:
            # Find the interval [xgood[i], xgood[i+1]] containing x_val
            # Binary search would be better, but linear search is simpler
            i = 0
            for k in range(len(xgood) - 1):
                if xgood[k] <= x_val <= xgood[k+1]:
                    i = k
                    break
            
            # Hermite cubic interpolation in interval [xgood[i], xgood[i+1]]
            x0, x1 = xgood[i], xgood[i+1]
            y0, y1 = ygood[i], ygood[i+1]
            d0, d1 = d[i], d[i+1]
            
            # Normalized position in interval
            t = (x_val - x0) / (x1 - x0)
            
            # Hermite cubic basis functions
            h00 = (1 + 2*t) * (1 - t)**2
            h10 = t * (1 - t)**2
            h01 = t**2 * (3 - 2*t)
            h11 = t**2 * (t - 1)
            
            # Interpolated value
            yi[j] = h00 * y0 + h10 * (x1 - x0) * d0 + h01 * y1 + h11 * (x1 - x0) * d1
    
    return yi


@match_args_return
def pchip_interp(x, y, xi, axis=0):
    """
    Interpolate using Piecewise Cubic Hermite Interpolating Polynomial

    This is a shape-preserving algorithm; it does not introduce new local
    extrema.  The implementation in C that is wrapped here is largely taken
    from the scipy implementation,
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html.

    Points outside the range of the interpolation table are filled using the
    end values in the table.  (In contrast,
    scipy.interpolate.pchip_interpolate() extrapolates using the end
    polynomials.)

    Parameters
    ----------
    x, y : array-like
        Interpolation table x and y; n-dimensional, must be broadcastable to
        the same dimensions.
    xi : array-like
        One-dimensional array of new x values.
    axis : int, optional, default is 0
        Axis along which xi is taken.

    Returns
    -------
    yi : torch.Tensor
        Values of y interpolated to xi along the specified axis.
    """
    xi = torch.as_tensor(xi, dtype=torch.float64)
    if xi.ndim > 1:
        raise ValueError("xi must be no more than 1-dimensional")
    nxi = xi.numel()
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    x, y = torch.broadcast_tensors(x, y)
    out_shape = list(x.shape)
    out_shape[axis] = nxi
    yi = torch.full(tuple(out_shape), float("nan"), dtype=torch.float64, device=x.device)

    goodmask = ~(torch.isnan(x) | torch.isnan(y))

    order = "C"  # Default order
    for ind in indexer(y.shape, axis, order=order):
        igood = goodmask[ind]
        xgood = x[ind][igood]
        ygood = y[ind][igood]

        yi[ind] = _pchip_interp_core(xgood, ygood, xi)

    return yi
