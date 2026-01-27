"""
Functions for calculating geostrophic currents.
"""

import torch

from ._utilities import indexer, match_args_return
from .conversions import z_from_p

__all__ = [
    "geo_strf_dyn_height",
    "distance",
    "f",
    "geostrophic_velocity",
]


def unwrap(lon, centered=True, copy=True):
    """
    Unwrap a sequence of longitudes or headings in degrees.

    Optionally center it as close to zero as possible

    By default, return a copy; if *copy* is False, avoid a
    copy when possible.

    Parameters
    ----------
    lon : array-like
        Longitude values in degrees
    centered : bool, optional
        If True, center the values near zero
    copy : bool, optional
        If True, return a copy (not used for torch tensors)

    Returns
    -------
    torch.Tensor
        Unwrapped longitude values
    """
    lon = torch.as_tensor(lon, dtype=torch.float64)
    if lon.ndim != 1:
        raise ValueError("Only 1-D sequences are supported")
    if lon.shape[0] < 2:
        return lon

    # Handle nans
    valid_mask = ~torch.isnan(lon)
    if not torch.any(valid_mask):
        return lon

    x = lon[valid_mask]
    if len(x) < 2:
        return lon

    w = torch.zeros(x.shape[0] - 1, dtype=torch.int64, device=x.device)
    ld = torch.diff(x)
    w[ld > 180] = -1
    w[ld < -180] = 1
    x[1:] += w.cumsum(0) * 360.0

    if centered:
        x -= 360 * torch.round(x.mean() / 360.0)

    result = lon.clone()
    result[valid_mask] = x
    return result


@match_args_return
def geo_strf_dyn_height(SA, CT, p, p_ref=0, axis=0, max_dp=1.0, interp_method="pchip"):
    """
    Dynamic height anomaly as a function of pressure.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    p_ref : float or array-like, optional
        Reference pressure, dbar
    axis : int, optional, default is 0
        The index of the pressure dimension in SA and CT.
    max_dp : float
        If any pressure interval in the input p exceeds max_dp, the dynamic
        height will be calculated after interpolating to a grid with this
        spacing.
    interp_method : string {'mrst', 'pchip', 'linear'}
        Interpolation algorithm.

    Returns
    -------
    dynamic_height : torch.Tensor
        This is the integral of specific volume anomaly with respect
        to pressure, from each pressure in p to the specified
        reference pressure.  It is the geostrophic streamfunction
        in an isobaric surface, relative to the reference surface.
    """
    import torch
    from ._core.density import specvol_anom_standard
    from .interpolation import sa_ct_interp
    from .utility import pchip_interp
    
    SA = torch.as_tensor(SA, dtype=torch.float64)
    CT = torch.as_tensor(CT, dtype=torch.float64)
    p = torch.as_tensor(p, dtype=torch.float64)
    p_ref = float(p_ref)
    
    interp_methods = {'mrst': 3, 'pchip': 2, 'linear': 1}
    if interp_method not in interp_methods:
        raise ValueError(f'interp_method must be one of {list(interp_methods.keys())}')
    
    if SA.shape != CT.shape:
        raise ValueError(f'Shapes of SA and CT must match; found {SA.shape} and {CT.shape}')
    
    # Handle 1D p case
    if p.ndim == 1 and SA.ndim > 1:
        if len(p) != SA.shape[axis]:
            raise ValueError(
                f'With 1-D p, len(p) must be SA.shape[axis];\n'
                f' found {len(p)} versus {SA.shape[axis]} on specified axis, {axis}'
            )
        ind = [None] * SA.ndim
        ind[axis] = slice(None)
        p = p[tuple(ind)]
    
    # Check for monotonic pressure
    if torch.any(torch.diff(p, dim=axis) <= 0):
        raise ValueError('p must be increasing along the specified axis')
    
    p = torch.broadcast_to(p, SA.shape)
    goodmask = ~(torch.isnan(SA) | torch.isnan(CT) | torch.isnan(p))
    dh = torch.full(SA.shape, float('nan'), dtype=torch.float64, device=SA.device)
    
    # Constants
    db2pa = 1e4  # dbar to Pa conversion
    
    from ._utilities import indexer
    order = "C"  # Default order
    
    for ind in indexer(SA.shape, axis, order=order):
        igood = goodmask[ind]
        pgood = p[ind][igood]
        
        if len(pgood) > 1 and pgood[-1] >= p_ref:
            sa = SA[ind][igood]
            ct = CT[ind][igood]
            
            # Check if we need to add surface points
            ntop = 0
            if pgood[0] > p_ref:
                # Add points from p_ref to pgood[0] with spacing max_dp
                ptop = torch.arange(p_ref, pgood[0], max_dp, dtype=torch.float64, device=SA.device)
                ntop = len(ptop)
                sa = torch.cat([sa[0:1].expand(ntop), sa])
                ct = torch.cat([ct[0:1].expand(ntop), ct])
                pgood = torch.cat([ptop, pgood])
            
            # Calculate pressure differences
            dp = pgood[1:] - pgood[:-1]
            dp_min = torch.min(dp).item()
            dp_max = torch.max(dp).item()
            
            # Check if we need interpolation
            ipref = -1
            for i in range(len(pgood)):
                if abs(pgood[i].item() - p_ref) < 1e-10:
                    ipref = i
                    break
            
            if dp_max <= max_dp and ipref >= 0:
                # Simple case: no interpolation needed
                # Calculate specific volume anomaly
                b = specvol_anom_standard(sa, ct, pgood)
                
                # Average between consecutive levels
                b_av = 0.5 * (b[1:] + b[:-1])
                
                # Integrate: dyn_height[i] = dyn_height[i-1] - b_av[i-1]*dp[i-1]*db2pa
                dyn_height = torch.zeros_like(pgood)
                for i in range(1, len(pgood)):
                    dyn_height[i] = dyn_height[i-1] - b_av[i-1] * dp[i-1] * db2pa
                
                # Subtract value at reference pressure
                dh_ref = dyn_height[ipref]
                dyn_height = dyn_height - dh_ref
                
                # Extract values for original pressure levels
                if ntop > 0:
                    dh[ind][igood] = dyn_height[ntop:]
                else:
                    dh[ind][igood] = dyn_height
            else:
                # Need interpolation - create refined grid with spacing <= max_dp
                # Create refined pressure grid
                p_refined_list = []
                for i in range(len(pgood) - 1):
                    p_start = pgood[i].item()
                    p_end = pgood[i + 1].item()
                    # Add start point
                    if i == 0 or abs(p_refined_list[-1] - p_start) > 1e-10:
                        p_refined_list.append(p_start)
                    # Add intermediate points with spacing <= max_dp
                    n_intermediate = int((p_end - p_start) / max_dp)
                    if n_intermediate > 0:
                        for j in range(1, n_intermediate + 1):
                            p_intermediate = p_start + j * max_dp
                            if p_intermediate < p_end - 1e-10:
                                p_refined_list.append(p_intermediate)
                    # Add end point
                    p_refined_list.append(p_end)
                
                # Remove duplicates and ensure p_ref is included
                p_refined_list = sorted(set(p_refined_list))
                if p_ref not in p_refined_list and pgood[0].item() <= p_ref <= pgood[-1].item():
                    p_refined_list.append(p_ref)
                    p_refined_list = sorted(p_refined_list)
                
                p_refined = torch.tensor(p_refined_list, dtype=torch.float64, device=sa.device)
                
                # Interpolate SA and CT to refined grid
                if interp_method == 'pchip':
                    sa_refined = pchip_interp(pgood, sa, p_refined)
                    ct_refined = pchip_interp(pgood, ct, p_refined)
                elif interp_method == 'linear':
                    # Linear interpolation
                    sa_refined = torch.zeros_like(p_refined)
                    ct_refined = torch.zeros_like(p_refined)
                    for i in range(len(p_refined)):
                        p_val = p_refined[i].item()
                        # Find interval
                        if p_val <= pgood[0].item():
                            sa_refined[i] = sa[0]
                            ct_refined[i] = ct[0]
                        elif p_val >= pgood[-1].item():
                            sa_refined[i] = sa[-1]
                            ct_refined[i] = ct[-1]
                        else:
                            # Binary search
                            j = 0
                            k = len(pgood) - 1
                            while k - j > 1:
                                m = (j + k) // 2
                                if p_val > pgood[m].item():
                                    j = m
                                else:
                                    k = m
                            # Linear interpolation
                            t = (p_val - pgood[j].item()) / (pgood[k].item() - pgood[j].item())
                            sa_refined[i] = sa[j] + t * (sa[k] - sa[j])
                            ct_refined[i] = ct[j] + t * (ct[k] - ct[j])
                else:  # mrst
                    # Use sa_ct_interp (requires at least 4 points)
                    if len(pgood) >= 4:
                        sa_refined, ct_refined = sa_ct_interp(sa, ct, pgood, p_refined, axis=0)
                    else:
                        # Fall back to PCHIP if not enough points
                        sa_refined = pchip_interp(pgood, sa, p_refined)
                        ct_refined = pchip_interp(pgood, ct, p_refined)
                
                # Calculate specific volume anomaly on refined grid
                b = specvol_anom_standard(sa_refined, ct_refined, p_refined)
                
                # Calculate pressure differences on refined grid
                dp_refined = p_refined[1:] - p_refined[:-1]
                
                # Average between consecutive levels
                b_av = 0.5 * (b[1:] + b[:-1])
                
                # Integrate on refined grid
                dyn_height_refined = torch.zeros_like(p_refined)
                for i in range(1, len(p_refined)):
                    dyn_height_refined[i] = dyn_height_refined[i-1] - b_av[i-1] * dp_refined[i-1] * db2pa
                
                # Find reference pressure index in refined grid
                ipref_refined = -1
                for i in range(len(p_refined)):
                    if abs(p_refined[i].item() - p_ref) < 1e-10:
                        ipref_refined = i
                        break
                
                if ipref_refined >= 0:
                    dh_ref = dyn_height_refined[ipref_refined]
                    dyn_height_refined = dyn_height_refined - dh_ref
                
                # Interpolate back to original pressure levels
                if interp_method == 'pchip':
                    dyn_height = pchip_interp(p_refined, dyn_height_refined, pgood)
                elif interp_method == 'linear':
                    # Linear interpolation back
                    dyn_height = torch.zeros_like(pgood)
                    for i in range(len(pgood)):
                        p_val = pgood[i].item()
                        if p_val <= p_refined[0].item():
                            dyn_height[i] = dyn_height_refined[0]
                        elif p_val >= p_refined[-1].item():
                            dyn_height[i] = dyn_height_refined[-1]
                        else:
                            # Binary search and linear interpolation
                            j = 0
                            k = len(p_refined) - 1
                            while k - j > 1:
                                m = (j + k) // 2
                                if p_val > p_refined[m].item():
                                    j = m
                                else:
                                    k = m
                            t = (p_val - p_refined[j].item()) / (p_refined[k].item() - p_refined[j].item())
                            dyn_height[i] = dyn_height_refined[j] + t * (dyn_height_refined[k] - dyn_height_refined[j])
                else:  # mrst
                    if len(p_refined) >= 4:
                        # Use PCHIP for interpolation back (sa_ct_interp is for SA/CT, not for single variable)
                        dyn_height = pchip_interp(p_refined, dyn_height_refined, pgood)
                    else:
                        dyn_height = pchip_interp(p_refined, dyn_height_refined, pgood)
                
                # Extract values for original pressure levels
                if ntop > 0:
                    dh[ind][igood] = dyn_height[ntop:]
                else:
                    dh[ind][igood] = dyn_height
    
    return dh


@match_args_return
def distance(lon, lat, p=0, axis=-1):
    """
    Great-circle distance in m between lon, lat points.

    Parameters
    ----------
    lon, lat : array-like, 1-D or 2-D (shapes must match)
        Longitude, latitude, in degrees.
    p : array-like, scalar, 1-D or 2-D, optional, default is 0
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    axis : int, -1, 0, 1, optional
        The axis or dimension along which *lat and lon* vary.
        This differs from most functions, for which axis is the
        dimension along which p increases.

    Returns
    -------
    distance : torch.Tensor
        distance in meters between adjacent points.
    """
    earth_radius = 6371e3

    lon = torch.as_tensor(lon, dtype=torch.float64)
    lat = torch.as_tensor(lat, dtype=torch.float64)

    if lon.shape != lat.shape:
        raise ValueError(f"lon, lat shapes must match; found {lon.shape}, {lat.shape}")
    if not (lon.ndim in (1, 2) and lon.shape[axis] > 1):
        raise ValueError(
            "lon, lat must be 1-D or 2-D with more than one point"
            f" along axis; found shape {lon.shape} and axis {axis}"
        )
    if lon.ndim == 1:
        one_d = True
        lon = lon.unsqueeze(0)
        lat = lat.unsqueeze(0)
        axis = -1
    else:
        one_d = False

    # Handle scalar default
    p = torch.as_tensor(p, dtype=torch.float64)
    one_d = one_d and p.ndim == 1

    if axis == 0:
        indm = (slice(0, -1), slice(None))
        indp = (slice(1, None), slice(None))
    else:
        indm = (slice(None), slice(0, -1))
        indp = (slice(None), slice(1, None))

    if torch.all(p == 0):
        z = torch.tensor(0.0, dtype=torch.float64, device=lon.device)
    else:
        lon, lat, p = torch.broadcast_tensors(lon, lat, p)

        p_mid = 0.5 * (p[indm] + p[indp])
        lat_mid = 0.5 * (lat[indm] + lat[indp])

        z = z_from_p(p_mid, lat_mid)

    lon_rad = torch.deg2rad(lon)
    lat_rad = torch.deg2rad(lat)

    dlon = torch.diff(lon_rad, dim=axis)
    dlat = torch.diff(lat_rad, dim=axis)

    a = (
        (torch.sin(dlat / 2)) ** 2
        + torch.cos(lat_rad[indm]) * torch.cos(lat_rad[indp]) * (torch.sin(dlon / 2)) ** 2
    )

    angles = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    distance = (earth_radius + z) * angles

    if one_d:
        distance = distance[0]

    return distance


@match_args_return
def f(lat):
    """
    Coriolis parameter in 1/s for latitude in degrees.

    Parameters
    ----------
    lat : array-like
        Latitude, degrees

    Returns
    -------
    f : torch.Tensor
        Coriolis parameter, 1/s
    """
    omega = 7.292115e-5  # (1/s)   (Groten, 2004).
    lat = torch.as_tensor(lat, dtype=torch.float64)
    return 2 * omega * torch.sin(torch.deg2rad(lat))


@match_args_return
def geostrophic_velocity(geo_strf, lon, lat, p=0, axis=0):
    """
    Calculate geostrophic velocity from a streamfunction.

    Calculates geostrophic velocity relative to a reference pressure,
    given a geostrophic streamfunction and the position of each station
    in sequence along an ocean section.  The data can be from a single
    isobaric or "density" surface, or from a series of such surfaces.

    Parameters
    ----------
    geo_strf : array-like, 1-D or 2-D
        geostrophic streamfunction; see Notes below.
    lon : array-like, 1-D
        Longitude, -360 to 360 degrees
    lat : array-like, 1-D
        Latitude, degrees
    p : float or array-like, optional
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar.
        This used only for a tiny correction in the distance calculation;
        it is safe to omit it.
    axis : int, 0 or 1, optional
        The axis or dimension along which pressure increases in geo_strf.
        If geo_strf is 1-D, it is ignored.

    Returns
    -------
    velocity : torch.Tensor
        Geostrophic velocity in m/s relative to the sea surface,
        averaged between each successive pair of positions.
    mid_lon : torch.Tensor
        Midpoints of input lon.
    mid_lat : torch.Tensor
        Midpoints of input lat.
    """
    lon = unwrap(lon)

    lon = torch.as_tensor(lon, dtype=torch.float64)
    lat = torch.as_tensor(lat, dtype=torch.float64)

    if lon.shape != lat.shape or lon.ndim != 1:
        raise ValueError(
            f"lon, lat must be 1-D and matching; found shapes {lon.shape} and {lat.shape}"
        )

    geo_strf = torch.as_tensor(geo_strf, dtype=torch.float64)
    if geo_strf.ndim not in (1, 2):
        raise ValueError(f"geo_strf must be 1-D or 2-d; found shape {geo_strf.shape}")

    laxis = 0 if axis else -1

    ds = distance(lon, lat, p)

    mid_lon = 0.5 * (lon[:-1] + lon[1:])
    mid_lat = 0.5 * (lat[:-1] + lat[1:])

    u = torch.diff(geo_strf, dim=laxis) / (ds * f(mid_lat))

    return u, mid_lon, mid_lat
