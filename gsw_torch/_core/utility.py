"""
Core PyTorch implementations of utility functions.

These functions include oxygen solubility and other utility functions.
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
    paths_to_try.extend(
        [
            os.path.join(project_root, "reference"),
            os.path.join(current_dir, "../../../reference"),
            ref_direct,
        ]
    )

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
    "O2sol",
    "O2sol_SP_pt",
    "chem_potential_water_t_exact",
    "t_deriv_chem_potential_water_t_exact",
]


def O2sol(SA, CT, p, lon, lat):
    """
    Calculates the oxygen concentration expected at equilibrium with air.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    lon : array-like
        Longitude, -360 to 360 degrees
    lat : array-like
        Latitude, -90 to 90 degrees

    Returns
    -------
    O2sol : torch.Tensor, umol/kg
        solubility of oxygen in micro-moles per kg

    Notes
    -----
    This is a pure PyTorch implementation using the exact polynomial from
    Garcia and Gordon (1992, 1993), based on Benson and Krause (1984) data.
    The formula uses SP_from_SA and pt_from_CT internally.
    """
    from ..conversions import SP_from_SA, pt_from_CT

    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    lon = as_tensor(lon, dtype=torch.float64)
    lat = as_tensor(lat, dtype=torch.float64)

    # Broadcast inputs
    SA, CT, p, lon, lat = torch.broadcast_tensors(SA, CT, p, lon, lat)

    # Constants
    gsw_t0 = 273.15  # K

    # Convert SA to SP and CT to pt
    SP = SP_from_SA(SA, p, lon, lat)
    pt = pt_from_CT(SA, CT)

    # Convert pt to IPTS-68 (pt68 = pt * 1.00024)
    pt68 = pt * 1.00024

    # Calculate y = log((298.15 - pt68) / (gsw_t0 + pt68))
    y = torch.log((298.15 - pt68) / (gsw_t0 + pt68))

    # Polynomial coefficients from Garcia and Gordon (1992, 1993)
    a0 = 5.80871
    a1 = 3.20291
    a2 = 4.17887
    a3 = 5.10006
    a4 = -9.86643e-2
    a5 = 3.80369
    b0 = -7.01577e-3
    b1 = -7.70028e-3
    b2 = -1.13864e-2
    b3 = -9.51519e-3
    c0 = -2.75915e-7

    # Calculate O2 solubility using multiplicative formula:
    # exp(a0 + y*(a1 + y*(a2 + y*(a3 + y*(a4 + a5*y))))) * exp(SP*(b0 + y*(b1 + y*(b2 + b3*y)) + c0*SP))
    # This can be simplified to a single exponential:
    o2sol = torch.exp(
        a0
        + y * (a1 + y * (a2 + y * (a3 + y * (a4 + a5 * y))))
        + SP * (b0 + y * (b1 + y * (b2 + b3 * y)) + c0 * SP)
    )

    return o2sol


def O2sol_SP_pt(SP, pt):
    """
    Calculates the oxygen concentration expected at equilibrium with air
    from Practical Salinity and potential temperature.

    Parameters
    ----------
    SP : array-like
        Practical Salinity (PSS-78), unitless
    pt : array-like
        Potential temperature (ITS-90), degrees C

    Returns
    -------
    O2sol : torch.Tensor, umol/kg
        solubility of oxygen in micro-moles per kg

    Notes
    -----
    This is a pure PyTorch implementation using the exact polynomial from
    Garcia and Gordon (1992, 1993), based on Benson and Krause (1984) data.
    This version takes SP and pt directly, avoiding the SA/CT conversions.
    """
    SP = as_tensor(SP, dtype=torch.float64)
    pt = as_tensor(pt, dtype=torch.float64)

    # Broadcast inputs
    SP, pt = torch.broadcast_tensors(SP, pt)

    # Constants
    gsw_t0 = 273.15  # K

    # Convert pt to IPTS-68 (pt68 = pt * 1.00024)
    pt68 = pt * 1.00024

    # Calculate y = log((298.15 - pt68) / (gsw_t0 + pt68))
    y = torch.log((298.15 - pt68) / (gsw_t0 + pt68))

    # Polynomial coefficients from Garcia and Gordon (1992, 1993)
    a0 = 5.80871
    a1 = 3.20291
    a2 = 4.17887
    a3 = 5.10006
    a4 = -9.86643e-2
    a5 = 3.80369
    b0 = -7.01577e-3
    b1 = -7.70028e-3
    b2 = -1.13864e-2
    b3 = -9.51519e-3
    c0 = -2.75915e-7

    # Calculate O2 solubility using multiplicative formula:
    # exp(a0 + y*(a1 + y*(a2 + y*(a3 + y*(a4 + a5*y))))) * exp(SP*(b0 + y*(b1 + y*(b2 + b3*y)) + c0*SP))
    # This can be simplified to a single exponential:
    o2sol = torch.exp(
        a0
        + y * (a1 + y * (a2 + y * (a3 + y * (a4 + a5 * y))))
        + SP * (b0 + y * (b1 + y * (b2 + b3 * y)) + c0 * SP)
    )

    return o2sol


def chem_potential_water_t_exact(SA, t, p):
    """
    Calculates the chemical potential of water in seawater.

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
    chem_potential_water_t_exact : torch.Tensor, J/g
        chemical potential of water in seawater

    Notes
    -----
    This is a pure PyTorch implementation using the exact Gibbs function
    g03_g and g08_g polynomials. The chemical potential is calculated as:
    chem_potential = kg2g * (g03_g + g08_g - 0.5*x2*g_sa_part)
    where kg2g = 1e-3 converts from J/kg to J/g.
    """
    SA = as_tensor(SA, dtype=torch.float64)
    t = as_tensor(t, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)

    # Broadcast inputs
    SA, t, p = torch.broadcast_tensors(SA, t, p)

    # Constants from GSW_TEOS10_CONSTANTS
    gsw_sfac = 0.0248826675584615  # 1/(40*gsw_ups)
    kg2g = 1e-3

    # Reduced variables
    x2 = gsw_sfac * SA
    # Use numerically stable sqrt to avoid gradient issues at SA=0
    # Add small epsilon to avoid division by zero in gradient computation
    x = torch.sqrt(torch.clamp(x2, min=1e-20))
    y = t * 0.025
    z = p * 1e-4

    # g03_g polynomial (from GSW-C source)
    g03_g = (
        101.342743139674
        + z
        * (
            100015.695367145
            + z
            * (
                -2544.5765420363
                + z
                * (
                    284.517778446287
                    + z * (-33.3146754253611 + (4.20263108803084 - 0.546428511471039 * z) * z)
                )
            )
        )
        + y
        * (
            5.90578347909402
            + z
            * (
                -270.983805184062
                + z
                * (
                    776.153611613101
                    + z * (-196.51255088122 + (28.9796526294175 - 2.13290083518327 * z) * z)
                )
            )
            + y
            * (
                -12357.785933039
                + z
                * (
                    1455.0364540468
                    + z
                    * (
                        -756.558385769359
                        + z * (273.479662323528 + z * (-55.5604063817218 + 4.34420671917197 * z))
                    )
                )
                + y
                * (
                    736.741204151612
                    + z
                    * (
                        -672.50778314507
                        + z
                        * (
                            499.360390819152
                            + z
                            * (-239.545330654412 + (48.8012518593872 - 1.66307106208905 * z) * z)
                        )
                    )
                    + y
                    * (
                        -148.185936433658
                        + z
                        * (
                            397.968445406972
                            + z
                            * (-301.815380621876 + (152.196371733841 - 26.3748377232802 * z) * z)
                        )
                        + y
                        * (
                            58.0259125842571
                            + z
                            * (
                                -194.618310617595
                                + z
                                * (
                                    120.520654902025
                                    + z * (-55.2723052340152 + 6.48190668077221 * z)
                                )
                            )
                            + y
                            * (
                                -18.9843846514172
                                + y * (3.05081646487967 - 9.63108119393062 * z)
                                + z
                                * (
                                    63.5113936641785
                                    + z * (-22.2897317140459 + 8.17060541818112 * z)
                                )
                            )
                        )
                    )
                )
            )
        )
    )

    # g08_g polynomial (from GSW-C source)
    g08_g = x2 * (
        1416.27648484197
        + x
        * (
            -2432.14662381794
            + x
            * (
                2025.80115603697
                + y
                * (
                    543.835333000098
                    + y
                    * (
                        -68.5572509204491
                        + y * (49.3667694856254 + y * (-17.1397577419788 + 2.49697009569508 * y))
                    )
                    - 22.6683558512829 * z
                )
                + x
                * (
                    -1091.66841042967
                    - 196.028306689776 * y
                    + x * (374.60123787784 - 48.5891069025409 * x + 36.7571622995805 * y)
                    + 36.0284195611086 * z
                )
                + z * (-54.7919133532887 + (-4.08193978912261 - 30.1755111971161 * z) * z)
            )
            + z
            * (
                199.459603073901
                + z * (-52.2940909281335 + (68.0444942726459 - 3.41251932441282 * z) * z)
            )
            + y
            * (
                -493.407510141682
                + z * (-175.292041186547 + (83.1923927801819 - 29.483064349429 * z) * z)
                + y
                * (
                    -43.0664675978042
                    + z * (383.058066002476 + z * (-54.1917262517112 + 25.6398487389914 * z))
                    + y
                    * (
                        -10.0227370861875
                        - 460.319931801257 * z
                        + y * (0.875600661808945 + 234.565187611355 * z)
                    )
                )
            )
        )
        + y * (168.072408311545)
    )

    # g_sa_part polynomial (from GSW-C source)
    g_sa_part = (
        8645.36753595126
        + x
        * (
            -7296.43987145382
            + x
            * (
                8103.20462414788
                + y
                * (
                    2175.341332000392
                    + y
                    * (
                        -274.2290036817964
                        + y * (197.4670779425016 + y * (-68.5590309679152 + 9.98788038278032 * y))
                    )
                    - 90.6734234051316 * z
                )
                + x
                * (
                    -5458.34205214835
                    - 980.14153344888 * y
                    + x * (2247.60742726704 - 340.1237483177863 * x + 220.542973797483 * y)
                    + 180.142097805543 * z
                )
                + z * (-219.1676534131548 + (-16.32775915649044 - 120.7020447884644 * z) * z)
            )
            + z
            * (
                598.378809221703
                + z * (-156.8822727844005 + (204.1334828179377 - 10.23755797323846 * z) * z)
            )
            + y
            * (
                -1480.222530425046
                + z * (-525.876123559641 + (249.57717834054571 - 88.449193048287 * z) * z)
                + y
                * (
                    -129.1994027934126
                    + z * (1149.174198007428 + z * (-162.5751787551336 + 76.9195462169742 * z))
                    + y
                    * (
                        -30.0682112585625
                        - 1380.9597954037708 * z
                        + y * (2.626801985426835 + 703.695562834065 * z)
                    )
                )
            )
        )
        + y * (1187.3715515697959)
    )

    # Chemical potential = kg2g * (g03_g + g08_g - 0.5*x2*g_sa_part)
    chem_potential = kg2g * (g03_g + g08_g - 0.5 * x2 * g_sa_part)

    return chem_potential


def t_deriv_chem_potential_water_t_exact(SA, t, p):
    """
    Calculates the temperature derivative of the chemical potential of water
    in seawater so that it is valid at exactly SA = 0.

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
    t_deriv_chem_potential_water_t_exact : torch.Tensor, J g^-1 K^-1
        temperature derivative of the chemical potential of water in seawater

    Notes
    -----
    This is a pure PyTorch implementation using the exact Gibbs function
    derivatives. The derivative is calculated using g03_t and g08_sa_t polynomials.
    Note: The kg2g factor (1e-3) converts from J/(kg K) to J/(g K).
    """
    SA = as_tensor(SA, dtype=torch.float64)
    t = as_tensor(t, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)

    # Broadcast inputs
    SA, t, p = torch.broadcast_tensors(SA, t, p)

    # Constants from GSW_TEOS10_CONSTANTS
    gsw_sfac = 0.0248826675584615  # 1/(40*gsw_ups)
    rec_db2pa = 1e-4  # 1/db2pa where db2pa = 1e4
    kg2g = 1e-3

    # Reduced variables
    x2 = gsw_sfac * SA
    # Use numerically stable sqrt to avoid gradient issues at SA=0
    # Add small epsilon to avoid division by zero in gradient computation
    x = torch.sqrt(torch.clamp(x2, min=1e-20))
    y = t * 0.025
    z = p * rec_db2pa  # Note: z uses rec_db2pa, not 1e-4

    # g03_t polynomial (from GSW-C source)
    g03_t = (
        5.90578347909402
        + z
        * (
            -270.983805184062
            + z
            * (
                776.153611613101
                + z * (-196.51255088122 + (28.9796526294175 - 2.13290083518327 * z) * z)
            )
        )
        + y
        * (
            -24715.571866078
            + z
            * (
                2910.0729080936
                + z
                * (
                    -1513.116771538718
                    + z * (546.959324647056 + z * (-111.1208127634436 + 8.68841343834394 * z))
                )
            )
            + y
            * (
                2210.2236124548363
                + z
                * (
                    -2017.52334943521
                    + z
                    * (
                        1498.081172457456
                        + z
                        * (-718.6359919632359 + (146.4037555781616 - 4.9892131862671505 * z) * z)
                    )
                )
                + y
                * (
                    -592.743745734632
                    + z
                    * (
                        1591.873781627888
                        + z * (-1207.261522487504 + (608.785486935364 - 105.4993508931208 * z) * z)
                    )
                    + y
                    * (
                        290.12956292128547
                        + z
                        * (
                            -973.091553087975
                            + z
                            * (602.603274510125 + z * (-276.361526170076 + 32.40953340386105 * z))
                        )
                        + y
                        * (
                            -113.90630790850321
                            + y * (21.35571525415769 - 67.41756835751434 * z)
                            + z
                            * (
                                381.06836198507096
                                + z * (-133.7383902842754 + 49.023632509086724 * z)
                            )
                        )
                    )
                )
            )
        )
    )

    # g08_t polynomial (from GSW-C source)
    g08_t = x2 * (
        168.072408311545
        + x
        * (
            -493.407510141682
            + x
            * (
                543.835333000098
                + x * (-196.028306689776 + 36.7571622995805 * x)
                + y
                * (
                    -137.1145018408982
                    + y * (148.10030845687618 + y * (-68.5590309679152 + 12.4848504784754 * y))
                )
                - 22.6683558512829 * z
            )
            + z * (-175.292041186547 + (83.1923927801819 - 29.483064349429 * z) * z)
            + y
            * (
                -86.1329351956084
                + z * (766.116132004952 + z * (-108.3834525034224 + 51.2796974779828 * z))
                + y
                * (
                    -30.0682112585625
                    - 1380.9597954037708 * z
                    + y * (3.50240264723578 + 938.26075044542 * z)
                )
            )
        )
    )

    # g08_sa_t polynomial (from GSW-C source)
    g08_sa_t = 1187.3715515697959 + x * (
        -1480.222530425046
        + x
        * (
            2175.341332000392
            + x * (-980.14153344888 + 220.542973797483 * x)
            + y
            * (
                -548.4580073635929
                + y * (592.4012338275047 + y * (-274.2361238716608 + 49.9394019139016 * y))
            )
            - 90.6734234051316 * z
        )
        + z * (-525.876123559641 + (249.57717834054571 - 88.449193048287 * z) * z)
        + y
        * (
            -258.3988055868252
            + z * (2298.348396014856 + z * (-325.1503575102672 + 153.8390924339484 * z))
            + y
            * (
                -90.2046337756875
                - 4142.8793862113125 * z
                + y * (10.50720794170734 + 2814.78225133626 * z)
            )
        )
    )

    # Temperature derivative = kg2g * (g03_t + g08_t - 0.5*x2*g08_sa_t)
    # Note: The derivative with respect to t (not y) requires multiplying by 0.025
    # since y = t*0.025, so d/dy = (1/0.025) * d/dt
    t_deriv = kg2g * (g03_t + g08_t - 0.5 * x2 * g08_sa_t) * 0.025

    return t_deriv
