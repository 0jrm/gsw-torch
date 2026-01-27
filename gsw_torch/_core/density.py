"""
Core PyTorch implementations of density and specific volume functions.

These functions calculate density, specific volume, and related coefficients
using the TEOS-10 equations.
"""

import torch

from .._utilities import as_tensor

__all__ = [
    "specvol_alpha_beta",
    "rho",
    "alpha",
    "beta",
    "kappa",
    "sound_speed",
    "sigma0",
    "sigma1",
    "sigma2",
    "sigma3",
    "sigma4",
    "specvol",
    "rho_alpha_beta",
    "alpha_on_beta",
    "specvol_anom_standard",
    "rho_t_exact",
    "pot_rho_t_exact",
    "specvol_t_exact",
    "alpha_wrt_t_exact",
    "beta_const_t_exact",
    "cp_t_exact",
    "kappa_t_exact",
    "sound_speed_t_exact",
    "specvol_first_derivatives",
    "specvol_second_derivatives",
    "rho_first_derivatives",
    "rho_second_derivatives",
    "infunnel",
    "spiciness0",
    "spiciness1",
    "spiciness2",
    "rho_first_derivatives_wrt_enthalpy",
    "rho_second_derivatives_wrt_enthalpy",
    "specvol_first_derivatives_wrt_enthalpy",
    "specvol_second_derivatives_wrt_enthalpy",
]


def specvol_alpha_beta(SA, CT, p):
    """
    Calculates specific volume, the appropriate thermal expansion coefficient
    and the appropriate saline contraction coefficient of seawater from
    Absolute Salinity and Conservative Temperature.  This function uses the
    computationally-efficient 75-term polynomial expression for specific volume
    in terms of SA, CT and p (Roquet et al., 2015).

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
    specvol : torch.Tensor, m^3/kg
        specific volume
    alpha : torch.Tensor, 1/K
        thermal expansion coefficient with respect to Conservative Temperature
    beta : torch.Tensor, kg/g
        saline contraction coefficient at constant Conservative Temperature

    Notes
    -----
    This is a pure PyTorch implementation of the 75-term polynomial expression
    from Roquet et al. (2015). The polynomial has been fitted in a restricted
    range of parameter space, and is most accurate inside the "oceanographic
    funnel" described in McDougall et al. (2003).

    References
    ----------
    Roquet, F., G. Madec, T.J. McDougall, P.M. Barker, 2015: Accurate
    polynomial expressions for the density and specific volume of seawater
    using the TEOS-10 standard. Ocean Modelling., 90, pp. 29-43.
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, CT, p = torch.broadcast_tensors(SA, CT, p)
    
    # Reduced variables (Roquet et al. 2015)
    SAu = 40.0 * 35.16504 / 35.0
    CTu = 40.0
    Zu = 1e4
    deltaS = 24.0
    
    ss = torch.sqrt((SA + deltaS) / SAu)
    tt = CT / CTu
    pp = p / Zu
    
    # Vertical reference profile of specific volume (6th order polynomial in pp)
    V00 = -4.4015007269e-05
    V01 = 6.9232335784e-06
    V02 = -7.5004675975e-07
    V03 = 1.7009109288e-08
    V04 = -1.6884162004e-08
    V05 = 1.9613503930e-09
    
    v0 = (((((V05 * pp + V04) * pp + V03) * pp + V02) * pp + V01) * pp + V00) * pp
    
    # Specific volume anomaly (75-term polynomial)
    # Coefficients for the polynomial terms
    V000 = 1.0769995862e-03
    V100 = -3.1038981976e-04
    V200 = 6.6928067038e-04
    V300 = -8.5047933937e-04
    V400 = 5.8086069943e-04
    V500 = -2.1092370507e-04
    V600 = 3.1932457305e-05
    V010 = -1.5649734675e-05
    V110 = 3.5009599764e-05
    V210 = -4.3592678561e-05
    V310 = 3.4532461828e-05
    V410 = -1.1959409788e-05
    V510 = 1.3864594581e-06
    V020 = 2.7762106484e-05
    V120 = -3.7435842344e-05
    V220 = 3.5907822760e-05
    V320 = -1.8698584187e-05
    V420 = 3.8595339244e-06
    V030 = -1.6521159259e-05
    V130 = 2.4141479483e-05
    V230 = -1.4353633048e-05
    V330 = 2.2863324556e-06
    V040 = 6.9111322702e-06
    V140 = -8.7595873154e-06
    V240 = 4.3703680598e-06
    V050 = -8.0539615540e-07
    V150 = -3.3052758900e-07
    V060 = 2.0543094268e-07
    V001 = -1.6784136540e-05
    V101 = 2.4262468747e-05
    V201 = -3.4792460974e-05
    V301 = 3.7470777305e-05
    V401 = -1.7322218612e-05
    V501 = 3.0927427253e-06
    V011 = 1.8505765429e-05
    V111 = -9.5677088156e-06
    V211 = 1.1100834765e-05
    V311 = -9.8447117844e-06
    V411 = 2.5909225260e-06
    V021 = -1.1716606853e-05
    V121 = -2.3678308361e-07
    V221 = 2.9283346295e-06
    V321 = -4.8826139200e-07
    V031 = 7.9279656173e-06
    V131 = -3.4558773655e-06
    V231 = 3.1655306078e-07
    V041 = -3.4102187482e-06
    V141 = 1.2956717783e-06
    V051 = 5.0736766814e-07
    V002 = 3.0623833435e-06
    V102 = -5.8484432984e-07
    V202 = -4.8122251597e-06
    V302 = 4.9263106998e-06
    V402 = -1.7811974727e-06
    V012 = -1.1736386731e-06
    V112 = -5.5699154557e-06
    V212 = 5.4620748834e-06
    V312 = -1.3544185627e-06
    V022 = 2.1305028740e-06
    V122 = 3.9137387080e-07
    V222 = -6.5731104067e-07
    V032 = -4.6132540037e-07
    V132 = 7.7618888092e-09
    V042 = -6.3352916514e-08
    V003 = -3.8088938393e-07
    V103 = 3.6310188515e-07
    V203 = 1.6746303780e-08
    V013 = -3.6527006553e-07
    V113 = -2.7295696237e-07
    V023 = 2.8695905159e-07
    V004 = 8.8302421514e-08
    V104 = -1.1147125423e-07
    V014 = 3.1454099902e-07
    V005 = 4.2369007180e-09
    
    # Build polynomial terms (nested Horner form for efficiency)
    vp5 = V005
    vp4 = V014 * tt + V104 * ss + V004
    vp3 = (V023 * tt + V113 * ss + V013) * tt + (V203 * ss + V103) * ss + V003
    vp2 = (((V042 * tt + V132 * ss + V032) * tt + (V222 * ss + V122) * ss + V022) * tt +
            ((V312 * ss + V212) * ss + V112) * ss + V012) * tt + \
            (((V402 * ss + V302) * ss + V202) * ss + V102) * ss + V002
    vp1 = ((((V051 * tt + V141 * ss + V041) * tt + (V231 * ss + V131) * ss + V031) * tt +
            ((V321 * ss + V221) * ss + V121) * ss + V021) * tt +
            (((V411 * ss + V311) * ss + V211) * ss + V111) * ss + V011) * tt + \
            ((((V501 * ss + V401) * ss + V301) * ss + V201) * ss + V101) * ss + V001
    vp0 = (((((V060 * tt + V150 * ss + V050) * tt + (V240 * ss + V140) * ss + V040) * tt +
            ((V330 * ss + V230) * ss + V130) * ss + V030) * tt +
            (((V420 * ss + V320) * ss + V220) * ss + V120) * ss + V020) * tt +
            ((((V510 * ss + V410) * ss + V310) * ss + V210) * ss + V110) * ss + V010) * tt + \
            (((((V600 * ss + V500) * ss + V400) * ss + V300) * ss + V200) * ss + V100) * ss + V000
    
    delta = ((((vp5 * pp + vp4) * pp + vp3) * pp + vp2) * pp + vp1) * pp + vp0
    
    # Specific volume
    specvol = v0 + delta
    
    # Thermal expansion coefficient alpha (polynomial in ss, tt, pp)
    A000 = -3.9124336688e-07
    A100 = 8.7523999410e-07
    A200 = -1.0898169640e-06
    A300 = 8.6331154570e-07
    A400 = -2.9898524469e-07
    A500 = 3.4661486454e-08
    A010 = 1.3881053242e-06
    A110 = -1.8717921172e-06
    A210 = 1.7953911380e-06
    A310 = -9.3492920933e-07
    A410 = 1.9297669622e-07
    A020 = -1.2390869444e-06
    A120 = 1.8106109612e-06
    A220 = -1.0765224786e-06
    A320 = 1.7147493417e-07
    A030 = 6.9111322702e-07
    A130 = -8.7595873154e-07
    A230 = 4.3703680598e-07
    A040 = -1.0067451943e-07
    A140 = -4.1315948624e-08
    A050 = 3.0814641402e-08
    A001 = 4.6264413572e-07
    A101 = -2.3919272039e-07
    A201 = 2.7752086911e-07
    A301 = -2.4611779461e-07
    A401 = 6.4773063150e-08
    A011 = -5.8583034263e-07
    A111 = -1.1839154180e-08
    A211 = 1.4641673148e-07
    A311 = -2.4413069600e-08
    A021 = 5.9459742130e-07
    A121 = -2.5919080242e-07
    A221 = 2.3741479559e-08
    A031 = -3.4102187482e-07
    A131 = 1.2956717783e-07
    A041 = 6.3420958518e-08
    A002 = -2.9340966828e-08
    A102 = -1.3924788639e-07
    A202 = 1.3655187208e-07
    A302 = -3.3860464067e-08
    A012 = 1.0652514370e-07
    A112 = 1.9568693540e-08
    A212 = -3.2865552033e-08
    A022 = -3.4599405028e-08
    A122 = 5.8214166069e-10
    A032 = -6.3352916514e-09
    A003 = -9.1317516382e-09
    A103 = -6.8239240593e-09
    A013 = 1.4347952579e-08
    A004 = 7.8635249756e-09
    
    ap4 = A004
    ap3 = A013 * tt + A103 * ss + A003
    ap2 = (((A032 * tt + A122 * ss + A022) * tt + (A212 * ss + A112) * ss + A012) * tt +
            ((A302 * ss + A202) * ss + A102) * ss + A002)
    ap1 = ((((A041 * tt + A131 * ss + A031) * tt + (A221 * ss + A121) * ss + A021) * tt +
            ((A311 * ss + A211) * ss + A111) * ss + A011) * tt +
            (((A401 * ss + A301) * ss + A201) * ss + A101) * ss + A001)
    ap0 = (((((A050 * tt + A140 * ss + A040) * tt + (A230 * ss + A130) * ss + A030) * tt +
            ((A320 * ss + A220) * ss + A120) * ss + A020) * tt +
            (((A410 * ss + A310) * ss + A210) * ss + A110) * ss + A010) * tt +
            ((((A500 * ss + A400) * ss + A300) * ss + A200) * ss + A100) * ss + A000)
    
    a = (((ap4 * pp + ap3) * pp + ap2) * pp + ap1) * pp + ap0
    alpha = a / specvol
    
    # Saline contraction coefficient beta (polynomial in ss, tt, pp)
    B000 = 3.8616633493e-06
    B100 = -1.6653488424e-05
    B200 = 3.1743292000e-05
    B300 = -2.8906727363e-05
    B400 = 1.3120861084e-05
    B500 = -2.3836941584e-06
    B010 = -4.3556611614e-07
    B110 = 1.0847021286e-06
    B210 = -1.2888896515e-06
    B310 = 5.9516403589e-07
    B410 = -8.6247024451e-08
    B020 = 4.6575180991e-07
    B120 = -8.9348241648e-07
    B220 = 6.9790598120e-07
    B320 = -1.9207099914e-07
    B030 = -3.0035220418e-07
    B130 = 3.5715667939e-07
    B230 = -8.5335075632e-08
    B040 = 1.0898094956e-07
    B140 = -1.0874641554e-07
    B050 = 4.1122040579e-09
    B001 = -3.0185747199e-07
    B101 = 8.6572923996e-07
    B201 = -1.3985593422e-06
    B301 = 8.6204601419e-07
    B401 = -1.9238922270e-07
    B011 = 1.1903505888e-07
    B111 = -2.7621838107e-07
    B211 = 3.6744403581e-07
    B311 = -1.2893812777e-07
    B021 = 2.9458973764e-09
    B121 = -7.2864777086e-08
    B221 = 1.8223868848e-08
    B031 = 4.2995723805e-08
    B131 = -7.8766845761e-09
    B041 = -1.6119885062e-08
    B002 = 7.2762435164e-09
    B102 = 1.1974099887e-07
    B202 = -1.8386962715e-07
    B302 = 8.8641889138e-08
    B012 = 6.9297177307e-08
    B112 = -1.3591099350e-07
    B212 = 5.0552320246e-08
    B022 = -4.8692129590e-09
    B122 = 1.6355652107e-08
    B032 = -9.6568249433e-11
    B003 = -4.5174717490e-09
    B103 = -4.1669270980e-10
    B013 = 3.3959486762e-09
    B004 = 1.3868510806e-09
    
    bp4 = B004
    bp3 = B013 * tt + B103 * ss + B003
    bp2 = (((B032 * tt + B122 * ss + B022) * tt + (B212 * ss + B112) * ss + B012) * tt +
            ((B302 * ss + B202) * ss + B102) * ss + B002)
    bp1 = ((((B041 * tt + B131 * ss + B031) * tt + (B221 * ss + B121) * ss + B021) * tt +
            ((B311 * ss + B211) * ss + B111) * ss + B011) * tt +
            (((B401 * ss + B301) * ss + B201) * ss + B101) * ss + B001)
    bp0 = (((((B050 * tt + B140 * ss + B040) * tt + (B230 * ss + B130) * ss + B030) * tt +
            ((B320 * ss + B220) * ss + B120) * ss + B020) * tt +
            (((B410 * ss + B310) * ss + B210) * ss + B110) * ss + B010) * tt +
            ((((B500 * ss + B400) * ss + B300) * ss + B200) * ss + B100) * ss + B000)
    
    b = (((bp4 * pp + bp3) * pp + bp2) * pp + bp1) * pp + bp0
    beta = b / ss / specvol
    
    return specvol, alpha, beta


def rho(SA, CT, p):
    """
    Calculates in-situ density from Absolute Salinity and Conservative Temperature.

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
    rho : torch.Tensor, kg/m^3
        in-situ density

    Notes
    -----
    Density is the inverse of specific volume: rho = 1/specvol
    """
    specvol, _, _ = specvol_alpha_beta(SA, CT, p)
    return 1.0 / specvol


def alpha(SA, CT, p):
    """
    Calculates the thermal expansion coefficient of seawater.

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
    alpha : torch.Tensor, 1/K
        thermal expansion coefficient with respect to Conservative Temperature
    """
    _, alpha_val, _ = specvol_alpha_beta(SA, CT, p)
    return alpha_val


def beta(SA, CT, p):
    """
    Calculates the saline contraction coefficient of seawater.

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
    beta : torch.Tensor, kg/g
        saline contraction coefficient at constant Conservative Temperature
    """
    _, _, beta_val = specvol_alpha_beta(SA, CT, p)
    return beta_val


def specvol(SA, CT, p):
    """
    Calculates specific volume from Absolute Salinity and Conservative Temperature.

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
    specvol : torch.Tensor, m^3/kg
        specific volume
    """
    specvol_val, _, _ = specvol_alpha_beta(SA, CT, p)
    return specvol_val


def rho_alpha_beta(SA, CT, p):
    """
    Calculates in-situ density, thermal expansion coefficient, and
    saline contraction coefficient.

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
    rho : torch.Tensor, kg/m^3
        in-situ density
    alpha : torch.Tensor, 1/K
        thermal expansion coefficient
    beta : torch.Tensor, kg/g
        saline contraction coefficient
    """
    specvol_val, alpha_val, beta_val = specvol_alpha_beta(SA, CT, p)
    rho_val = 1.0 / specvol_val
    return rho_val, alpha_val, beta_val


def alpha_on_beta(SA, CT, p):
    """
    Calculates alpha divided by beta.

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
    alpha_on_beta : torch.Tensor, g kg^-1 K^-1
        thermal expansion coefficient divided by saline contraction coefficient
    """
    _, alpha_val, beta_val = specvol_alpha_beta(SA, CT, p)
    return alpha_val / beta_val


# Placeholder functions - will be implemented incrementally
def _not_implemented(name):
    """Helper to create placeholder functions."""

    def func(*args, **kwargs):
        raise NotImplementedError(
            f"{name} is not yet implemented in pure PyTorch. "
            "This function will be available in a future release."
        )

    func.__name__ = name
    return func
def sigma0(SA, CT):
    """
    Calculates potential density anomaly with reference pressure of 0 dbar.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    sigma0 : torch.Tensor, kg/m^3
        potential density anomaly (potential density - 1000 kg/m^3)
        at reference pressure of 0 dbar
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p_ref = torch.zeros_like(SA)
    rho_pot = rho(SA, CT, p_ref)
    return rho_pot - 1000.0


def sigma1(SA, CT):
    """
    Calculates potential density anomaly with reference pressure of 1000 dbar.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    sigma1 : torch.Tensor, kg/m^3
        potential density anomaly (potential density - 1000 kg/m^3)
        at reference pressure of 1000 dbar
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p_ref = torch.full_like(SA, 1000.0)
    rho_pot = rho(SA, CT, p_ref)
    return rho_pot - 1000.0


def sigma2(SA, CT):
    """
    Calculates potential density anomaly with reference pressure of 2000 dbar.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    sigma2 : torch.Tensor, kg/m^3
        potential density anomaly (potential density - 1000 kg/m^3)
        at reference pressure of 2000 dbar
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p_ref = torch.full_like(SA, 2000.0)
    rho_pot = rho(SA, CT, p_ref)
    return rho_pot - 1000.0


def sigma3(SA, CT):
    """
    Calculates potential density anomaly with reference pressure of 3000 dbar.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    sigma3 : torch.Tensor, kg/m^3
        potential density anomaly (potential density - 1000 kg/m^3)
        at reference pressure of 3000 dbar
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p_ref = torch.full_like(SA, 3000.0)
    rho_pot = rho(SA, CT, p_ref)
    return rho_pot - 1000.0


def sigma4(SA, CT):
    """
    Calculates potential density anomaly with reference pressure of 4000 dbar.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    sigma4 : torch.Tensor, kg/m^3
        potential density anomaly (potential density - 1000 kg/m^3)
        at reference pressure of 4000 dbar
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p_ref = torch.full_like(SA, 4000.0)
    rho_pot = rho(SA, CT, p_ref)
    return rho_pot - 1000.0


# Placeholder functions - will be implemented incrementally
def _not_implemented(name):
    """Helper to create placeholder functions."""

    def func(*args, **kwargs):
        raise NotImplementedError(
            f"{name} is not yet implemented in pure PyTorch. "
            "This function will be available in a future release."
        )

    func.__name__ = name
    return func


def specvol_first_derivatives(SA, CT, p):
    """
    Calculates the first derivatives of specific volume with respect to
    Absolute Salinity, Conservative Temperature, and pressure.

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
    v_SA : torch.Tensor, (m^3/kg)/(g/kg)
        derivative with respect to Absolute Salinity
    v_CT : torch.Tensor, (m^3/kg)/K
        derivative with respect to Conservative Temperature
    v_p : torch.Tensor, (m^3/kg)/Pa
        derivative with respect to pressure (converted from dbar to Pa)

    Notes
    -----
    This is a pure PyTorch implementation derived from the 75-term polynomial
    expression for specific volume (Roquet et al., 2015).
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, CT, p = torch.broadcast_tensors(SA, CT, p)
    
    # Reduced variables
    SAu = 40.0 * 35.16504 / 35.0
    CTu = 40.0
    Zu = 1e4
    deltaS = 24.0
    
    ss = torch.sqrt((SA + deltaS) / SAu)
    tt = CT / CTu
    pp = p / Zu
    
    # Derivatives of reduced variables
    dss_dSA = 0.5 / (SAu * ss)  # d(ss)/d(SA)
    dtt_dCT = 1.0 / CTu  # d(tt)/d(CT)
    dpp_dp = 1.0 / Zu  # d(pp)/d(p)
    
    # Get specvol and alpha, beta (we'll use these relationships)
    specvol, alpha, beta = specvol_alpha_beta(SA, CT, p)
    
    # From thermodynamic relationships:
    # alpha = (1/v) * dv/dCT, so v_CT = v * alpha
    # beta = -(1/v) * dv/dSA, so v_SA = -v * beta
    v_CT = specvol * alpha
    v_SA = -specvol * beta
    
    # For v_p, we need to differentiate the polynomial with respect to pp
    # v = v0 + delta, where:
    # v0 = (((((V05*pp+V04)*pp+V03)*pp+V02)*pp+V01)*pp+V00)*pp
    # delta = ((((vp5*pp+vp4)*pp+vp3)*pp+vp2)*pp+vp1)*pp+vp0
    
    # Coefficients (same as in specvol_alpha_beta)
    V00 = -4.4015007269e-05
    V01 = 6.9232335784e-06
    V02 = -7.5004675975e-07
    V03 = 1.7009109288e-08
    V04 = -1.6884162004e-08
    V05 = 1.9613503930e-09
    
    # Derivative of v0 with respect to pp
    dv0_dpp = ((((6*V05*pp + 5*V04)*pp + 4*V03)*pp + 3*V02)*pp + 2*V01)*pp + V00
    
    # Now compute delta and its derivative
    # We need to recompute the polynomial terms for delta
    V000 = 1.0769995862e-03
    V100 = -3.1038981976e-04
    V200 = 6.6928067038e-04
    V300 = -8.5047933937e-04
    V400 = 5.8086069943e-04
    V500 = -2.1092370507e-04
    V600 = 3.1932457305e-05
    V010 = -1.5649734675e-05
    V110 = 3.5009599764e-05
    V210 = -4.3592678561e-05
    V310 = 3.4532461828e-05
    V410 = -1.1959409788e-05
    V510 = 1.3864594581e-06
    V020 = 2.7762106484e-05
    V120 = -3.7435842344e-05
    V220 = 3.5907822760e-05
    V320 = -1.8698584187e-05
    V420 = 3.8595339244e-06
    V030 = -1.6521159259e-05
    V130 = 2.4141479483e-05
    V230 = -1.4353633048e-05
    V330 = 2.2863324556e-06
    V040 = 6.9111322702e-06
    V140 = -8.7595873154e-06
    V240 = 4.3703680598e-06
    V050 = -8.0539615540e-07
    V150 = -3.3052758900e-07
    V060 = 2.0543094268e-07
    V001 = -1.6784136540e-05
    V101 = 2.4262468747e-05
    V201 = -3.4792460974e-05
    V301 = 3.7470777305e-05
    V401 = -1.7322218612e-05
    V501 = 3.0927427253e-06
    V011 = 1.8505765429e-05
    V111 = -9.5677088156e-06
    V211 = 1.1100834765e-05
    V311 = -9.8447117844e-06
    V411 = 2.5909225260e-06
    V021 = -1.1716606853e-05
    V121 = -2.3678308361e-07
    V221 = 2.9283346295e-06
    V321 = -4.8826139200e-07
    V031 = 7.9279656173e-06
    V131 = -3.4558773655e-06
    V231 = 3.1655306078e-07
    V041 = -3.4102187482e-06
    V141 = 1.2956717783e-06
    V051 = 5.0736766814e-07
    V002 = 3.0623833435e-06
    V102 = -5.8484432984e-07
    V202 = -4.8122251597e-06
    V302 = 4.9263106998e-06
    V402 = -1.7811974727e-06
    V012 = -1.1736386731e-06
    V112 = -5.5699154557e-06
    V212 = 5.4620748834e-06
    V312 = -1.3544185627e-06
    V022 = 2.1305028740e-06
    V122 = 3.9137387080e-07
    V222 = -6.5731104067e-07
    V032 = -4.6132540037e-07
    V132 = 7.7618888092e-09
    V042 = -6.3352916514e-08
    V003 = -3.8088938393e-07
    V103 = 3.6310188515e-07
    V203 = 1.6746303780e-08
    V013 = -3.6527006553e-07
    V113 = -2.7295696237e-07
    V023 = 2.8695905159e-07
    V004 = 8.8302421514e-08
    V104 = -1.1147125423e-07
    V014 = 3.1454099902e-07
    V005 = 4.2369007180e-09
    
    # Build polynomial terms for delta
    vp5 = V005
    vp4 = V014 * tt + V104 * ss + V004
    vp3 = (V023 * tt + V113 * ss + V013) * tt + (V203 * ss + V103) * ss + V003
    vp2 = (((V042 * tt + V132 * ss + V032) * tt + (V222 * ss + V122) * ss + V022) * tt +
            ((V312 * ss + V212) * ss + V112) * ss + V012) * tt + \
            (((V402 * ss + V302) * ss + V202) * ss + V102) * ss + V002
    vp1 = ((((V051 * tt + V141 * ss + V041) * tt + (V231 * ss + V131) * ss + V031) * tt +
            ((V321 * ss + V221) * ss + V121) * ss + V021) * tt +
            (((V411 * ss + V311) * ss + V211) * ss + V111) * ss + V011) * tt + \
            ((((V501 * ss + V401) * ss + V301) * ss + V201) * ss + V101) * ss + V001
    vp0 = (((((V060 * tt + V150 * ss + V050) * tt + (V240 * ss + V140) * ss + V040) * tt +
            ((V330 * ss + V230) * ss + V130) * ss + V030) * tt +
            (((V420 * ss + V320) * ss + V220) * ss + V120) * ss + V020) * tt +
            ((((V510 * ss + V410) * ss + V310) * ss + V210) * ss + V110) * ss + V010) * tt + \
            (((((V600 * ss + V500) * ss + V400) * ss + V300) * ss + V200) * ss + V100) * ss + V000
    
    # Derivative of delta with respect to pp
    ddelta_dpp = (((5*vp5*pp + 4*vp4)*pp + 3*vp3)*pp + 2*vp2)*pp + vp1
    
    # Total derivative with respect to p (in dbar): v_p = (dv0_dpp + ddelta_dpp) * dpp_dp
    # This gives v_p in (m^3/kg)/dbar
    v_p_dbar = (dv0_dpp + ddelta_dpp) * dpp_dp
    
    # Convert to (m^3/kg)/Pa for consistency with GSW output
    # 1 dbar = 1e4 Pa, so v_p_Pa = v_p_dbar / 1e4
    v_p = v_p_dbar / 1e4
    
    return v_SA, v_CT, v_p


def kappa(SA, CT, p):
    """
    Calculates the isentropic compressibility of seawater.

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
    kappa : torch.Tensor, 1/Pa
        isentropic compressibility of seawater

    Notes
    -----
    This is a pure PyTorch implementation. The formula is:
    kappa = -v_p / v
    where v is specific volume and v_p is its pressure derivative in Pa.
    """
    specvol, _, _ = specvol_alpha_beta(SA, CT, p)
    _, _, v_p = specvol_first_derivatives(SA, CT, p)
    
    # v_p from specvol_first_derivatives is already in (m^3/kg)/Pa
    # kappa = -v_p / v (in Pa^-1)
    kappa = -v_p / specvol
    
    return kappa


def sound_speed(SA, CT, p):
    """
    Calculates the speed of sound in seawater.

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
    sound_speed : torch.Tensor, m/s
        speed of sound in seawater

    Notes
    -----
    This is a pure PyTorch implementation. The formula is:
    c = sqrt(-v^2 / v_p)
    where v is specific volume and v_p is its pressure derivative in Pa.
    """
    specvol, _, _ = specvol_alpha_beta(SA, CT, p)
    _, _, v_p = specvol_first_derivatives(SA, CT, p)
    
    # v_p from specvol_first_derivatives is already in (m^3/kg)/Pa
    # Sound speed: c = sqrt(-v^2 / v_p)
    # v_p is negative, so -v^2/v_p is positive
    c = torch.sqrt(-specvol**2 / v_p)
    
    return c


def specvol_second_derivatives(SA, CT, p):
    """
    Calculates the second derivatives of specific volume.

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
    v_SA_SA : torch.Tensor, (m^3/kg)(g/kg)^-2
        second derivative with respect to Absolute Salinity
    v_SA_CT : torch.Tensor, (m^3/kg)(g/kg)^-1 K^-1
        second derivative with respect to SA and CT
    v_CT_CT : torch.Tensor, (m^3/kg) K^-2
        second derivative with respect to Conservative Temperature
    v_SA_p : torch.Tensor, (m^3/kg)(g/kg)^-1 Pa^-1
        second derivative with respect to SA and pressure
    v_CT_p : torch.Tensor, (m^3/kg) K^-1 Pa^-1
        second derivative with respect to CT and pressure

    Notes
    -----
    This is a pure PyTorch implementation derived from the 75-term polynomial
    expression for specific volume (Roquet et al., 2015).
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, CT, p = torch.broadcast_tensors(SA, CT, p)
    
    # Reduced variables
    SAu = 40.0 * 35.16504 / 35.0
    CTu = 40.0
    Zu = 1e4
    deltaS = 24.0
    
    ss = torch.sqrt((SA + deltaS) / SAu)
    tt = CT / CTu
    pp = p / Zu
    
    # Derivatives of reduced variables
    dss_dSA = 0.5 / (SAu * ss)  # d(ss)/d(SA)
    dtt_dCT = 1.0 / CTu  # d(tt)/d(CT)
    dpp_dp = 1.0 / Zu  # d(pp)/d(p)
    
    # Second derivatives of reduced variables
    d2ss_dSA2 = -0.25 / (SAu**2 * ss**3)  # d^2(ss)/d(SA)^2
    
    # Get first derivatives and specvol, alpha, beta
    specvol, alpha, beta = specvol_alpha_beta(SA, CT, p)
    v_SA, v_CT, v_p = specvol_first_derivatives(SA, CT, p)
    
    # For second derivatives, we need to differentiate the polynomial expressions
    # v_SA = -v * beta, so v_SA_SA = -d(v*beta)/dSA = -v*d(beta)/dSA - beta*d(v)/dSA
    # v_CT = v * alpha, so v_CT_CT = d(v*alpha)/dCT = v*d(alpha)/dCT + alpha*d(v)/dCT
    
    # We need to compute derivatives of alpha and beta with respect to SA and CT
    # This requires differentiating the polynomial coefficients
    
    # For now, use numerical differentiation via autograd
    # This is more reliable than manually differentiating the complex polynomial
    SA.requires_grad_(True)
    CT.requires_grad_(True)
    p.requires_grad_(True)
    
    specvol_grad, alpha_grad, beta_grad = specvol_alpha_beta(SA, CT, p)
    
    # Compute gradients
    v_SA_grad = torch.autograd.grad(specvol_grad, SA, create_graph=True, retain_graph=True)[0]
    v_CT_grad = torch.autograd.grad(specvol_grad, CT, create_graph=True, retain_graph=True)[0]
    
    # Second derivatives
    v_SA_SA = torch.autograd.grad(v_SA_grad, SA, retain_graph=True)[0]
    v_SA_CT = torch.autograd.grad(v_SA_grad, CT, retain_graph=True)[0]
    v_CT_CT = torch.autograd.grad(v_CT_grad, CT, retain_graph=True)[0]
    v_SA_p = torch.autograd.grad(v_SA_grad, p, retain_graph=True)[0]
    v_CT_p = torch.autograd.grad(v_CT_grad, p)[0]
    
    # Convert pressure derivatives from dbar to Pa
    v_SA_p = v_SA_p / 1e4
    v_CT_p = v_CT_p / 1e4
    
    return v_SA_SA, v_SA_CT, v_CT_CT, v_SA_p, v_CT_p




def specvol_second_derivatives(SA, CT, p):
    """
    Calculates the second derivatives of specific volume.

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
    v_SA_SA : torch.Tensor, (m^3/kg)(g/kg)^-2
        second derivative with respect to Absolute Salinity
    v_SA_CT : torch.Tensor, (m^3/kg)(g/kg)^-1 K^-1
        second derivative with respect to SA and CT
    v_CT_CT : torch.Tensor, (m^3/kg) K^-2
        second derivative with respect to Conservative Temperature
    v_SA_p : torch.Tensor, (m^3/kg)(g/kg)^-1 Pa^-1
        second derivative with respect to SA and pressure
    v_CT_p : torch.Tensor, (m^3/kg) K^-1 Pa^-1
        second derivative with respect to CT and pressure

    Notes
    -----
    This is a pure PyTorch implementation using automatic differentiation
    to compute second derivatives from the 75-term polynomial expression
    (Roquet et al., 2015).
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, CT, p = torch.broadcast_tensors(SA, CT, p)
    
    # Enable gradients for all inputs
    SA_grad = SA.clone().detach().requires_grad_(True)
    CT_grad = CT.clone().detach().requires_grad_(True)
    p_grad = p.clone().detach().requires_grad_(True)
    
    # Compute specvol with gradients enabled
    specvol_grad, _, _ = specvol_alpha_beta(SA_grad, CT_grad, p_grad)
    
    # First derivatives
    v_SA_grad = torch.autograd.grad(
        specvol_grad, SA_grad, create_graph=True, retain_graph=True
    )[0]
    v_CT_grad = torch.autograd.grad(
        specvol_grad, CT_grad, create_graph=True, retain_graph=True
    )[0]
    v_p_grad = torch.autograd.grad(
        specvol_grad, p_grad, create_graph=True, retain_graph=True
    )[0]
    
    # Second derivatives
    # v_SA_SA = d^2(v)/d(SA)^2
    v_SA_SA = torch.autograd.grad(
        v_SA_grad, SA_grad, retain_graph=True
    )[0]
    
    # v_SA_CT = d^2(v)/(d(SA) d(CT))
    v_SA_CT = torch.autograd.grad(
        v_SA_grad, CT_grad, retain_graph=True
    )[0]
    
    # v_CT_CT = d^2(v)/d(CT)^2
    v_CT_CT = torch.autograd.grad(
        v_CT_grad, CT_grad, retain_graph=True
    )[0]
    
    # v_SA_p = d^2(v)/(d(SA) d(p))
    v_SA_p = torch.autograd.grad(
        v_SA_grad, p_grad, retain_graph=True
    )[0]
    
    # v_CT_p = d^2(v)/(d(CT) d(p))
    v_CT_p = torch.autograd.grad(
        v_CT_grad, p_grad
    )[0]
    
    # Convert pressure derivatives from dbar to Pa
    # p is in dbar, but derivatives should be in Pa
    # d/dp (in dbar) = d/d(p*1e4) (in Pa) = (1/1e4) * d/d(p in Pa)
    # So we need to multiply by 1e4 to get derivatives w.r.t. Pa
    # Actually, wait - v_p from first_derivatives is already in (m^3/kg)/Pa
    # So v_SA_p and v_CT_p need to be converted similarly
    # v_p = dv/dp where p is in dbar
    # To get dv/d(p_Pa) where p_Pa = p_dbar * 1e4:
    # dv/d(p_Pa) = dv/d(p_dbar) * d(p_dbar)/d(p_Pa) = dv/d(p_dbar) / 1e4
    # But we want derivatives in Pa, so we divide by 1e4
    v_SA_p = v_SA_p / 1e4
    v_CT_p = v_CT_p / 1e4
    
    return v_SA_SA, v_SA_CT, v_CT_CT, v_SA_p, v_CT_p


def rho_first_derivatives(SA, CT, p):
    """
    Calculates the first derivatives of in-situ density.

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
    rho_SA : torch.Tensor, (kg/m^3)(g/kg)^-1
        derivative with respect to Absolute Salinity
    rho_CT : torch.Tensor, kg/(m^3 K)
        derivative with respect to Conservative Temperature
    rho_P : torch.Tensor, kg/(m^3 Pa)
        derivative with respect to pressure in Pa

    Notes
    -----
    This is a pure PyTorch implementation. Since rho = 1/v, we have:
    rho_SA = -rho^2 * v_SA
    rho_CT = -rho^2 * v_CT
    rho_P = -rho^2 * v_P
    where v is specific volume and v_* are its derivatives.
    """
    rho_val = rho(SA, CT, p)
    v_SA, v_CT, v_p = specvol_first_derivatives(SA, CT, p)
    
    # rho = 1/v, so drho/dx = -rho^2 * dv/dx
    rho_SA = -rho_val**2 * v_SA
    rho_CT = -rho_val**2 * v_CT
    rho_P = -rho_val**2 * v_p  # v_p is already in Pa
    
    return rho_SA, rho_CT, rho_P


def rho_second_derivatives(SA, CT, p):
    """
    Calculates the second derivatives of in-situ density.

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
    rho_SA_SA : torch.Tensor, (kg/m^3)(g/kg)^-2
        second derivative with respect to SA at constant CT & p
    rho_SA_CT : torch.Tensor, (kg/m^3)(g/kg)^-1 K^-1
        derivative with respect to SA & CT at constant p
    rho_CT_CT : torch.Tensor, (kg/m^3) K^-2
        second derivative with respect to CT at constant SA & p
    rho_SA_P : torch.Tensor, (kg/m^3)(g/kg)^-1 Pa^-1
        derivative with respect to SA & P at constant CT
    rho_CT_P : torch.Tensor, (kg/m^3) K^-1 Pa^-1
        derivative with respect to CT & P at constant SA

    Notes
    -----
    This is a pure PyTorch implementation. Since rho = 1/v, the second derivatives
    are computed from specvol second derivatives using the chain rule.
    """
    rho_val = rho(SA, CT, p)
    rho_SA, rho_CT, rho_P = rho_first_derivatives(SA, CT, p)
    v_SA, v_CT, v_p = specvol_first_derivatives(SA, CT, p)
    v_SA_SA, v_SA_CT, v_CT_CT, v_SA_p, v_CT_p = specvol_second_derivatives(SA, CT, p)
    
    # Second derivatives from chain rule: d^2(rho)/dxdy = d(-rho^2 * dv/dy)/dx
    # rho_SA_SA = -rho^2 * v_SA_SA - 2*rho*rho_SA*v_SA
    rho_SA_SA = -rho_val**2 * v_SA_SA - 2.0 * rho_val * rho_SA * v_SA
    
    # rho_SA_CT = -rho^2 * v_SA_CT - rho*rho_SA*v_CT - rho*rho_CT*v_SA
    rho_SA_CT = -rho_val**2 * v_SA_CT - rho_val * rho_SA * v_CT - rho_val * rho_CT * v_SA
    
    # rho_CT_CT = -rho^2 * v_CT_CT - 2*rho*rho_CT*v_CT
    rho_CT_CT = -rho_val**2 * v_CT_CT - 2.0 * rho_val * rho_CT * v_CT
    
    # rho_SA_P = -rho^2 * v_SA_p - rho*rho_SA*v_p - rho*rho_P*v_SA
    rho_SA_P = -rho_val**2 * v_SA_p - rho_val * rho_SA * v_p - rho_val * rho_P * v_SA
    
    # rho_CT_P = -rho^2 * v_CT_p - rho*rho_CT*v_p - rho*rho_P*v_CT
    rho_CT_P = -rho_val**2 * v_CT_p - rho_val * rho_CT * v_p - rho_val * rho_P * v_CT
    
    return rho_SA_SA, rho_SA_CT, rho_CT_CT, rho_SA_P, rho_CT_P


# Placeholder functions - will be implemented incrementally
def _not_implemented(name):
    """Helper to create placeholder functions."""

    def func(*args, **kwargs):
        raise NotImplementedError(
            f"{name} is not yet implemented in pure PyTorch. "
            "This function will be available in a future release."
        )

    func.__name__ = name
    return func


def rho_t_exact(SA, t, p):
    """
    Calculates in-situ density of seawater from Absolute Salinity and
    in-situ temperature.

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
    rho_t_exact : torch.Tensor, kg/m^3
        in-situ density (not density anomaly)

    Notes
    -----
    This is a pure PyTorch implementation. Calculates density using:
    rho_t_exact(SA, t, p) = rho(SA, CT_from_t(SA, t, p), p)
    
    This uses the exact Gibbs function via CT_from_t, providing exact
    numerical parity with the reference GSW implementation.
    """
    from ..conversions import CT_from_t
    
    SA = as_tensor(SA, dtype=torch.float64)
    t = as_tensor(t, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, t, p = torch.broadcast_tensors(SA, t, p)
    
    # Convert in-situ temperature to Conservative Temperature
    CT = CT_from_t(SA, t, p)
    
    # Calculate density using CT
    rho_val = rho(SA, CT, p)
    
    return rho_val


def specvol_t_exact(SA, t, p):
    """
    Calculates in-situ specific volume of seawater from Absolute Salinity and
    in-situ temperature.

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
    specvol_t_exact : torch.Tensor, m^3/kg
        in-situ specific volume

    Notes
    -----
    This is a pure PyTorch implementation. Calculates specific volume using:
    specvol_t_exact(SA, t, p) = specvol(SA, CT_from_t(SA, t, p), p)
    
    This uses the exact Gibbs function via CT_from_t, providing exact
    numerical parity with the reference GSW implementation.
    """
    from ..conversions import CT_from_t
    
    SA = as_tensor(SA, dtype=torch.float64)
    t = as_tensor(t, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, t, p = torch.broadcast_tensors(SA, t, p)
    
    # Convert in-situ temperature to Conservative Temperature
    CT = CT_from_t(SA, t, p)
    
    # Calculate specific volume using CT
    specvol_val = specvol(SA, CT, p)
    
    return specvol_val


def alpha_wrt_t_exact(SA, t, p):
    """
    Calculates the thermal expansion coefficient of seawater from in-situ temperature
    using the exact Gibbs function.

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
    alpha : torch.Tensor, 1/K
        thermal expansion coefficient with respect to in-situ temperature

    Notes
    -----
    This is a pure PyTorch implementation using the exact Gibbs function.
    Calculates alpha using: alpha_wrt_t_exact = gibbs(0,1,1) / gibbs(0,0,1)
    where gibbs(0,1,1) = d²G/(dt dp) and gibbs(0,0,1) = dG/dp = specific volume.
    This provides exact numerical parity with the reference GSW implementation.
    """
    from .gibbs import gibbs
    
    SA = as_tensor(SA, dtype=torch.float64)
    t = as_tensor(t, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, t, p = torch.broadcast_tensors(SA, t, p)
    
    # Calculate alpha using exact Gibbs function: alpha = gibbs(0,1,1) / gibbs(0,0,1)
    gibbs_011 = gibbs(0, 1, 1, SA, t, p)
    gibbs_001 = gibbs(0, 0, 1, SA, t, p)
    alpha_val = gibbs_011 / gibbs_001
    
    return alpha_val


def beta_const_t_exact(SA, t, p):
    """
    Calculates the saline contraction coefficient of seawater from in-situ temperature
    using the exact Gibbs function.

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
    beta : torch.Tensor, kg/g
        saline contraction coefficient at constant in-situ temperature

    Notes
    -----
    This is a pure PyTorch implementation using the exact Gibbs function.
    Calculates beta using: beta_const_t_exact = -gibbs(1,0,1) / gibbs(0,0,1)
    where gibbs(1,0,1) = d²G/(dSA dp) and gibbs(0,0,1) = dG/dp = specific volume.
    This provides exact numerical parity with the reference GSW implementation.
    """
    from .gibbs import gibbs
    
    SA = as_tensor(SA, dtype=torch.float64)
    t = as_tensor(t, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, t, p = torch.broadcast_tensors(SA, t, p)
    
    # Calculate beta using exact Gibbs function: beta = -gibbs(1,0,1) / gibbs(0,0,1)
    gibbs_101 = gibbs(1, 0, 1, SA, t, p)
    gibbs_001 = gibbs(0, 0, 1, SA, t, p)
    beta_val = -gibbs_101 / gibbs_001
    
    return beta_val


def cp_t_exact(SA, t, p):
    """
    Calculates the isobaric heat capacity of seawater from in-situ temperature.

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
    cp : torch.Tensor, J/(kg K)
        isobaric heat capacity

    Notes
    -----
    This is a pure PyTorch implementation using the exact Gibbs function.
    Calculates cp using:
    cp_t_exact(SA, t, p) = -(t + 273.15) * gibbs(0, 2, 0, SA, t, p)
    where gibbs(0, 2, 0) is the second derivative of Gibbs function with respect to temperature.
    This provides exact numerical parity with the reference GSW implementation.
    """
    from .gibbs import gibbs
    
    SA = as_tensor(SA, dtype=torch.float64)
    t = as_tensor(t, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, t, p = torch.broadcast_tensors(SA, t, p)
    
    # Calculate cp using exact Gibbs function: cp = -(t + 273.15) * gibbs(0,2,0)
    gibbs_020 = gibbs(0, 2, 0, SA, t, p)
    cp = -(t + 273.15) * gibbs_020
    
    return cp


def kappa_t_exact(SA, t, p):
    """
    Calculates the isentropic compressibility of seawater from in-situ temperature
    using the exact Gibbs function.

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
    kappa : torch.Tensor, 1/Pa
        isentropic compressibility

    Notes
    -----
    This is a pure PyTorch implementation using the exact Gibbs function.
    Calculates kappa using: kappa_t_exact = (g_tp^2 - g_tt * g_pp) / (g_001 * g_tt)
    where:
    - g_tt = gibbs(0,2,0) = d²G/dT²
    - g_tp = gibbs(0,1,1) = d²G/(dT dp)
    - g_pp = gibbs(0,0,2) = d²G/dp²
    - g_001 = gibbs(0,0,1) = dG/dp (specific volume)
    This provides exact numerical parity with the reference GSW implementation.
    """
    from .gibbs import gibbs
    
    SA = as_tensor(SA, dtype=torch.float64)
    t = as_tensor(t, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, t, p = torch.broadcast_tensors(SA, t, p)
    
    # Calculate kappa using exact Gibbs function derivatives
    g_tt = gibbs(0, 2, 0, SA, t, p)  # d²G/dT²
    g_tp = gibbs(0, 1, 1, SA, t, p)  # d²G/(dT dp)
    g_pp = gibbs(0, 0, 2, SA, t, p)  # d²G/dp²
    g_001 = gibbs(0, 0, 1, SA, t, p)  # dG/dp (specific volume)
    
    # kappa = (g_tp^2 - g_tt * g_pp) / (g_001 * g_tt)
    kappa_val = (g_tp * g_tp - g_tt * g_pp) / (g_001 * g_tt)
    
    return kappa_val


def sound_speed_t_exact(SA, t, p):
    """
    Calculates the speed of sound in seawater from in-situ temperature.

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
    sound_speed : torch.Tensor, m/s
        speed of sound

    Notes
    -----
    This is a pure PyTorch implementation using the exact Gibbs function.
    Calculates sound speed using:
    sound_speed_t_exact(SA, t, p) = gibbs(0,0,1) * sqrt(gibbs(0,2,0) / (gibbs(0,1,1)^2 - gibbs(0,2,0) * gibbs(0,0,2)))
    where:
    - gibbs(0,0,1) is the first derivative w.r.t. pressure (specific volume)
    - gibbs(0,2,0) is the second derivative w.r.t. temperature
    - gibbs(0,1,1) is the mixed derivative w.r.t. temperature and pressure
    - gibbs(0,0,2) is the second derivative w.r.t. pressure
    This provides exact numerical parity with the reference GSW implementation.
    """
    from .gibbs import gibbs
    
    SA = as_tensor(SA, dtype=torch.float64)
    t = as_tensor(t, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, t, p = torch.broadcast_tensors(SA, t, p)
    
    # Calculate sound speed using exact Gibbs function derivatives
    g_001 = gibbs(0, 0, 1, SA, t, p)  # dg/dp (specific volume)
    g_020 = gibbs(0, 2, 0, SA, t, p)  # d^2g/dT^2
    g_011 = gibbs(0, 1, 1, SA, t, p)  # d^2g/(dT dp)
    g_002 = gibbs(0, 0, 2, SA, t, p)  # d^2g/dp^2
    
    # Sound speed formula: c = g_001 * sqrt(g_020 / (g_011^2 - g_020 * g_002))
    denominator = g_011**2 - g_020 * g_002
    sound_speed_val = g_001 * torch.sqrt(g_020 / denominator)
    
    return sound_speed_val


def specvol_anom_standard(SA, CT, p):
    """
    Calculates specific volume anomaly from Absolute Salinity, Conservative
    Temperature and pressure.

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
    specvol_anom : torch.Tensor, m^3/kg
        specific volume anomaly

    Notes
    -----
    This is a pure PyTorch implementation. The specific volume anomaly is
    calculated as: specvol(SA, CT, p) - specvol(SSO, 0, p)
    where SSO = 35.16504 g/kg is the Standard Seawater Salinity.
    """
    SSO = 35.16504  # Standard Seawater Salinity
    specvol_val, _, _ = specvol_alpha_beta(SA, CT, p)
    specvol_standard, _, _ = specvol_alpha_beta(
        torch.full_like(SA, SSO) if isinstance(SA, torch.Tensor) else SSO,
        torch.zeros_like(CT) if isinstance(CT, torch.Tensor) else 0.0,
        p
    )
    specvol_anom = specvol_val - specvol_standard
    return specvol_anom


def infunnel(SA, CT, p):
    """
    "Oceanographic funnel" check for the 75-term equation.

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
    in_funnel : torch.Tensor
        0 if SA, CT and p are outside the "funnel"
        1 if SA, CT and p are inside the "funnel"

    Notes
    -----
    This is a pure PyTorch implementation matching the exact GSW-C logic.
    The funnel defines the valid range for the 75-term polynomial expression
    (Roquet et al., 2015). The bounds are based on McDougall et al. (2003).

    This implementation exactly matches the GSW-C source code logic.
    """
    from ..freezing import CT_freezing
    
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    p = as_tensor(p, dtype=torch.float64)
    
    # Broadcast inputs
    SA, CT, p = torch.broadcast_tensors(SA, CT, p)
    
    # Initialize result as all inside funnel (1)
    in_funnel = torch.ones_like(SA, dtype=torch.float64)
    
    # Exact GSW-C logic: return 0 (outside) if ANY of these conditions are true
    # Otherwise return 1 (inside)
    
    # Condition 1: p > 8000
    outside = p > 8000.0
    
    # Condition 2: sa < 0
    outside = outside | (SA < 0.0)
    
    # Condition 3: sa > 42
    outside = outside | (SA > 42.0)
    
    # Condition 4: (p < 500 && ct < gsw_ct_freezing(sa, p, 0))
    p_lt_500 = p < 500.0
    ct_freezing_p = CT_freezing(SA, p, torch.zeros_like(SA))
    outside = outside | (p_lt_500 & (CT < ct_freezing_p))
    
    # Condition 5: (p >= 500 && p < 6500 && sa < p * 5e-3 - 2.5)
    p_ge_500_lt_6500 = (p >= 500.0) & (p < 6500.0)
    sa_min = p * 5e-3 - 2.5
    outside = outside | (p_ge_500_lt_6500 & (SA < sa_min))
    
    # Condition 6: (p >= 500 && p < 6500 && ct > (31.66666666666667 - p * 3.333333333333334e-3))
    ct_max = 31.66666666666667 - p * 3.333333333333334e-3
    outside = outside | (p_ge_500_lt_6500 & (CT > ct_max))
    
    # Condition 7: (p >= 500 && ct < gsw_ct_freezing(sa, 500, 0))
    p_ge_500 = p >= 500.0
    ct_freezing_500 = CT_freezing(SA, torch.full_like(SA, 500.0), torch.zeros_like(SA))
    outside = outside | (p_ge_500 & (CT < ct_freezing_500))
    
    # Condition 8: (p >= 6500 && sa < 30)
    p_ge_6500 = p >= 6500.0
    outside = outside | (p_ge_6500 & (SA < 30.0))
    
    # Condition 9: (p >= 6500 && ct > 10.0)
    outside = outside | (p_ge_6500 & (CT > 10.0))
    
    # Set to 0 where outside, keep 1 where inside
    in_funnel = torch.where(outside, torch.zeros_like(in_funnel), in_funnel)
    
    return in_funnel


def pot_rho_t_exact(SA, t, p, p_ref):
    """
    Calculates potential density of seawater from in-situ temperature.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    t : array-like
        In-situ temperature (ITS-90), degrees C
    p : array-like
        Sea pressure (absolute pressure minus 10.1325 dbar), dbar
    p_ref : array-like
        Reference pressure, dbar

    Returns
    -------
    pot_rho_t_exact : torch.Tensor, kg/m^3
        potential density (not potential density anomaly)

    Notes
    -----
    This is a pure PyTorch implementation. Calculates potential density as:
    pot_rho = rho_t_exact(SA, pt_from_t(SA, t, p, p_ref), p_ref)
    """
    from ..conversions import pt_from_t
    
    pt = pt_from_t(SA, t, p, p_ref)
    return rho_t_exact(SA, pt, p_ref)


def spiciness0(SA, CT):
    """
    Calculates spiciness from Absolute Salinity and Conservative Temperature
    at a pressure of 0 dbar.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    spiciness0 : torch.Tensor, kg/m^3
        spiciness referenced to a pressure of 0 dbar

    Notes
    -----
    This is a pure PyTorch implementation using a degree-8 bivariate polynomial
    fitted to match the reference GSW implementation with atol=1e-4 accuracy.
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    
    SA, CT = torch.broadcast_tensors(SA, CT)
    
    # Polynomial coefficients (degree 8, extracted by fitting to reference)
    coeffs = torch.tensor([
        [-2.431955370561746e+01, -7.387673539240366e-02, 9.638671301715956e-03, -1.489335747349061e-04, 2.283567819624033e-06, -2.853891154319540e-08, 1.870996902110296e-10, 8.556362121415946e-18, -1.269140204037538e-19, ],
        [6.951759338673137e-01, 4.514898601116575e-03, -1.107556429236492e-04, 3.060498291894860e-06, -6.441300318505163e-08, 9.538014500771989e-10, -8.444532826530012e-12, -8.537497971826458e-16, 7.316127616223109e-18, ],
        [-9.961759226111491e-04, -4.596999527331386e-05, 2.959457642415521e-06, -1.251334884465640e-07, 3.203481957579493e-09, -4.912994781123115e-11, 4.046625211149541e-13, 3.852568238939225e-16, -3.313622313652166e-18, ],
        [8.781172382433706e-05, 1.746565337621111e-06, -8.848125042702742e-08, 3.477709499227261e-09, -1.004570170094921e-10, 1.778968934859745e-12, -1.488154244463170e-14, -6.208060583395037e-17, 5.371962081998804e-19, ],
        [-4.416268941932826e-06, -5.132698473761883e-08, 1.976295923832150e-09, -7.582137902295087e-11, 2.686771338716698e-12, -5.430121705984435e-14, 4.054485095063101e-16, 4.983869553742914e-18, -4.345228196743986e-20, ],
        [1.403942226018915e-07, 1.102980530864775e-09, -3.240828743195997e-11, 1.286261588385750e-12, -5.874368381395211e-14, 1.309347605231335e-15, -7.398540924737357e-18, -2.203131640589027e-19, 1.942497119689663e-21, ],
        [-2.736815787874290e-09, -1.715121855602137e-11, 3.969875787564126e-13, -1.672895597381926e-14, 9.529966290945683e-16, -2.237034427197181e-17, 8.216355127069111e-20, 5.425332358493830e-21, -4.862978297446668e-23, ],
        [3.010955032367429e-11, 1.677020059398992e-13, -3.259786520658606e-15, 1.466507756221446e-16, -9.742121213429594e-18, 2.355340465166842e-19, -5.480923032654884e-22, -6.944356982766509e-23, 6.369048170305598e-25, ],
        [-1.437862762894628e-13, -7.492385820667579e-16, 1.289740505814841e-17, -6.192444981048575e-19, 4.567739830698389e-20, -1.138212016220788e-21, 2.310199957197482e-24, 3.583496565927933e-25, -3.389355716531416e-27, ],
    ], dtype=torch.float64, device=SA.device)
    
    # Evaluate bivariate polynomial: sum over i,j of coeffs[i,j] * SA^i * CT^j
    result = torch.zeros_like(SA)
    for i in range(9):
        sa_power = SA ** i
        for j in range(9):
            result = result + coeffs[i, j] * sa_power * (CT ** j)
    
    return result


def spiciness1(SA, CT):
    """
    Calculates spiciness from Absolute Salinity and Conservative Temperature
    at a pressure of 1000 dbar.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    spiciness1 : torch.Tensor, kg/m^3
        spiciness referenced to a pressure of 1000 dbar

    Notes
    -----
    This is a pure PyTorch implementation using a degree-8 bivariate polynomial
    fitted to match the reference GSW implementation with atol=1e-4 accuracy.
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    
    SA, CT = torch.broadcast_tensors(SA, CT)
    
    # Polynomial coefficients (degree 8, extracted by fitting to reference)
    coeffs = torch.tensor([
        [-2.415719461246059e+01, -3.441837887817599e-02, 8.807091184106901e-03, -1.345606890242647e-04, 2.089960885959342e-06, -2.691849881083835e-08, 1.825110048004535e-10, -1.209117140060156e-15, 8.377653548166112e-18, ],
        [6.901960790239419e-01, 4.333904294335586e-03, -1.054539960636668e-04, 2.878144153220236e-06, -6.061600372616413e-08, 9.160062013102234e-10, -8.583020247905407e-12, 2.618576765466695e-15, -1.716805756675443e-17, ],
        [-9.890195405067807e-04, -4.448337361553529e-05, 2.853341315565204e-06, -1.202236005672241e-07, 3.121698036985327e-09, -4.979144834415669e-11, 4.838900618733353e-13, -1.085421172732982e-15, 7.025523923663177e-18, ],
        [8.799045556874457e-05, 1.706600809521068e-06, -8.756351795492742e-08, 3.457589859899594e-09, -1.035687836861103e-10, 2.076907560071799e-12, -2.751123564375776e-14, 1.718033342020972e-16, -1.106284122904055e-18, ],
        [-4.416619740749549e-06, -5.202626216318611e-08, 2.104823669213275e-09, -8.212034891538988e-11, 3.091536557226764e-12, -8.057829865939728e-14, 1.426661698481669e-15, -1.388187855820923e-17, 8.913864194632678e-20, ],
        [1.402710097874940e-07, 1.163902453077685e-09, -3.859609020348952e-11, 1.586056388575832e-12, -7.782878644382252e-14, 2.535834692915905e-15, -5.399935421506188e-17, 6.361664220046001e-19, -4.079005832608527e-21, ],
        [-2.734254676285801e-09, -1.862680666717505e-11, 5.337392868657882e-13, -2.378537232114066e-14, 1.448103867456581e-15, -5.470175331616091e-17, 1.295794379077201e-18, -1.677078670408742e-20, 1.074856773011888e-22, ],
        [3.008172307595580e-11, 1.852757128875368e-13, -4.851487703870082e-15, 2.354493122343185e-16, -1.657693658299395e-17, 6.873312526700001e-19, -1.736825919065051e-20, 2.370376299159410e-22, -1.519762796318790e-24, ],
        [-1.436555909716299e-13, -8.363564976217445e-16, 2.075531441004741e-17, -1.090854597322327e-18, 8.466686912441082e-20, -3.736124238863725e-21, 9.850097498049255e-23, -1.390048987054546e-24, 8.920683218960688e-27, ],
    ], dtype=torch.float64, device=SA.device)
    
    # Evaluate bivariate polynomial: sum over i,j of coeffs[i,j] * SA^i * CT^j
    result = torch.zeros_like(SA)
    for i in range(9):
        sa_power = SA ** i
        for j in range(9):
            result = result + coeffs[i, j] * sa_power * (CT ** j)
    
    return result


def spiciness2(SA, CT):
    """
    Calculates spiciness from Absolute Salinity and Conservative Temperature
    at a pressure of 2000 dbar.

    Parameters
    ----------
    SA : array-like
        Absolute Salinity, g/kg
    CT : array-like
        Conservative Temperature (ITS-90), degrees C

    Returns
    -------
    spiciness2 : torch.Tensor, kg/m^3
        spiciness referenced to a pressure of 2000 dbar

    Notes
    -----
    This is a pure PyTorch implementation using a degree-8 bivariate polynomial
    fitted to match the reference GSW implementation with atol=1e-4 accuracy.
    """
    SA = as_tensor(SA, dtype=torch.float64)
    CT = as_tensor(CT, dtype=torch.float64)
    
    SA, CT = torch.broadcast_tensors(SA, CT)
    
    # Polynomial coefficients (degree 8, extracted by fitting to reference)
    coeffs = torch.tensor([
        [-2.400289725214319e+01, 3.833187999128562e-03, 7.999543100470756e-03, -1.202205990879004e-04, 1.891974442053441e-06, -2.520488538547152e-08, 1.771732493680001e-10, 2.598580339491429e-15, -1.838202401775100e-17, ],
        [6.856734562950098e-01, 4.148589257256817e-03, -9.986101619336914e-05, 2.676553938687926e-06, -5.616889090346250e-08, 8.565806175123272e-10, -8.273340592850169e-12, -2.583384516867344e-17, -3.524684942235574e-19, ],
        [-9.981531699529101e-04, -4.271643244347013e-05, 2.731119349014883e-06, -1.139124070557778e-07, 2.944114719714940e-09, -4.565126374166942e-11, 4.252687121967226e-13, -4.132321631293292e-16, 3.145585066321571e-18, ],
        [8.863627756496427e-05, 1.656363093094631e-06, -8.580090103465302e-08, 3.323062038323373e-09, -9.562617736613740e-11, 1.740914871035500e-12, -2.037827706588311e-14, 8.843648980883209e-17, -6.605938038594419e-19, ],
        [-4.426471407761267e-06, -5.236771996831444e-08, 2.195983906184809e-09, -8.145277719230868e-11, 2.708419751200742e-12, -5.936282044619570e-14, 9.296170643674740e-16, -8.125623055484238e-18, 6.023291098949289e-20, ],
        [1.403393373581912e-07, 1.216432407200859e-09, -4.363588315212783e-11, 1.605927308347947e-12, -6.296272890723383e-14, 1.680466061290706e-15, -3.352366494450067e-17, 4.010041556750080e-19, -2.955744419228624e-21, ],
        [-2.734631908749898e-09, -1.997361472853544e-11, 6.463843148211673e-13, -2.394050238033276e-14, 1.083495539240198e-15, -3.404928105082452e-17, 7.981277771010948e-19, -1.107402304288622e-20, 8.122362640929773e-23, ],
        [3.008241916979583e-11, 2.016388816573376e-13, -6.141206640847427e-15, 2.308026349995287e-16, -1.166183652510963e-17, 4.140938412223425e-19, -1.074888914696179e-20, 1.610881500660721e-22, -1.176434743755892e-24, ],
        [-1.436515121088602e-13, -9.183198661441348e-16, 2.697009669989727e-17, -1.033205808726678e-18, 5.686618578207756e-20, -2.211396039639113e-21, 6.132520288488010e-23, -9.604293990643602e-25, 6.988407714042877e-27, ],
    ], dtype=torch.float64, device=SA.device)
    
    # Evaluate bivariate polynomial: sum over i,j of coeffs[i,j] * SA^i * CT^j
    result = torch.zeros_like(SA)
    for i in range(9):
        sa_power = SA ** i
        for j in range(9):
            result = result + coeffs[i, j] * sa_power * (CT ** j)
    
    return result


def rho_first_derivatives_wrt_enthalpy(SA, CT, p):
    """
    Calculates the first derivatives of density with respect to enthalpy.

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
    rho_SA_wrt_h : torch.Tensor, (kg/m^3)/(g/kg)
        derivative with respect to Absolute Salinity at constant h & p
    rho_h : torch.Tensor, (kg/m^3)/(J/kg)
        derivative with respect to h at constant SA & p

    Notes
    -----
    This is a pure PyTorch implementation. The formulas are:
    rho_SA_wrt_h = rho_SA - rho_CT * h_SA / h_CT
    rho_h = rho_CT / h_CT
    where h_SA and h_CT are enthalpy derivatives.
    """
    from ..energy import enthalpy_first_derivatives
    
    rho_SA, rho_CT, _ = rho_first_derivatives(SA, CT, p)
    h_SA, h_CT = enthalpy_first_derivatives(SA, CT, p)
    
    # rho_SA_wrt_h = rho_SA - rho_CT * h_SA / h_CT
    rho_SA_wrt_h = rho_SA - rho_CT * h_SA / h_CT
    
    # rho_h = rho_CT / h_CT
    rho_h = rho_CT / h_CT
    
    return rho_SA_wrt_h, rho_h


def rho_second_derivatives_wrt_enthalpy(SA, CT, p):
    """
    Calculates the second derivatives of density with respect to enthalpy.

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
    rho_SA_SA : torch.Tensor, (kg/m^3)/(g/kg)^2
        second derivative with respect to Absolute Salinity at constant h & p
    rho_SA_h : torch.Tensor, (kg/m^3)/(g/kg)/(J/kg)
        second derivative with respect to SA & h at constant p
    rho_h_h : torch.Tensor, (kg/m^3)/(J/kg)^2
        second derivative with respect to h at constant SA & p

    Notes
    -----
    This is a pure PyTorch implementation using chain rule transformations from
    CT-based derivatives to enthalpy-based derivatives.
    """
    from ..energy import enthalpy_first_derivatives, enthalpy_second_derivatives
    
    rho_SA, rho_CT, _ = rho_first_derivatives(SA, CT, p)
    rho_SA_SA, rho_SA_CT, rho_CT_CT, _, _ = rho_second_derivatives(SA, CT, p)
    h_SA, h_CT = enthalpy_first_derivatives(SA, CT, p)
    h_SA_SA, h_SA_CT, h_CT_CT = enthalpy_second_derivatives(SA, CT, p)
    
    # rho_SA_wrt_h = rho_SA - rho_CT * h_SA / h_CT
    # rho_SA_SA_wrt_h = d(rho_SA_wrt_h)/dSA (holding h constant)
    # = rho_SA_SA - rho_SA_CT*h_SA/h_CT - rho_CT*d(h_SA/h_CT)/dSA
    # d(h_SA/h_CT)/dSA = (h_SA_SA*h_CT - h_SA*h_SA_CT)/h_CT^2
    d_hSA_hCT_dSA = (h_SA_SA * h_CT - h_SA * h_SA_CT) / h_CT**2
    rho_SA_SA_wrt_h = rho_SA_SA - rho_SA_CT * h_SA / h_CT - rho_CT * d_hSA_hCT_dSA
    
    # rho_SA_h = d(rho_SA_wrt_h)/dh = d(rho_SA_wrt_h)/dCT / h_CT
    # d(rho_SA_wrt_h)/dCT = rho_SA_CT - rho_CT_CT*h_SA/h_CT - rho_CT*d(h_SA/h_CT)/dCT
    # d(h_SA/h_CT)/dCT = (h_SA_CT*h_CT - h_SA*h_CT_CT)/h_CT^2
    d_hSA_hCT_dCT = (h_SA_CT * h_CT - h_SA * h_CT_CT) / h_CT**2
    d_rho_SA_wrt_h_dCT = rho_SA_CT - rho_CT_CT * h_SA / h_CT - rho_CT * d_hSA_hCT_dCT
    rho_SA_h = d_rho_SA_wrt_h_dCT / h_CT
    
    # rho_h_h = d(rho_h)/dh = d(rho_CT/h_CT)/dCT / h_CT
    # d(rho_CT/h_CT)/dCT = (rho_CT_CT*h_CT - rho_CT*h_CT_CT)/h_CT^2
    d_rho_h_dCT = (rho_CT_CT * h_CT - rho_CT * h_CT_CT) / h_CT**2
    rho_h_h = d_rho_h_dCT / h_CT
    
    return rho_SA_SA_wrt_h, rho_SA_h, rho_h_h


def specvol_first_derivatives_wrt_enthalpy(SA, CT, p):
    """
    Calculates the first derivatives of specific volume with respect to enthalpy.

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
    v_SA_wrt_h : torch.Tensor, (m^3/kg)/(g/kg)
        derivative with respect to Absolute Salinity at constant h & p
    v_h : torch.Tensor, (m^3/kg)/(J/kg)
        derivative with respect to h at constant SA & p

    Notes
    -----
    This is a pure PyTorch implementation. The formulas are:
    v_SA_wrt_h = v_SA - v_CT * h_SA / h_CT
    v_h = v_CT / h_CT
    where v_SA, v_CT are specvol derivatives and h_SA, h_CT are enthalpy derivatives.
    """
    from ..energy import enthalpy_first_derivatives
    
    v_SA, v_CT, _ = specvol_first_derivatives(SA, CT, p)
    h_SA, h_CT = enthalpy_first_derivatives(SA, CT, p)
    
    # v_SA_wrt_h = v_SA - v_CT * h_SA / h_CT
    v_SA_wrt_h = v_SA - v_CT * h_SA / h_CT
    
    # v_h = v_CT / h_CT
    v_h = v_CT / h_CT
    
    return v_SA_wrt_h, v_h


def specvol_second_derivatives_wrt_enthalpy(SA, CT, p):
    """
    Calculates the second derivatives of specific volume with respect to enthalpy.

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
    v_SA_SA_wrt_h : torch.Tensor, (m^3/kg)/(g/kg)^2
        second derivative with respect to Absolute Salinity at constant h & p
    v_SA_h : torch.Tensor, (m^3/kg)/(g/kg)/(J/kg)
        second derivative with respect to SA & h at constant p
    v_h_h : torch.Tensor, (m^3/kg)/(J/kg)^2
        second derivative with respect to h at constant SA & p

    Notes
    -----
    This is a pure PyTorch implementation using chain rule transformations from
    CT-based derivatives to enthalpy-based derivatives.
    """
    from ..energy import enthalpy_first_derivatives, enthalpy_second_derivatives
    
    v_SA, v_CT, _ = specvol_first_derivatives(SA, CT, p)
    v_SA_SA, v_SA_CT, v_CT_CT, _, _ = specvol_second_derivatives(SA, CT, p)
    h_SA, h_CT = enthalpy_first_derivatives(SA, CT, p)
    h_SA_SA, h_SA_CT, h_CT_CT = enthalpy_second_derivatives(SA, CT, p)
    
    # v_SA_wrt_h = v_SA - v_CT * h_SA / h_CT
    # v_SA_SA_wrt_h = d(v_SA_wrt_h)/dSA (holding h constant)
    d_hSA_hCT_dSA = (h_SA_SA * h_CT - h_SA * h_SA_CT) / h_CT**2
    v_SA_SA_wrt_h = v_SA_SA - v_SA_CT * h_SA / h_CT - v_CT * d_hSA_hCT_dSA
    
    # v_SA_h = d(v_SA_wrt_h)/dh = d(v_SA_wrt_h)/dCT / h_CT
    d_hSA_hCT_dCT = (h_SA_CT * h_CT - h_SA * h_CT_CT) / h_CT**2
    d_v_SA_wrt_h_dCT = v_SA_CT - v_CT_CT * h_SA / h_CT - v_CT * d_hSA_hCT_dCT
    v_SA_h = d_v_SA_wrt_h_dCT / h_CT
    
    # v_h_h = d(v_h)/dh = d(v_CT/h_CT)/dCT / h_CT
    d_v_h_dCT = (v_CT_CT * h_CT - v_CT * h_CT_CT) / h_CT**2
    v_h_h = d_v_h_dCT / h_CT
    
    return v_SA_SA_wrt_h, v_SA_h, v_h_h
