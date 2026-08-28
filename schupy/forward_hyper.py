"""
Exact closed-form forward modeling using Gauss hypergeometric functions (Prácser et al., 2021).
"""
from typing import Sequence, Union
import numpy as np
from schupy.constants import MU0, EARTH_RADIUS
from schupy.heights import get_heights
from schupy.hyper import calc_nu, eval_p_nu, eval_dp_nu_dcg
from schupy.types import SRSpectrum, HeightModel


def forward_hyper(
    s_lat: Union[float, Sequence[float]],
    s_lon: Union[float, Sequence[float]],
    s_int: Union[float, Sequence[float]],
    m_lat: float,
    m_lon: float,
    freq: Union[float, Sequence[float], np.ndarray],
    h: Union[str, HeightModel] = "mushtak",
    tau: float = 0.0,
    ret: str = "all",
) -> Union[SRSpectrum, np.ndarray]:
    """
    Calculate theoretical Schumann resonance spectra via exact hypergeometric
    evaluation of the complex-order Legendre functions (Prácser et al., 2021).

    Parameters
    ----------
    s_lat, s_lon : float or sequence of float
        Source latitudes and longitudes [degrees].
    s_int : float or sequence of float
        Source intensities [C^2 km^2 / s].
    m_lat, m_lon : float
        Observer latitude and longitude [degrees].
    freq : float or array_like
        Frequencies [Hz].
    h : str or HeightModel, optional
        Height calculation model ('mushtak' or 'kulak').
    tau : float, optional
        Decay time constant [s] (default: 0.0).
    ret : str, optional
        Returned components ('all', 'e_z', 'b_ns', 'b_ew').

    Returns
    -------
    result : SRSpectrum or np.ndarray
    """
    s_lat_arr = np.atleast_1d(np.asarray(s_lat, dtype=float))
    s_lon_arr = np.atleast_1d(np.asarray(s_lon, dtype=float))
    s_int_arr = np.atleast_1d(np.asarray(s_int, dtype=float)) * 1.0e6
    freq_arr = np.atleast_1d(np.asarray(freq, dtype=float))

    assert len(s_lat_arr) == len(s_lon_arr) and len(s_lat_arr) == len(s_int_arr), (
        "s_lat, s_lon and s_int must have the same number of elements."
    )

    omega = 2.0 * np.pi * freq_arr
    he, hm = get_heights(freq_arr, h=h)
    nu = calc_nu(freq_arr, he, hm, R=EARTH_RADIUS)
    sin_nu_pi = np.sin(nu * np.pi)

    E_Z = np.zeros(len(freq_arr), dtype=float)
    B_EW = np.zeros(len(freq_arr), dtype=float)
    B_NS = np.zeros(len(freq_arr), dtype=float)

    xm = np.sin(np.radians(m_lat))
    pm = np.radians(m_lon)
    sin_theta_m = np.sin(np.radians(90.0 - m_lat))
    cos_theta_m = xm

    for s in range(len(s_lat_arr)):
        xs = np.sin(np.radians(s_lat_arr[s]))
        ps = np.radians(s_lon_arr[s])
        sin_theta_s = np.cos(np.radians(s_lat_arr[s]))

        cos_gamma = xm * xs + sin_theta_m * sin_theta_s * np.cos(pm - ps)
        cos_gamma = np.clip(cos_gamma, -1.0, 1.0)

        # Evaluate P_nu(-cos gamma)
        p_nu = eval_p_nu(nu, cos_gamma)
        dp_nu_dcg = eval_dp_nu_dcg(nu, cos_gamma)

        # E_Z component (mV / m / sqrt(Hz) per unit C*m)
        er_amp = (
            1000.0
            * (1j * omega * MU0 * hm)
            / (4.0 * np.pi * he ** 2)
            * (-np.pi * p_nu / sin_nu_pi)
        )
        E_Z += np.abs(er_amp) ** 2 * s_int_arr[s]

        # Derivatives of cos_gamma
        d_cg_d_theta_m = -sin_theta_m * xs + cos_theta_m * sin_theta_s * np.cos(pm - ps)
        d_cg_d_phi_m = -sin_theta_m * sin_theta_s * np.sin(pm - ps)

        # B_EW component (pT)
        b_ew_amp = (
            1.0e12
            * (MU0 / (4.0 * np.pi * he * EARTH_RADIUS))
            * (-np.pi * dp_nu_dcg * d_cg_d_theta_m / sin_nu_pi)
        )
        B_EW += np.abs(b_ew_amp) ** 2 * s_int_arr[s]

        # B_NS component (pT)
        if sin_theta_m > 1e-12:
            b_ns_amp = (
                1.0e12
                * (MU0 / (4.0 * np.pi * he * EARTH_RADIUS * sin_theta_m))
                * (np.pi * dp_nu_dcg * d_cg_d_phi_m / sin_nu_pi)
            )
        else:
            b_ns_amp = np.zeros_like(freq_arr)
        B_NS += np.abs(b_ns_amp) ** 2 * s_int_arr[s]

    if tau > 0.0:
        source_factor = 1.0 / (1.0 + (omega * tau) ** 2)
        E_Z *= source_factor
        B_EW *= source_factor
        B_NS *= source_factor

    ret_key = str(ret).lower().replace("-", "_").replace(" ", "_")
    if ret_key == "all":
        return SRSpectrum(freq=freq_arr, E_Z=E_Z, B_NS=B_NS, B_EW=B_EW)
    elif ret_key in ("e_z", "ez", "er"):
        return E_Z
    elif ret_key in ("b_ns", "bns", "bt", "btheta"):
        return B_NS
    elif ret_key in ("b_ew", "bew", "bp", "bphi"):
        return B_EW
    else:
        raise ValueError(
            f"Invalid return option '{ret}'. Expected 'all', 'e_z', 'b_ns', or 'b_ew'."
        )


def forward_hyper_pole(
    theta: Union[float, Sequence[float], np.ndarray],
    s_int: Union[float, Sequence[float]],
    freq: Union[float, Sequence[float], np.ndarray],
    h: Union[str, HeightModel] = "mushtak",
    tau: float = 0.0,
    ret: str = "all",
) -> Union[SRSpectrum, np.ndarray]:
    """
    Simplified exact hypergeometric forward model with source at the North Pole.
    """
    theta_rad = np.radians(np.atleast_1d(np.asarray(theta, dtype=float))[0])
    s_int_val = float(np.atleast_1d(np.asarray(s_int, dtype=float))[0]) * 1.0e6
    freq_arr = np.atleast_1d(np.asarray(freq, dtype=float))

    omega = 2.0 * np.pi * freq_arr
    he, hm = get_heights(freq_arr, h=h)
    nu = calc_nu(freq_arr, he, hm, R=EARTH_RADIUS)
    sin_nu_pi = np.sin(nu * np.pi)

    cos_theta = np.cos(theta_rad)
    sin_theta = np.sin(theta_rad)

    p_nu = eval_p_nu(nu, cos_theta)
    dp_nu_dcg = eval_dp_nu_dcg(nu, cos_theta)

    er_amp = (
        1000.0
        * (1j * omega * MU0 * hm)
        / (4.0 * np.pi * he ** 2)
        * (-np.pi * p_nu / sin_nu_pi)
    )
    E_Z = np.abs(er_amp) ** 2 * s_int_val

    # For pole source, d(cos theta)/d(theta) = -sin(theta)
    b_ew_amp = (
        1.0e12
        * (MU0 / (4.0 * np.pi * he * EARTH_RADIUS))
        * (np.pi * dp_nu_dcg * sin_theta / sin_nu_pi)
    )
    B_EW = np.abs(b_ew_amp) ** 2 * s_int_val
    B_NS = np.zeros_like(E_Z)

    if tau > 0.0:
        source_factor = 1.0 / (1.0 + (omega * tau) ** 2)
        E_Z *= source_factor
        B_EW *= source_factor

    ret_key = str(ret).lower().replace("-", "_").replace(" ", "_")
    if ret_key == "all":
        return SRSpectrum(freq=freq_arr, E_Z=E_Z, B_NS=B_NS, B_EW=B_EW)
    elif ret_key in ("e_z", "ez", "er"):
        return E_Z
    elif ret_key in ("b_ns", "bns", "bt", "btheta"):
        return B_NS
    elif ret_key in ("b_ew", "bew", "bp", "bphi"):
        return B_EW
    else:
        raise ValueError(
            f"Invalid return option '{ret}'. Expected 'all', 'e_z', 'b_ns', or 'b_ew'."
        )
