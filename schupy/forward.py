"""
Forward modeling functions for Schumann resonance spectra using Legendre series.
"""
from typing import Sequence, Union
import numpy as np
from schupy.constants import MU0, EARTH_RADIUS
from schupy.heights import get_heights
from schupy.greens import greens, greens_d, greens_pole, greens_d_pole
from schupy.types import SRSpectrum, HeightModel


def forward_tdte(
    s_lat: Union[float, Sequence[float]],
    s_lon: Union[float, Sequence[float]],
    s_int: Union[float, Sequence[float]],
    m_lat: float,
    m_lon: float,
    freq: Union[float, Sequence[float], np.ndarray],
    n_max: int = 5000,
    h: Union[str, HeightModel] = "mushtak",
    tau: float = 0.0,
    ret: str = "all",
) -> Union[SRSpectrum, np.ndarray]:
    """
    Calculate theoretical Schumann resonance power spectral densities using
    Legendre polynomial series summation (Bozóki et al., 2019).

    Parameters
    ----------
    s_lat : float or sequence of float
        Geographical latitude(s) of lightning sources [degrees].
    s_lon : float or sequence of float
        Geographical longitude(s) of lightning sources [degrees].
    s_int : float or sequence of float
        Intensities (charge moment squared) of sources [C^2 km^2 / s].
    m_lat : float
        Geographical latitude of the observing station [degrees].
    m_lon : float
        Geographical longitude of the observing station [degrees].
    freq : float or array_like
        Frequencies at which to compute spectra [Hz].
    n_max : int, optional
        Maximum order of Legendre polynomials in summation (default: 5000).
    h : str or HeightModel, optional
        Height calculation model ('mushtak' or 'kulak', default: 'mushtak').
    tau : float, optional
        Decay time constant for continuing current in seconds (default: 0.0).
        tau=0.0 corresponds to Dirac-delta (impulsive) excitation.
    ret : str, optional
        Returned components: 'all' (default), 'er', 'b_ns', or 'b_ew'.

    Returns
    -------
    result : SRSpectrum or np.ndarray
        If ret='all', returns SRSpectrum(freq, Er, B_NS, B_EW).
        Otherwise returns the specified 1D numpy array.
    """
    s_lat_arr = np.atleast_1d(np.asarray(s_lat, dtype=float))
    s_lon_arr = np.atleast_1d(np.asarray(s_lon, dtype=float))
    # Safe copy and unit conversion: C^2 km^2 / s -> C^2 m^2 / s
    s_int_arr = np.atleast_1d(np.asarray(s_int, dtype=float)) * 1.0e6
    freq_arr = np.atleast_1d(np.asarray(freq, dtype=float))

    assert len(s_lat_arr) == len(s_lon_arr) and len(s_lat_arr) == len(s_int_arr), (
        "s_lat, s_lon and s_int must have the same number of elements."
    )

    omega = 2.0 * np.pi * freq_arr
    he, hm = get_heights(freq_arr, h=h)

    Ez = np.zeros(len(freq_arr), dtype=float)
    B_EW = np.zeros(len(freq_arr), dtype=float)
    B_NS = np.zeros(len(freq_arr), dtype=float)

    xm = np.sin(np.radians(m_lat))
    pm = np.radians(m_lon)
    sin_theta_m = np.sin(np.radians(90.0 - m_lat))

    for s in range(len(s_lat_arr)):
        xs = np.sin(np.radians(s_lat_arr[s]))
        ps = np.radians(s_lon_arr[s])

        # Vertical electric field Er
        ez = greens(freq_arr, EARTH_RADIUS, xs, ps, xm, pm, n_max=n_max, h=h)
        ez_amp = np.abs(1000.0 * ez * (1j * omega * MU0 * hm) / (he ** 2)) ** 2
        Ez += ez_amp * s_int_arr[s]

        # East-West horizontal magnetic field B_EW (B_phi)
        b_ew = greens_d(freq_arr, EARTH_RADIUS, xs, ps, xm, pm, n_max=n_max, t=1, h=h)
        b_ew_amp = np.abs(1.0e12 * b_ew * MU0 / (he * EARTH_RADIUS)) ** 2
        B_EW += b_ew_amp * s_int_arr[s]

        # North-South horizontal magnetic field B_NS (B_theta)
        b_ns = greens_d(freq_arr, EARTH_RADIUS, xs, ps, xm, pm, n_max=n_max, t=2, h=h)
        if sin_theta_m > 1e-12:
            b_ns_amp = np.abs(1.0e12 * b_ns * MU0 / (he * EARTH_RADIUS * sin_theta_m)) ** 2
        else:
            b_ns_amp = np.zeros_like(freq_arr)
        B_NS += b_ns_amp * s_int_arr[s]

    # Apply finite decay time factor |I(omega)|^2 = 1 / (1 + omega^2 * tau^2)
    if tau > 0.0:
        source_factor = 1.0 / (1.0 + (omega * tau) ** 2)
        Ez *= source_factor
        B_EW *= source_factor
        B_NS *= source_factor

    ret_key = str(ret).lower().replace("-", "_").replace(" ", "_")
    if ret_key == "all":
        return SRSpectrum(freq=freq_arr, Er=Ez, B_NS=B_NS, B_EW=B_EW)
    elif ret_key in ("er", "ez"):
        return Ez
    elif ret_key in ("b_ns", "bns", "bt", "btheta"):
        return B_NS
    elif ret_key in ("b_ew", "bew", "bp", "bphi"):
        return B_EW
    else:
        raise ValueError(
            f"Invalid return option '{ret}'. Expected 'all', 'er', 'b_ns', or 'b_ew'."
        )


def forward_tdte_pole(
    theta: Union[float, Sequence[float], np.ndarray],
    s_int: Union[float, Sequence[float]],
    freq: Union[float, Sequence[float], np.ndarray],
    n_max: int = 5000,
    h: Union[str, HeightModel] = "mushtak",
    tau: float = 0.0,
    ret: str = "all",
) -> Union[SRSpectrum, np.ndarray]:
    """
    Simplified fast forward model for a source located at the North Pole (theta'=0).

    In this axisymmetric geometry:
    - Great-circle angular distance gamma equals observer colatitude theta.
    - By rotational symmetry, the meridional magnetic field B_NS is identically 0.
    - The non-zero horizontal magnetic field is purely azimuthal B_EW.

    Parameters
    ----------
    theta : float or array_like
        Observer colatitude / angular distance from North Pole [degrees].
    s_int : float
        Total source intensity [C^2 km^2 / s].
    freq : float or array_like
        Frequencies [Hz].
    n_max : int, optional
        Maximum Legendre summation order (default: 5000).
    h : str or HeightModel, optional
        Height model selector (default: 'mushtak').
    tau : float, optional
        Decay time constant [s] (default: 0.0).
    ret : str, optional
        Returned components ('all', 'er', 'b_ns', 'b_ew').

    Returns
    -------
    result : SRSpectrum or np.ndarray
    """
    theta_rad = np.radians(np.atleast_1d(np.asarray(theta, dtype=float))[0])
    s_int_val = float(np.atleast_1d(np.asarray(s_int, dtype=float))[0]) * 1.0e6
    freq_arr = np.atleast_1d(np.asarray(freq, dtype=float))

    omega = 2.0 * np.pi * freq_arr
    he, hm = get_heights(freq_arr, h=h)
    cos_theta = np.cos(theta_rad)

    ez = greens_pole(freq_arr, cos_theta, EARTH_RADIUS, n_max=n_max, h=h)
    Ez = np.abs(1000.0 * ez * (1j * omega * MU0 * hm) / (he ** 2)) ** 2 * s_int_val

    b_ew = greens_d_pole(freq_arr, cos_theta, EARTH_RADIUS, n_max=n_max, h=h)
    B_EW = np.abs(1.0e12 * b_ew * MU0 / (he * EARTH_RADIUS)) ** 2 * s_int_val

    B_NS = np.zeros_like(Ez)

    if tau > 0.0:
        source_factor = 1.0 / (1.0 + (omega * tau) ** 2)
        Ez *= source_factor
        B_EW *= source_factor

    ret_key = str(ret).lower().replace("-", "_").replace(" ", "_")
    if ret_key == "all":
        return SRSpectrum(freq=freq_arr, Er=Ez, B_NS=B_NS, B_EW=B_EW)
    elif ret_key in ("er", "ez"):
        return Ez
    elif ret_key in ("b_ns", "bns", "bt", "btheta"):
        return B_NS
    elif ret_key in ("b_ew", "bew", "bp", "bphi"):
        return B_EW
    else:
        raise ValueError(
            f"Invalid return option '{ret}'. Expected 'all', 'er', 'b_ns', or 'b_ew'."
        )
