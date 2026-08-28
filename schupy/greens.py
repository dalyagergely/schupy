"""
Legendre polynomial series summation Green's functions for 2-D Telegraph Equation.

References
----------
- Bozóki, T. et al. (2019). Modeling Schumann resonances with schupy.
  J. Atmos. Sol.-Terr. Phys., 196, 105144.
- Prácser, E. et al. (2019). Reconstruction of global lightning activity based
  on Schumann resonance measurements. Radio Sci., 54(3), 254-267.
"""
from typing import Union
import numpy as np
from schupy.constants import EPS0, MU0, EARTH_RADIUS
from schupy.heights import get_heights
from schupy.types import HeightModel


def calc_zyr2(
    freq: np.ndarray,
    R: float = EARTH_RADIUS,
    h: Union[str, HeightModel] = "mushtak",
) -> np.ndarray:
    """
    Calculate the complex propagation parameter product ZYR2 = nu(nu + 1).

    Note on sign convention:
    Internally, ZYR2 stores +omega^2 * mu0 * eps0 * (Hm / He) * R^2 = nu(nu + 1).
    This matches the denominators n(n+1) - ZYR2 = n(n+1) - nu(nu+1).
    """
    He, Hm = get_heights(freq, h=h)
    omega = 2.0 * np.pi * freq
    C = EPS0 / He
    L = MU0 * Hm
    Z = 1j * omega * L
    Y = -1j * omega * C
    return Y * Z * R * R


def greens(
    freq: np.ndarray,
    R: float,
    xs: float,
    ps: float,
    xm: float,
    pm: float,
    n_max: int = 10000,
    h: Union[str, HeightModel] = "mushtak",
) -> np.ndarray:
    """
    Evaluate the electric field Green's function series using Legendre polynomials.

    Parameters
    ----------
    freq : np.ndarray
        Array of frequencies [Hz].
    R : float
        Earth radius [m].
    xs : float
        cos(colatitude_source) = sin(lat_source).
    ps : float
        Source longitude in radians.
    xm : float
        cos(colatitude_observer) = sin(lat_observer).
    pm : float
        Observer longitude in radians.
    n_max : int, optional
        Maximum order of Legendre polynomials in the summation (default: 10000).
    h : str or HeightModel, optional
        Height model selector (default: 'mushtak').

    Returns
    -------
    gr : np.ndarray (complex128)
        Green's function values evaluated across frequencies.
    """
    ZYR2 = calc_zyr2(freq, R=R, h=h)
    cg = xm * xs + np.sqrt(np.maximum(0.0, 1.0 - xm * xm)) * np.sqrt(
        np.maximum(0.0, 1.0 - xs * xs)
    ) * np.cos(pm - ps)

    p0 = 1.0
    p1 = cg
    gr = -p0 / ZYR2
    gr = gr + 3.0 * p1 / (2.0 - ZYR2)

    for n in range(2, n_max):
        pn = ((2.0 * n - 1.0) * p1 * cg - (n - 1.0) * p0) / n
        grp = (2.0 * n + 1.0) * pn / (n * (n + 1.0) - ZYR2)
        gr = gr + grp
        p0 = p1
        p1 = pn

    return gr / (4.0 * np.pi)


def greens_d(
    freq: np.ndarray,
    R: float,
    xs: float,
    ps: float,
    xm: float,
    pm: float,
    n_max: int = 10000,
    t: int = 1,
    h: Union[str, HeightModel] = "mushtak",
) -> np.ndarray:
    """
    Evaluate the derivative of the Green's function for magnetic field calculation.

    Parameters
    ----------
    freq : np.ndarray
        Array of frequencies [Hz].
    R : float
        Earth radius [m].
    xs, ps, xm, pm : float
        Spherical coordinates (sin(lat), lon in rad).
    n_max : int, optional
        Maximum Legendre polynomial order (default: 10000).
    t : int, optional
        Derivative component:
        t=1 -> d(cos gamma) / d(theta_m) for B_phi (East-West field, B_EW).
        t=2 -> d(cos gamma) / d(phi_m) for B_theta (North-South field, B_NS).
    h : str or HeightModel, optional
        Height model selector.

    Returns
    -------
    gr : np.ndarray (complex128)
        Derivative Green's function values.
    """
    ZYR2 = calc_zyr2(freq, R=R, h=h)
    sin_theta_m = np.sqrt(np.maximum(0.0, 1.0 - xm * xm))
    sin_theta_s = np.sqrt(np.maximum(0.0, 1.0 - xs * xs))
    cg = xm * xs + sin_theta_m * sin_theta_s * np.cos(pm - ps)

    p0 = 1.0
    p0d = 0.0
    p1 = cg
    p1d = 1.0

    gr = -p0d / ZYR2
    gr = gr + 3.0 * p1d / (2.0 - ZYR2)

    for n in range(2, n_max):
        pn = ((2.0 * n - 1.0) * p1 * cg - (n - 1.0) * p0) / n
        pnd = (2.0 * n - 1.0) * p1 + p0d
        grp = (2.0 * n + 1.0) * pnd / (n * (n + 1.0) - ZYR2)
        gr = gr + grp
        p0 = p1
        p1 = pn
        p0d = p1d
        p1d = pnd

    if t == 1:
        # Spatial derivative d(cos gamma) / d(theta_m) for B_EW (B_phi)
        gr = gr * (-sin_theta_m * xs + xm * sin_theta_s * np.cos(pm - ps))
    elif t == 2:
        # Spatial derivative d(cos gamma) / d(phi_m) for B_NS (B_theta)
        gr = -gr * sin_theta_m * sin_theta_s * np.sin(pm - ps)
    else:
        raise ValueError(f"Invalid derivative mode t={t}. Expected 1 (B_EW) or 2 (B_NS).")

    return gr / (4.0 * np.pi)


def greens_pole(
    freq: np.ndarray,
    cos_theta: float,
    R: float = EARTH_RADIUS,
    n_max: int = 10000,
    h: Union[str, HeightModel] = "mushtak",
) -> np.ndarray:
    """
    Simplified electric field Green's function for source at the North Pole.
    """
    ZYR2 = calc_zyr2(freq, R=R, h=h)
    cg = cos_theta

    p0 = 1.0
    p1 = cg
    gr = -p0 / ZYR2
    gr = gr + 3.0 * p1 / (2.0 - ZYR2)

    for n in range(2, n_max):
        pn = ((2.0 * n - 1.0) * p1 * cg - (n - 1.0) * p0) / n
        grp = (2.0 * n + 1.0) * pn / (n * (n + 1.0) - ZYR2)
        gr = gr + grp
        p0 = p1
        p1 = pn

    return gr / (4.0 * np.pi)


def greens_d_pole(
    freq: np.ndarray,
    cos_theta: float,
    R: float = EARTH_RADIUS,
    n_max: int = 10000,
    h: Union[str, HeightModel] = "mushtak",
) -> np.ndarray:
    """
    Simplified azimuthal derivative Green's function for source at the North Pole.
    Evaluates d(Pn(cos theta)) / d(theta) = -sin(theta) * d(Pn)/d(cos theta) = Pn^1(cos theta).
    """
    ZYR2 = calc_zyr2(freq, R=R, h=h)
    sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta * cos_theta))
    cg = cos_theta

    p0 = 1.0
    p0d = 0.0
    p1 = cg
    p1d = 1.0

    gr = -p0d / ZYR2
    gr = gr + 3.0 * p1d / (2.0 - ZYR2)

    for n in range(2, n_max):
        pn = ((2.0 * n - 1.0) * p1 * cg - (n - 1.0) * p0) / n
        pnd = (2.0 * n - 1.0) * p1 + p0d
        grp = (2.0 * n + 1.0) * pnd / (n * (n + 1.0) - ZYR2)
        gr = gr + grp
        p0 = p1
        p1 = pn
        p0d = p1d
        p1d = pnd

    # d(cos theta) / d(theta) = -sin(theta)
    gr = gr * (-sin_theta)
    return gr / (4.0 * np.pi)
