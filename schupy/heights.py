"""
Ionospheric characteristic height calculation models.

References
----------
- Mushtak, V. C., & Williams, E. R. (2002). ELF propagation parameters for
  uniform models of the Earth-ionosphere waveguide. J. Atmos. Sol.-Terr. Phys., 64.
- Pechony, O., & Price, C. (2004). Schumann resonance parameters calculated
  with a partially uniform knee model on Earth, Venus, Mars, and Titan.
  Radio Sci., 39(5), RS5007.
- Kulak, A., & Mlynarczyk, J. (2013). ELF propagation parameters for the
  ground-ionosphere waveguide with finite ground conductivity.
  IEEE Trans. Antennas Propag., 61(4), 2269-2275.
"""
from typing import Tuple, Union
import numpy as np
from schupy.types import HeightModel


def height_mushtak(
    freq: Union[float, np.ndarray],
    fkn: float = 10.0,
    hkn: float = 55000.0,
    kszib: float = 8300.0,
    kszia: float = 2900.0,
    fm: float = 8.0,
    hm: float = 96500.0,
    kszim: float = 4000.0,
    bm: float = 6500.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute complex electric (He) and magnetic (Hm) characteristic altitudes
    using the two-scale-height 'knee' model of Mushtak & Williams (2002).

    Parameters
    ----------
    freq : float or np.ndarray
        Frequencies at which to compute altitudes [Hz].
    fkn : float, optional
        Knee reference frequency [Hz] (default: 10.0).
    hkn : float, optional
        Electric knee altitude [m] (default: 55000.0).
    kszib : float, optional
        Scale height below electric knee [m] (default: 8300.0).
    kszia : float, optional
        Scale height above electric knee [m] (default: 2900.0).
    fm : float, optional
        Magnetic reference frequency [Hz] (default: 8.0).
    hm : float, optional
        Magnetic altitude at fm [m] (default: 96500.0).
    kszim : float, optional
        Magnetic scale height [m] (default: 4000.0).
    bm : float, optional
        Magnetic frequency parameter [m] (default: 6500.0).

    Returns
    -------
    He : np.ndarray (complex128)
        Complex capacitive / electric characteristic height [m].
    Hm : np.ndarray (complex128)
        Complex inductive / magnetic characteristic height [m].
    """
    freq_arr = np.atleast_1d(np.asarray(freq, dtype=float))

    Re_He = (
        hkn
        + kszia * np.log(freq_arr / fkn)
        + 0.5 * (kszia - kszib) * np.log(1.0 + (fkn / freq_arr) ** 2)
    )
    Im_He = -0.5 * np.pi * kszia + (kszia - kszib) * np.arctan(fkn / freq_arr)

    Re_Hm = hm - (kszim + bm * (1.0 / freq_arr - 1.0 / fm)) * np.log(freq_arr / fm)
    Im_Hm = 0.5 * np.pi * (kszim + bm * (1.0 / freq_arr - 1.0 / fm))

    He = Re_He + 1j * Im_He
    Hm = Re_Hm + 1j * Im_Hm

    return He, Hm


def height_kulak(
    freq: Union[float, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute complex electric (He) and magnetic (Hm) characteristic altitudes
    using the day/night averaged model of Kulak & Mlynarczyk (2013).

    Parameters
    ----------
    freq : float or np.ndarray
        Frequencies at which to compute altitudes [Hz].

    Returns
    -------
    He : np.ndarray (complex128)
        Complex capacitive / electric characteristic height [m].
    Hm : np.ndarray (complex128)
        Complex inductive / magnetic characteristic height [m].
    """
    freq_arr = np.atleast_1d(np.asarray(freq, dtype=float))

    # Nighttime electric altitude [km] (Eqs. 27-28 in Kulak & Mlynarczyk 2013)
    Re_He_n = (
        67.5
        + 2.0 * np.log(freq_arr / 7.7)
        - 2.54 * (7.7 / freq_arr) ** 0.813
        - 2.72 * (7.7 / freq_arr) ** 1.626
    )
    Im_He_n = (
        -3.14 - 8.70 * (7.7 / freq_arr) ** 0.813 + 1.92 * (7.7 / freq_arr) ** 1.626
    )

    # Nighttime magnetic altitude [km] (Eqs. 31-32)
    Re_Hm_n = 114.7 - 8.4 * np.log(freq_arr / 7.7)
    Im_Hm_n = 13.2 - 2.0 * np.log(freq_arr / 7.7)

    # Daytime electric altitude [km] (Eqs. 29-30)
    Re_He_d = (
        51.1
        + 1.9 * np.log(freq_arr / 1.7)
        - 2.45 * (1.7 / freq_arr) ** 0.822
        - 2.84 * (1.7 / freq_arr) ** 1.645
    )
    Im_He_d = (
        -2.98 - 8.80 * (1.7 / freq_arr) ** 0.822 + 1.86 * (1.7 / freq_arr) ** 1.645
    )

    # Daytime magnetic altitude [km] (Eqs. 33-34)
    Re_Hm_d = 101.5 - 3.1 * np.log(freq_arr / 7.7)
    Im_Hm_d = 7.0 - 0.9 * np.log(freq_arr / 7.7)

    # Convert km -> m and average day/night
    He_n = 1000.0 * (Re_He_n + 1j * Im_He_n)
    Hm_n = 1000.0 * (Re_Hm_n + 1j * Im_Hm_n)
    He_d = 1000.0 * (Re_He_d + 1j * Im_He_d)
    Hm_d = 1000.0 * (Re_Hm_d + 1j * Im_Hm_d)

    return 0.5 * (He_n + He_d), 0.5 * (Hm_n + Hm_d)


def get_heights(
    freq: Union[float, np.ndarray],
    h: Union[str, HeightModel] = "mushtak",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Resolve and compute the complex characteristic heights for the chosen model.

    Parameters
    ----------
    freq : float or np.ndarray
        Frequencies [Hz].
    h : str or HeightModel
        Height model selector ('mushtak' or 'kulak').

    Returns
    -------
    He, Hm : tuple of np.ndarray
        Complex electric and magnetic characteristic heights [m].
    """
    h_str = str(h).lower()
    if h_str in ("mushtak", HeightModel.MUSHTAK):
        return height_mushtak(freq)
    elif h_str in ("kulak", HeightModel.KULAK):
        return height_kulak(freq)
    else:
        raise ValueError(
            f"Invalid height calculation model '{h}'. Expected 'mushtak' or 'kulak'."
        )
