"""
Type definitions and structured result containers for schupy.
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np


class HeightModel(str, Enum):
    """Supported ionospheric height calculation models."""
    MUSHTAK = "mushtak"
    KULAK = "kulak"


@dataclass(frozen=True)
class SRSpectrum:
    """
    Container for modeled Schumann resonance power spectral densities.

    Attributes
    ----------
    freq : np.ndarray
        Array of evaluation frequencies [Hz].
    Er : np.ndarray
        Vertical electric field power spectral density [mV^2 / m^2 / Hz].
    B_NS : np.ndarray
        Meridional horizontal magnetic field power spectral density [pT^2 / Hz],
        measured by a North-South oriented induction coil (detects B_theta).
    B_EW : np.ndarray
        Azimuthal horizontal magnetic field power spectral density [pT^2 / Hz],
        measured by an East-West oriented induction coil (detects B_phi).
    """
    freq: np.ndarray
    Er: np.ndarray
    B_NS: np.ndarray
    B_EW: np.ndarray

    def __iter__(self):
        """Allow sequence unpacking: Er, B_NS, B_EW = res"""
        return iter((self.Er, self.B_NS, self.B_EW))

    def __getitem__(self, item):
        """Allow tuple-style indexing res[0], res[1], res[2]."""
        return (self.Er, self.B_NS, self.B_EW)[item]
