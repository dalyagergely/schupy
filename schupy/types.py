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
    E_Z : np.ndarray
        Vertical electric field power spectral density [mV^2 / m^2 / Hz].
    B_NS : np.ndarray
        Meridional horizontal magnetic field power spectral density [pT^2 / Hz],
        measured by a North-South oriented induction coil (detects B_theta).
    B_EW : np.ndarray
        Azimuthal horizontal magnetic field power spectral density [pT^2 / Hz],
        measured by an East-West oriented induction coil (detects B_phi).
    """
    freq: np.ndarray
    E_Z: np.ndarray
    B_NS: np.ndarray
    B_EW: np.ndarray

    @property
    def E_z(self) -> np.ndarray:
        """Alias for E_Z."""
        return self.E_Z

    @property
    def Ez(self) -> np.ndarray:
        """Alias for E_Z."""
        return self.E_Z

    @property
    def Er(self) -> np.ndarray:
        """Backwards-compatibility alias for E_Z."""
        return self.E_Z

    def __iter__(self):
        """Allow sequence unpacking: E_Z, B_NS, B_EW = res"""
        return iter((self.E_Z, self.B_NS, self.B_EW))

    def __getitem__(self, item):
        """Allow tuple-style indexing res[0], res[1], res[2]."""
        return (self.E_Z, self.B_NS, self.B_EW)[item]
