"""
schupy: A Python package for modeling and analyzing Schumann resonances.
"""
from schupy.constants import EPS0, MU0, EARTH_RADIUS, SPEED_OF_LIGHT
from schupy.types import SRSpectrum, HeightModel
from schupy.heights import height_mushtak, height_kulak, get_heights
from schupy.greens import greens, greens_d, greens_pole, greens_d_pole
from schupy.forward import forward_tdte, forward_tdte_pole
from schupy.forward_hyper import forward_hyper, forward_hyper_pole

__version__ = "2.0.0"

__all__ = [
    "EPS0",
    "MU0",
    "EARTH_RADIUS",
    "SPEED_OF_LIGHT",
    "SRSpectrum",
    "HeightModel",
    "height_mushtak",
    "height_kulak",
    "get_heights",
    "greens",
    "greens_d",
    "greens_pole",
    "greens_d_pole",
    "forward_tdte",
    "forward_tdte_pole",
    "forward_hyper",
    "forward_hyper_pole",
    "__version__",
]
