"""
Physical constants used in Schumann resonance modeling.
"""
import numpy as np

# Permittivity of free space [F/m]
EPS0: float = 8.8541878128e-12

# Magnetic permeability of free space [H/m]
MU0: float = 4.0e-7 * np.pi

# Mean radius of the Earth [m]
EARTH_RADIUS: float = 6371000.0

# Speed of light in vacuum [m/s]
SPEED_OF_LIGHT: float = 1.0 / np.sqrt(EPS0 * MU0)
