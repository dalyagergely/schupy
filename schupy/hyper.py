"""
Hypergeometric formulation for exact closed-form 2-D Telegraph Equation solution.

References
----------
- Prácser, E. et al. (2021). Two Approaches for Modeling ELF Wave Propagation
  in the Earth-Ionosphere Cavity With Day-Night Asymmetry.
  IEEE Trans. Antennas Propag., 69(7), 4093-4099.
- NIST Digital Library of Mathematical Functions (DLMF), Section 15.8 (Transformations).
"""
import numpy as np
from scipy.special import digamma as cdigamma
from schupy.constants import EPS0, MU0, EARTH_RADIUS


def calc_nu(
    freq: np.ndarray,
    he: np.ndarray,
    hm: np.ndarray,
    R: float = EARTH_RADIUS,
) -> np.ndarray:
    """
    Compute the complex dimensionless propagation parameter nu(f).

    nu(nu + 1) = omega^2 * mu0 * eps0 * (Hm / He) * R^2
    nu = -1/2 + sqrt(1/4 + nu(nu + 1))
    """
    omega = 2.0 * np.pi * freq
    nu_nu1 = (omega * R) ** 2 * MU0 * EPS0 * (hm / he)
    return -0.5 + np.sqrt(0.25 + nu_nu1)


def _hyp2f1_direct(a: complex, b: complex, c: float, z: float, tol: float = 1e-15, max_iter: int = 500) -> complex:
    """Evaluate 2F1(a, b; c; z) using direct power series for |z| < 0.85."""
    term = 1.0 + 0.0j
    s = term
    for k in range(1, max_iter):
        term *= ((a + k - 1.0) * (b + k - 1.0) / ((c + k - 1.0) * k)) * z
        s += term
        if np.abs(term) < tol * np.abs(s):
            break
    return s


def _hyp2f1_near_1(a: complex, b: complex, z: float, tol: float = 1e-15, max_iter: int = 200) -> complex:
    """
    Evaluate 2F1(a, b; a+b; z) for z near 1 (0.85 <= z <= 1.0)
    using DLMF 15.8.1 for c = a + b = 1.
    """
    omz = max(1.0 - z, 1e-30)
    ln_omz = np.log(omz)
    pref = np.sin(b * np.pi) / np.pi

    term_coef = 1.0 + 0.0j
    s = 0.0 + 0.0j

    for k in range(max_iter):
        if k > 0:
            term_coef *= ((a + k - 1.0) * (b + k - 1.0) / (k * k)) * omz

        psi_term = 2.0 * cdigamma(k + 1.0) - cdigamma(a + k) - cdigamma(b + k) - ln_omz
        term = term_coef * psi_term
        s += term
        if k > 0 and np.abs(term) < tol * np.abs(s):
            break

    return pref * s


def eval_p_nu(
    nu: np.ndarray,
    cos_gamma: float,
) -> np.ndarray:
    """
    Evaluate the complex-order Legendre function P_nu(-cos gamma)
    using the Gauss hypergeometric function 2F1:
    P_nu(-x) = 2F1(-nu, nu + 1; 1; (1 + x)/2)
    """
    z = float(np.clip(0.5 * (1.0 + cos_gamma), 0.0, 1.0))
    p_nu = np.empty_like(nu, dtype=np.complex128)

    for i in range(len(nu)):
        a = -nu[i]
        b = nu[i] + 1.0
        if z < 0.85:
            p_nu[i] = _hyp2f1_direct(a, b, 1.0, z)
        else:
            p_nu[i] = _hyp2f1_near_1(a, b, z)

    return p_nu


def eval_dp_nu_dcg(
    nu: np.ndarray,
    cos_gamma: float,
) -> np.ndarray:
    """
    Evaluate d(P_nu(-cos gamma)) / d(cos gamma) using the derivative of 2F1:
    d/dx P_nu(-x) = -(nu*(nu+1)/2) * 2F1(1-nu, nu+2; 2; (1+x)/2)
    """
    z = float(np.clip(0.5 * (1.0 + cos_gamma), 0.0, 1.0))
    dp_nu = np.empty_like(nu, dtype=np.complex128)

    for i in range(len(nu)):
        factor = -0.5 * nu[i] * (nu[i] + 1.0)
        a = 1.0 - nu[i]
        b = nu[i] + 2.0
        c = 2.0
        dp_nu[i] = factor * _hyp2f1_direct(a, b, c, z)

    return dp_nu
