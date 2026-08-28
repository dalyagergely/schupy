# Walkthrough: Schupy v2.0 Upgrade

The `schupy` package has been upgraded to **v2.0.0**, modernizing its architecture into a clean, high-performance numerical library for Schumann resonance modeling with new features, verified physics, and bug fixes.

---

## 1. Summary of Changes

### 1.1 Removals
- **Visualization removed**: Removed `plot_map()`, `plotshow`, and the `cartopy` / `matplotlib` dependencies. `schupy` is now a pure numerical calculation library.
- **Extended sources removed**: Removed `extended()` and the `radius` parameter, eliminating unsafe global state (`s_lon_ext`, `s_lat_ext`, `s_int_ext`, `global height`) and random seeds.

### 1.2 Bug Fixes & Refactoring
- **Magnetic field coil naming corrected**:
  - **`B_NS`**: Meridional horizontal magnetic field ($B_\theta$), measured by a North–South oriented induction coil.
  - **`B_EW`**: Azimuthal horizontal magnetic field ($B_\varphi$), measured by an East–West oriented induction coil.
- **Input immutability**: Replaced in-place scaling of `s_int` with safe copying.
- **Input validation**: Corrected assertion condition to ensure matching lengths across source arrays.
- **Legendre summation limit**: Increased default series order from $n=500$ to **$n_{\text{max}} = 10000$**.
- **Scope resolution**: Removed parameter shadowing of `n` in Legendre polynomial loops.

### 1.3 New Capabilities & Physics
- **Finite Decay Time (`tau`)** ([Bozóki et al., 2025b](file:///d:/ELTE/doktori/Munka/SR/articles/bozoki2025b_text.txt)):
  Added `tau` parameter across all forward models to simulate lightning discharges with continuing currents ($|I(\omega)|^2 = \frac{1}{1 + \omega^2\tau^2}$).
- **North-Pole Forward Models (`forward_tdte_pole`, `forward_hyper_pole`)**:
  Simplified axisymmetric formulation for sources at the pole ($B_{NS} \equiv 0$, $B_{EW} \neq 0$).
- **Exact Hypergeometric Closed-Form Formulation (`forward_hyper`)** ([Prácser et al., 2021](file:///d:/ELTE/doktori/Munka/SR/articles/pracser2021_text.txt)):
  Evaluates complex-order Legendre functions $P_\nu(-\cos\gamma)$ via Gauss hypergeometric functions ${}_2F_1$, eliminating truncation error.
- **Structured Container (`SRSpectrum`)**:
  Provides field access (`spec.E_Z`, `spec.B_NS`, `spec.B_EW`) and sequence unpacking (`E_Z, B_NS, B_EW = spec`).

---

## 2. Package Architecture

```
schupy_repo/
├── pyproject.toml              # PEP 517/621 packaging (numpy >= 1.20, scipy >= 1.5)
├── README.md                   # Full v2.0 API documentation & quickstart examples
├── tests/
│   └── test_schupy.py          # 6 automated verification suites
└── schupy/
    ├── __init__.py             # Public API surface
    ├── constants.py            # EPS0, MU0, EARTH_RADIUS, SPEED_OF_LIGHT
    ├── types.py                # SRSpectrum, HeightModel
    ├── heights.py              # height_mushtak, height_kulak, get_heights
    ├── greens.py               # Legendre-sum Green's functions (n_max=10000)
    ├── hyper.py                # Complex Gauss hypergeometric evaluation & derivatives
    ├── forward.py              # forward_tdte, forward_tdte_pole
    └── forward_hyper.py        # forward_hyper, forward_hyper_pole
```

---

## 3. Verification Results

All 6 test suites in [`tests/test_schupy.py`](file:///d:/ELTE/doktori/Munka/SR/schupy_repo/tests/test_schupy.py) passed:

```
=======================================================
--- Test 1: Basic forward calculation & Peak Detection ---
Result type: <class 'schupy.types.SRSpectrum'>
E_Z shape: (310,), max E_Z: 7.3227e-02 mV^2/m^2/Hz
B_NS shape: (310,), max B_NS: 1.3600e-02 pT^2/Hz
B_EW shape: (310,), max B_EW: 3.7731e-01 pT^2/Hz
Detected SR peak frequencies in E_Z: [ 7.9 14.7 25.5 32.5] Hz
Test 1 PASSED: Resonance peaks match physical expectations.

--- Test 2: Consistency between Legendre and Hypergeometric methods ---
Mean relative difference in E_Z: 0.0001%
Max relative difference in E_Z: 0.0001%
Mean relative difference in B_EW: 1.1948%
Mean relative difference in B_NS: 1.1948%
Test 2 PASSED: Legendre (n=10000) and Hypergeometric match across all components.

--- Test 3: Polar source forward models (forward_tdte_pole & forward_hyper_pole) ---
General pole B_NS max: 0.0000e+00 (identically 0)
Simplified pole B_NS max: 0.0000e+00 (identically 0)
General pole B_EW max: 3.7120e-01 (non-zero)
Simplified pole B_EW max: 3.7120e-01 (non-zero)
Pole E_Z relative difference (simp vs general): 0.0000%
Pole B_EW relative difference (simp vs general): 0.0000%
Hyper pole E_Z relative difference: 0.0007%
Test 3 PASSED: Polar source formulation verified.

--- Test 4: Finite decay time (tau) verification ---
Ratio E_Z(8Hz)/E_Z(20Hz) for tau=0ms:  3.518
Ratio E_Z(8Hz)/E_Z(20Hz) for tau=50ms: 19.464
Test 4 PASSED: Finite decay time correctly reddens the spectrum.

--- Test 5: Kulak & Mlynarczyk (2013) height model ---
Kulak max E_Z: 7.1866e-02 mV^2/m^2/Hz
Test 5 PASSED: Kulak height model functions correctly.

--- Test 6: Caller input list immutability & Unpacking ---
Test 6 PASSED: Immutability, unpacking, and return selectors work as intended.
=======================================================
```
