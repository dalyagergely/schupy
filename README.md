# schupy -- A Python Package for Modeling Schumann Resonances

`schupy` is an open-source Python package aimed at modeling and analyzing Schumann resonances (SRs), the global electromagnetic resonances of the Earth-ionosphere cavity resonator in the extremely low frequency (ELF) band (<100 Hz).

---

## Installation

```bash
pip install .
```

### Dependencies
- **`numpy >= 1.20`**
- **`scipy >= 1.5`** (for exact hypergeometric closed-form calculations)

---

## Features

- **`forward_tdte`**: General forward model for arbitrary source-observer configurations using Legendre polynomial series summation up to $n_{\text{max}} = 10000$ (Bozóki et al., 2019).
- **`forward_tdte_pole`**: Fast axisymmetric forward model for sources located at the North Pole.
- **`forward_hyper`**: Exact closed-form forward model using Gauss hypergeometric function ${}_2F_1$ (Prácser et al., 2021), eliminating truncation errors.
- **`forward_hyper_pole`**: Exact closed-form polar forward model.
- **Finite Decay Time (`tau`)**: Support for lightning continuing currents with exponential decay time constant $\tau$ (Bozóki et al., 2025b).
- **Height Models**: Mushtak & Williams (2002) knee model and Kulak & Mlynarczyk (2013) day/night model.

---

## Magnetic Field Naming Convention

Horizontal magnetic field components are labeled according to the orientation of the measuring induction coils:

| Component | Description | Spherical Field Component | Measured In |
|---|---|---|---|
| **`E_z`** | Vertical electric field | $E_z$ | $\text{mV}^2 / \text{m}^2 / \text{Hz}$ |
| **`B_NS`** | Meridional horizontal magnetic field | $B_\theta$ | $\text{pT}^2 / \text{Hz}$ |
| **`B_EW`** | Azimuthal horizontal magnetic field | $B_\varphi$ | $\text{pT}^2 / \text{Hz}$ |

---

## Quickstart

### 1. General Forward Calculation (`forward_tdte`)

```python
import schupy as sp
import numpy as np

# Define source and observer
source_latitudes = [10.0, 0.0, 0.0]
source_longitudes = [10.0, -80.0, 110.0]
source_intensities = [1e5, 8e4, 7e4]  # C^2 km^2 / s
obs_latitude = 47.6
obs_longitude = 16.7
frequencies = np.arange(4.0, 35.0, 0.1)

# Run model
spectrum = sp.forward_tdte(
    s_lat=source_latitudes,
    s_lon=source_longitudes,
    s_int=source_intensities,
    m_lat=obs_latitude,
    m_lon=obs_longitude,
    freq=frequencies,
    h="mushtak",
    tau=0.0  # Impulsive excitation (Dirac delta)
)

# Access fields as attributes or unpack
print(spectrum.freq)
print(spectrum.E_z)
print(spectrum.B_NS)
print(spectrum.B_EW)

# Or unpack directly:
E_z, B_NS, B_EW = spectrum
```

### 2. Fast Exact Hypergeometric Model (`forward_hyper`)

```python
# Exact closed-form solution via Gauss hypergeometric functions (Prácser et al., 2021)
spec_exact = sp.forward_hyper(
    s_lat=source_latitudes,
    s_lon=source_longitudes,
    s_int=source_intensities,
    m_lat=obs_latitude,
    m_lon=obs_longitude,
    freq=frequencies,
)
```

### 3. Source at the North Pole (`forward_tdte_pole` / `forward_hyper_pole`)

```python
# Observer at colatitude theta = 42.4 degrees
spec_pole = sp.forward_tdte_pole(
    theta=42.4,
    s_int=1.0e5,
    freq=frequencies,
)
# Note: B_NS is identically 0 by rotational symmetry
```

### 4. Lightning with Continuing Current (`tau > 0`)

```python
# Model lightning with a 20 ms decay time constant (Bozóki et al., 2025b)
spec_cc = sp.forward_tdte(
    s_lat=[0.0],
    s_lon=[0.0],
    s_int=[1.0e5],
    m_lat=0.0,
    m_lon=60.0,
    freq=frequencies,
    tau=0.020  # 20 ms
)
```

---

## References & Citation

If you use `schupy` in your research, please cite:

```bibtex
@article{bozoki2019schupy,
  title = {Modeling Schumann resonances with schupy},
  author = {Boz{'o}ki, Tam{'a}s and Pr{'a}cser, Ern{\H{o}} and S{'a}tori, Gabriella and D{'a}lya, Gergely and Kap{'a}s, Korn{'e}l and Tak{'a}tsy, J{'a}nos},
  journal = {Journal of Atmospheric and Solar-Terrestrial Physics},
  volume = {196},
  pages = {105144},
  year = {2019},
  doi = {10.1016/j.jastp.2019.105144}
}
```

Additional foundational literature implemented in `schupy`:
- **Bozóki, T. et al. (2025)**: *Modeling the Global Electromagnetic Resonance Field Produced by Lightning Discharges With a Continuing Current*, J. Geophys. Res. Atmos., 130, e2025JD043989.
- **Prácser, E. et al. (2021)**: *Two Approaches for Modeling ELF Wave Propagation in the Earth-Ionosphere Cavity With Day-Night Asymmetry*, IEEE Trans. Antennas Propag., 69(7), 4093-4099.
- **Kulak, A., & Mlynarczyk, J. (2013)**: *ELF Propagation Parameters for the Ground-Ionosphere Waveguide With Finite Ground Conductivity*, IEEE Trans. Antennas Propag., 61(4), 2269-2275.
- **Mushtak, V. C., & Williams, E. R. (2002)**: *ELF propagation parameters for uniform models of the Earth-ionosphere waveguide*, J. Atmos. Sol.-Terr. Phys., 64, 1989-2001.
