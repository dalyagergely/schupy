# Schupy v2.0: User Guide & Quickstart

`schupy` is a Python library for numerical modeling and analytical simulation of **Schumann resonances (SRs)** in the Earth–ionosphere cavity resonator (< 100 Hz).

---

## Table of Contents
1. [Installation & Requirements](#1-installation--requirements)
2. [Key Concepts & Conventions](#2-key-concepts--conventions)
3. [The 4 Core Forward Models](#3-the-4-core-forward-models)
4. [Step-by-Step Code Examples](#4-step-by-step-code-examples)
   - [Example 1: Basic Single-Source Simulation](#example-1-basic-single-source-simulation)
   - [Example 2: Multiple Global Lightning Centers](#example-2-multiple-global-lightning-centers)
   - [Example 3: Modeling Continuing Currents ($\tau > 0$)](#example-3-modeling-continuing-currents-tau--0)
   - [Example 4: High-Precision Hypergeometric Model](#example-4-high-precision-hypergeometric-model)
   - [Example 5: Axisymmetric North-Pole Geometry](#example-5-axisymmetric-north-pole-geometry)
   - [Example 6: Plotting Results with Matplotlib](#example-6-plotting-results-with-matplotlib)
5. [Input Parameters & Output Cheat Sheet](#5-input-parameters--output-cheat-sheet)
6. [References](#6-references)

---

## 1. Installation & Requirements

### Requirements
- **Python $\ge 3.8$**
- **`numpy >= 1.20`**
- **`scipy >= 1.5`**

### Install from source
Navigate to the `schupy_repo` directory and install using `pip`:
```bash
pip install .
```

For development (editable install):
```bash
pip install -e .
```

Verify the installation in Python:
```python
import schupy as sp
print(sp.__version__)  # Outputs: 2.0.0
```

---

## 2. Key Concepts & Conventions

### Magnetic Field Coil Naming
In Schumann resonance observatories, horizontal magnetic field components are recorded by induction coil magnetometers. `schupy` labels them by the **orientation of the coil axis**:

| Component | Physical Meaning | Spherical Field Component | Output Unit |
|---|---|---|---|
| **`E_z`** | Vertical electric field | $E_z$ | $\text{mV}^2 / \text{m}^2 / \text{Hz}$ |
| **`B_NS`** | Meridional magnetic field (along longitude) | $B_\theta$ | $\text{pT}^2 / \text{Hz}$ |
| **`B_EW`** | Azimuthal magnetic field (along latitude) | $B_\varphi$ | $\text{pT}^2 / \text{Hz}$ |

> **Note for North-Pole Sources:** When a source is located at the North Pole, rotational symmetry causes the meridional field to vanish ($B_{NS} \equiv 0$), while the East–West field is non-zero ($B_{EW} \neq 0$).

### Supported Height Models (`h`)
1. **`"mushtak"`** (Default): Two-scale-height exponential "knee" model (*Mushtak & Williams, 2002; Pechony & Price, 2004*).
2. **`"kulak"`**: Day/night averaged complex altitude model (*Kulak & Mlynarczyk, 2013*).

### Source Finite Decay Time (`tau`)
- Standard lightning return strokes are short impulses modeled as Dirac-delta sources ($\tau = 0.0\text{ s}$).
- Long-duration discharges with a **continuing current (CC)** are modeled with an exponential decay time constant $\tau > 0$ (*Bozóki et al., 2025*):
  $$|I(\omega)|^2 = \frac{1}{1 + \omega^2 \tau^2}$$
  Typical values: `tau=0.010` (10 ms), `tau=0.020` (20 ms), `tau=0.050` (50 ms). Increasing $\tau$ makes the spectrum more "reddish" (boosts power near the 8 Hz fundamental mode).

---

## 3. The 4 Core Forward Models

| Function | Method | Best Used For |
|---|---|---|
| **`sp.forward_tdte(...)`** | Legendre polynomial series ($n=10000$) | Standard modeling for arbitrary global source-observer configurations. |
| **`sp.forward_hyper(...)`** | Exact Gauss hypergeometric function ${}_2F_1$ | Exact analytical solution without series truncation errors. |
| **`sp.forward_tdte_pole(...)`** | Legendre series ($n=10000$) with source at Pole | High-speed axisymmetric simulation parameterized by colatitude $\theta$. |
| **`sp.forward_hyper_pole(...)`** | Hypergeometric ${}_2F_1$ with source at Pole | Exact closed-form axisymmetric polar simulation. |

---

## 4. Step-by-Step Code Examples

### Example 1: Basic Single-Source Simulation

```python
import schupy as sp
import numpy as np

# 1. Define source and observer positions (in degrees)
source_lat = [10.0]       # Latitude (deg N)
source_lon = [10.0]       # Longitude (deg E)
source_int = [1.0e5]      # Intensity (C^2 km^2 / s)

obs_lat = 47.6            # Nagycenk Observatory, Hungary
obs_lon = 16.7

# 2. Define frequency grid (in Hz)
freq = np.arange(4.0, 35.0, 0.1)

# 3. Compute Schumann resonance spectrum
result = sp.forward_tdte(
    s_lat=source_lat,
    s_lon=source_lon,
    s_int=source_int,
    m_lat=obs_lat,
    m_lon=obs_lon,
    freq=freq,
    h="mushtak"           # Height model: 'mushtak' or 'kulak'
)

# 4. Access output fields
print("Frequencies:", result.freq[:5])
print("E_z (first 5 points):", result.E_z[:5])
print("B_NS (first 5 points):", result.B_NS[:5])
print("B_EW (first 5 points):", result.B_EW[:5])

# Alternatively, unpack directly like a tuple:
E_z, B_NS, B_EW = result
```

---

### Example 2: Multiple Global Lightning Centers

Simulate the 3 primary global thunderstorm regions (Africa, Americas, Southeast Asia):

```python
import schupy as sp
import numpy as np

# Africa, South America, Maritime Continent
source_lats = [5.0, -10.0, 0.0]
source_lons = [20.0, -60.0, 110.0]
source_ints = [1.2e5, 0.9e5, 0.8e5]  # C^2 km^2 / s

obs_lat = 47.6
obs_lon = 16.7
freq = np.arange(4.0, 40.0, 0.1)

# Summed power spectral density from all 3 incoherent source regions:
spectrum = sp.forward_tdte(source_lats, source_lons, source_ints, obs_lat, obs_lon, freq)

print(f"Max E_z: {np.max(spectrum.E_z):.4e} mV^2/m^2/Hz")
print(f"Max B_NS: {np.max(spectrum.B_NS):.4e} pT^2/Hz")
print(f"Max B_EW: {np.max(spectrum.B_EW):.4e} pT^2/Hz")
```

---

### Example 3: Modeling Continuing Currents ($\tau > 0$)

Model the spectral reddening caused by continuing currents (Bozóki et al., 2025):

```python
import schupy as sp
import numpy as np

freq = np.arange(4.0, 35.0, 0.1)
s_lat, s_lon, s_int = [0.0], [0.0], [1.0e5]
obs_lat, obs_lon = 0.0, 60.0  # 60 degrees angular distance

# Impulsive lightning (Dirac delta, tau=0 ms)
spec_impulsive = sp.forward_tdte(s_lat, s_lon, s_int, obs_lat, obs_lon, freq, tau=0.0)

# Lightning with continuing current (tau=20 ms)
spec_cc20 = sp.forward_tdte(s_lat, s_lon, s_int, obs_lat, obs_lon, freq, tau=0.020)

# Lightning with long continuing current (tau=50 ms)
spec_cc50 = sp.forward_tdte(s_lat, s_lon, s_int, obs_lat, obs_lon, freq, tau=0.050)
```

---

### Example 4: High-Precision Hypergeometric Model

Use `forward_hyper` for exact analytical evaluation without series truncation errors (Prácser et al., 2021):

```python
import schupy as sp
import numpy as np

freq = np.arange(4.0, 35.0, 0.1)

# Exactly equivalent to forward_tdte, but solved via Gauss hypergeometric function 2F1:
exact_spectrum = sp.forward_hyper(
    s_lat=[10.0],
    s_lon=[10.0],
    s_int=[1.0e5],
    m_lat=47.6,
    m_lon=16.7,
    freq=freq,
    h="mushtak"
)

# Return only E_z array directly if preferred:
Er_only = sp.forward_hyper(
    s_lat=[10.0], s_lon=[10.0], s_int=[1.0e5],
    m_lat=47.6, m_lon=16.7,
    freq=freq,
    ret="E_z"
)
```

---

### Example 5: Axisymmetric North-Pole Geometry

When studying idealized source-observer distance $\theta$ (colatitude in degrees):

```python
import schupy as sp
import numpy as np

# Observer placed 60 degrees away from the pole
theta_distance_deg = 60.0
source_intensity = 1.0e5
freq = np.arange(4.0, 35.0, 0.1)

# Fast polar calculation
polar_spec = sp.forward_tdte_pole(
    theta=theta_distance_deg,
    s_int=source_intensity,
    freq=freq
)

# Note: B_NS is identically 0.0 by symmetry, B_EW is the non-zero azimuthal field
print("B_NS is zero:", np.all(polar_spec.B_NS == 0.0))  # True
print("Max B_EW:", np.max(polar_spec.B_EW))             # > 0
```

---

### Example 6: Plotting Results with Matplotlib

Since `schupy` is a pure computation library, you can easily visualize results with `matplotlib`:

```python
import schupy as sp
import numpy as np
import matplotlib.pyplot as plt

freq = np.arange(4.0, 35.0, 0.1)
spec = sp.forward_tdte(s_lat=[10.0], s_lon=[10.0], s_int=[1.0e5], m_lat=47.6, m_lon=16.7, freq=freq)

fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)

# 1. Electric field E_z
axes[0].plot(spec.freq, spec.E_z, color="navy", lw=1.5)
axes[0].set_ylabel(r"$E_z\ [\mathrm{mV^2 / m^2 / Hz}]$")
axes[0].grid(True, linestyle="--", alpha=0.6)
axes[0].set_title("Schumann Resonance Spectra")

# 2. Magnetic field B_NS (North-South coil)
axes[1].plot(spec.freq, spec.B_NS, color="darkgreen", lw=1.5)
axes[1].set_ylabel(r"$B_{NS}\ [\mathrm{pT^2 / Hz}]$")
axes[1].grid(True, linestyle="--", alpha=0.6)

# 3. Magnetic field B_EW (East-West coil)
axes[2].plot(spec.freq, spec.B_EW, color="crimson", lw=1.5)
axes[2].set_ylabel(r"$B_{EW}\ [\mathrm{pT^2 / Hz}]$")
axes[2].set_xlabel("Frequency [Hz]")
axes[2].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()
```

---

## 5. Input Parameters & Output Cheat Sheet

### Common Function Arguments

| Parameter | Type | Default | Description |
|---|---|---|---|
| `s_lat` | `float` or `list[float]` | *Required* | Source latitude(s) in degrees ($-90^\circ$ to $+90^\circ$). |
| `s_lon` | `float` or `list[float]` | *Required* | Source longitude(s) in degrees ($-180^\circ$ to $+180^\circ$). |
| `s_int` | `float` or `list[float]` | *Required* | Source intensity in $\text{C}^2\,\text{km}^2/\text{s}$ (charge moment change squared). |
| `m_lat` | `float` | *Required* | Observer latitude in degrees. |
| `m_lon` | `float` | *Required* | Observer longitude in degrees. |
| `freq` | `array_like` | *Required* | Evaluation frequencies in Hz (e.g. `np.arange(4, 35, 0.1)`). |
| `theta` | `float` | *Required (pole)* | Colatitude / angular distance from North Pole in degrees. |
| `n_max` | `int` | `5000` | Maximum order of Legendre polynomials summed in `forward_tdte`. |
| `h` | `str` | `"mushtak"` | Height calculation model (`"mushtak"` or `"kulak"`). |
| `tau` | `float` | `0.0` | Continuing current decay time constant $\tau$ in seconds. |
| `ret` | `str` | `"all"` | Return selector: `"all"`, `"er"`, `"b_ns"`, or `"b_ew"`. |

### `SRSpectrum` Return Object

When `ret="all"` (default), functions return an `SRSpectrum` instance:
- **`spectrum.freq`**: Evaluation frequency array [Hz].
- **`spectrum.E_z`**: Vertical electric field power spectral density [$\text{mV}^2/\text{m}^2/\text{Hz}$].
- **`spectrum.B_NS`**: North–South coil horizontal magnetic field PSD [$\text{pT}^2/\text{Hz}$].
- **`spectrum.B_EW`**: East–West coil horizontal magnetic field PSD [$\text{pT}^2/\text{Hz}$].
- Unpacking support: `E_z, B_NS, B_EW = spectrum` or indexing `spectrum[0]`, `spectrum[1]`, `spectrum[2]`.

---

## 6. References

1. **Bozóki, T., Prácser, E., Sátori, G., Dálya, G., Kapás, K., & Takátsy, J. (2019).** *Modeling Schumann resonances with schupy.* Journal of Atmospheric and Solar-Terrestrial Physics, 196, 105144. [doi:10.1016/j.jastp.2019.105144](https://doi.org/10.1016/j.jastp.2019.105144)
2. **Bozóki, T., Mlynarczyk, J., Prácser, E., Kulak, A., Sátori, G., Füllekrug, M., & Williams, E. (2025).** *Modeling the Global Electromagnetic Resonance Field Produced by Lightning Discharges With a Continuing Current.* Journal of Geophysical Research: Atmospheres, 130, e2025JD043989. [doi:10.1029/2025JD043989](https://doi.org/10.1029/2025JD043989)
3. **Prácser, E., Bozóki, T., Sátori, G., Takátsy, J., Williams, E., & Guha, A. (2021).** *Two Approaches for Modeling ELF Wave Propagation in the Earth-Ionosphere Cavity With Day-Night Asymmetry.* IEEE Transactions on Antennas and Propagation, 69(7), 4093–4099. [doi:10.1109/TAP.2020.3044669](https://doi.org/10.1109/TAP.2020.3044669)
4. **Kulak, A., & Mlynarczyk, J. (2013).** *ELF Propagation Parameters for the Ground-Ionosphere Waveguide With Finite Ground Conductivity.* IEEE Transactions on Antennas and Propagation, 61(4), 2269–2275. [doi:10.1109/TAP.2012.2227445](https://doi.org/10.1109/TAP.2012.2227445)
5. **Mushtak, V. C., & Williams, E. R. (2002).** *ELF propagation parameters for uniform models of the Earth-ionosphere waveguide.* Journal of Atmospheric and Solar-Terrestrial Physics, 64, 1989–2001. [doi:10.1016/S1364-6826(02)00222-5](https://doi.org/10.1016/S1364-6826(02)00222-5)
