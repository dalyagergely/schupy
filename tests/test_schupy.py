import sys
from pathlib import Path
import numpy as np

# Ensure local schupy is in path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

import schupy as sp

print(f"Schupy version: {sp.__version__}")

# -----------------------------------------------------------------------------
# Test 1: Basic forward calculation & Peak Detection
# -----------------------------------------------------------------------------
print("\n--- Test 1: Basic forward calculation & Peak Detection ---")
freq = np.round(np.arange(4.0, 35.0, 0.1), 2)
source_lat = [10.0]
source_lon = [10.0]
source_int = [1.0e5]  # C^2 km^2 / s
obs_lat = 47.6
obs_lon = 16.7

spec_tdte = sp.forward_tdte(source_lat, source_lon, source_int, obs_lat, obs_lon, freq, n_max=10000, h="mushtak")
print(f"Result type: {type(spec_tdte)}")
print(f"E_z shape: {spec_tdte.E_z.shape}, max E_z: {np.max(spec_tdte.E_z):.4e} mV^2/m^2/Hz")
print(f"B_NS shape: {spec_tdte.B_NS.shape}, max B_NS: {np.max(spec_tdte.B_NS):.4e} pT^2/Hz")
print(f"B_EW shape: {spec_tdte.B_EW.shape}, max B_EW: {np.max(spec_tdte.B_EW):.4e} pT^2/Hz")

# Verify alias properties: E_Z, Ez, Er
assert np.array_equal(spec_tdte.E_z, spec_tdte.E_Z)
assert np.array_equal(spec_tdte.E_z, spec_tdte.Ez)
assert np.array_equal(spec_tdte.E_z, spec_tdte.Er)

# Find peaks in E_z
peak_indices = []
for i in range(1, len(freq) - 1):
    if spec_tdte.E_z[i] > spec_tdte.E_z[i-1] and spec_tdte.E_z[i] > spec_tdte.E_z[i+1]:
        peak_indices.append(i)

peak_freqs = freq[peak_indices]
print(f"Detected SR peak frequencies in E_z: {peak_freqs[:4]} Hz")
assert len(peak_freqs) >= 3, "Failed to detect at least 3 SR modes"
assert 7.0 <= peak_freqs[0] <= 8.5, f"1st SR mode unexpected: {peak_freqs[0]}"
assert 13.0 <= peak_freqs[1] <= 15.0, f"2nd SR mode unexpected: {peak_freqs[1]}"
print("Test 1 PASSED: Resonance peaks match physical expectations.")

# -----------------------------------------------------------------------------
# Test 2: Consistency between forward_tdte (Legendre) and forward_hyper (Hypergeometric)
# -----------------------------------------------------------------------------
print("\n--- Test 2: Consistency between Legendre and Hypergeometric methods ---")
spec_hyper = sp.forward_hyper(source_lat, source_lon, source_int, obs_lat, obs_lon, freq, h="mushtak")

# Calculate relative differences
rel_diff_Ez = np.abs(spec_tdte.E_z - spec_hyper.E_z) / np.maximum(spec_tdte.E_z, spec_hyper.E_z)
rel_diff_B_EW = np.abs(spec_tdte.B_EW - spec_hyper.B_EW) / np.maximum(spec_tdte.B_EW, spec_hyper.B_EW)
rel_diff_B_NS = np.abs(spec_tdte.B_NS - spec_hyper.B_NS) / np.maximum(spec_tdte.B_NS, spec_hyper.B_NS)

print(f"Mean relative difference in E_z: {np.mean(rel_diff_Ez) * 100:.4f}%")
print(f"Max relative difference in E_z: {np.max(rel_diff_Ez) * 100:.4f}%")
print(f"Mean relative difference in B_EW: {np.mean(rel_diff_B_EW) * 100:.4f}%")
print(f"Mean relative difference in B_NS: {np.mean(rel_diff_B_NS) * 100:.4f}%")

assert np.mean(rel_diff_Ez) < 0.001, "Legendre and Hypergeometric E_z do not match within 0.1%"
assert np.mean(rel_diff_B_EW) < 0.02, "Legendre and Hypergeometric B_EW do not match within 2%"
assert np.mean(rel_diff_B_NS) < 0.02, "Legendre and Hypergeometric B_NS do not match within 2%"
print("Test 2 PASSED: Legendre (n=10000) and Hypergeometric match across all components.")

# -----------------------------------------------------------------------------
# Test 3: Polar symmetry and North-Pole simplified functions
# -----------------------------------------------------------------------------
print("\n--- Test 3: Polar source forward models (forward_tdte_pole & forward_hyper_pole) ---")
theta_deg = 42.4  # colatitude of observer (m_lat = 90 - 42.4 = 47.6)
s_int_val = 1.0e5

spec_pole_gen = sp.forward_tdte([90.0], [0.0], [s_int_val], 90.0 - theta_deg, 0.0, freq, n_max=10000)
spec_pole_simp = sp.forward_tdte_pole(theta_deg, s_int_val, freq, n_max=10000)
spec_hyper_pole = sp.forward_hyper_pole(theta_deg, s_int_val, freq)

print(f"General pole B_NS max: {np.max(spec_pole_gen.B_NS):.4e} (identically 0)")
print(f"Simplified pole B_NS max: {np.max(spec_pole_simp.B_NS):.4e} (identically 0)")
print(f"General pole B_EW max: {np.max(spec_pole_gen.B_EW):.4e} (non-zero)")
print(f"Simplified pole B_EW max: {np.max(spec_pole_simp.B_EW):.4e} (non-zero)")

assert np.max(spec_pole_gen.B_NS) < 1e-15 * np.max(spec_pole_gen.B_EW)
assert np.all(spec_pole_simp.B_NS == 0.0)

rel_diff_pole_Ez = np.abs(spec_pole_gen.E_z - spec_pole_simp.E_z) / spec_pole_gen.E_z
rel_diff_pole_BEW = np.abs(spec_pole_gen.B_EW - spec_pole_simp.B_EW) / spec_pole_gen.B_EW
print(f"Pole E_z relative difference (simp vs general): {np.mean(rel_diff_pole_Ez)*100:.4f}%")
print(f"Pole B_EW relative difference (simp vs general): {np.mean(rel_diff_pole_BEW)*100:.4f}%")
assert np.mean(rel_diff_pole_Ez) < 0.001
assert np.mean(rel_diff_pole_BEW) < 0.001

rel_diff_hyper_pole_Ez = np.abs(spec_hyper_pole.E_z - spec_pole_simp.E_z) / spec_pole_simp.E_z
print(f"Hyper pole E_z relative difference: {np.mean(rel_diff_hyper_pole_Ez)*100:.4f}%")
assert np.mean(rel_diff_hyper_pole_Ez) < 0.001
print("Test 3 PASSED: Polar source formulation verified.")

# -----------------------------------------------------------------------------
# Test 4: Finite decay time (tau) effect (Bozóki et al., 2025b)
# -----------------------------------------------------------------------------
print("\n--- Test 4: Finite decay time (tau) verification ---")
spec_tau0 = sp.forward_tdte(source_lat, source_lon, source_int, obs_lat, obs_lon, freq, tau=0.0)
spec_tau50 = sp.forward_tdte(source_lat, source_lon, source_int, obs_lat, obs_lon, freq, tau=0.050)

idx_8 = np.argmin(np.abs(freq - 8.0))
idx_20 = np.argmin(np.abs(freq - 20.0))

ratio_8Hz_tau0 = spec_tau0.E_z[idx_8] / spec_tau0.E_z[idx_20]
ratio_8Hz_tau50 = spec_tau50.E_z[idx_8] / spec_tau50.E_z[idx_20]

print(f"Ratio E_z(8Hz)/E_z(20Hz) for tau=0ms:  {ratio_8Hz_tau0:.3f}")
print(f"Ratio E_z(8Hz)/E_z(20Hz) for tau=50ms: {ratio_8Hz_tau50:.3f}")
assert ratio_8Hz_tau50 > ratio_8Hz_tau0, "Decay time tau did not increase spectral redness"
print("Test 4 PASSED: Finite decay time correctly reddens the spectrum.")

# -----------------------------------------------------------------------------
# Test 5: Kulak vs Mushtak Height Models
# -----------------------------------------------------------------------------
print("\n--- Test 5: Kulak & Mlynarczyk (2013) height model ---")
spec_kulak = sp.forward_tdte(source_lat, source_lon, source_int, obs_lat, obs_lon, freq, h="kulak")
print(f"Kulak max E_z: {np.max(spec_kulak.E_z):.4e} mV^2/m^2/Hz")
assert np.max(spec_kulak.E_z) > 0.0
print("Test 5 PASSED: Kulak height model functions correctly.")

# -----------------------------------------------------------------------------
# Test 6: Input immutability & unpacking test
# -----------------------------------------------------------------------------
print("\n--- Test 6: Caller input list immutability & Unpacking ---")
orig_s_int = [1.0e5, 2.0e5]
s_int_copy = list(orig_s_int)
_ = sp.forward_tdte([0.0, 10.0], [0.0, 10.0], s_int_copy, obs_lat, obs_lon, freq)
assert s_int_copy == orig_s_int, "s_int was mutated in place!"

# Sequence unpacking test
E_z, B_NS, B_EW = spec_tdte
assert np.array_equal(E_z, spec_tdte.E_z)
assert np.array_equal(B_NS, spec_tdte.B_NS)
assert np.array_equal(B_EW, spec_tdte.B_EW)

# ret parameter test
Ez_only = sp.forward_tdte(source_lat, source_lon, source_int, obs_lat, obs_lon, freq, ret="E_z")
assert np.array_equal(Ez_only, spec_tdte.E_z)
print("Test 6 PASSED: Immutability, unpacking, and return selectors work as intended.")

print("\n=======================================================")
print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY! (6/6)")
print("=======================================================")
