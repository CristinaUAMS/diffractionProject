# math_metrics.py
import numpy as np
## Given radial profile I[r] (r = 0,1,2,... in pixels),
##   compute normalized encircled energy:

##      E(R) = (2π ∫_0^R I(r) r dr) / (2π ∫_0^{∞} I(r) r dr)

##   Here we approximate the integral with a discrete sum:
##       E[k] ≈ sum_{r=0}^{k} I[r] * r
##       then normalize by the max.

def encircledEnergy(profile: np.ndarray) -> np.ndarray:

    r = np.arange(len(profile), dtype=float)
    # weight by r (area element in polar coords)
    weighted = profile * r
    cum = np.cumsum(weighted)
    E = cum / (cum[-1] + 1e-12)
    return E
## Return R_50: radius where encircled energy reaches 0.5.
def halfPowerRadius(profile: np.ndarray) -> float:

    E = encircledEnergy(profile)
    idx = np.argmax(E >= 0.5)
    return float(idx)
  
  
## A simple 'Airy-like' reference curve:
##     I(r) ∝ [2 J1(k r) / (k r)]^2
## This is not a fit, just a shape to overlay 
def airyLikeReference(r: np.ndarray, k: float = 0.02) -> np.ndarray:

    from scipy.special import j1  # Bessel J1

    kr = k * r
    # avoid division by zero at r=0
    eps = 1e-8
    num = 2 * j1(kr + eps)
    denom = (kr + eps)
    I = (num / denom) ** 2
    I /= I.max() + 1e-12
    return I


# Classic speckle relation (for near-plane wave illumination):
#     δx_speckle ≈ λ z / D_eff
# where:
#     δx_speckle: speckle grain size in meters (FWHM)
#     λ: wavelength
#     z: distance from scattering plane to observation plane
#     D_eff: effective aperture diameter
## Solve for D_eff:
#     D_eff ≈ λ z / δx_speckle
## Estimate δx_speckle from the FWHM in pixels times pixel size.
## Returns D_eff in meters.
def effectiveApertureFromSpeckle(
    fwhm_px: float,
    pixel_size_m: float,
    wavelength_m: float,
    z_m: float,
) -> float:

    if np.isnan(fwhm_px):
        return float("nan")
    delta_x = fwhm_px * pixel_size_m
    Deff = wavelength_m * z_m / (delta_x + 1e-12)
    return float(Deff)
