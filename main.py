import matplotlib.pyplot as plt  
from config import IMAGE_FILES, CORE_RADII, OUT_DIR
from io_utils import loadToGray
from radial_analysis import analyzeRadial, plotRadialProfiles
from fft_analysis import fftFingerprint
from speckle_analysis import speckleContrastMap, plotSpeckleContrast, speckleGrainSize
from math_metrics import effectiveApertureFromSpeckle

def main():
    print(f"Outputs will be saved to: {OUT_DIR}")
    radial_profiles = {}
    speckle_summary = {}

    WAVELENGTH_M = 532e-9      # 532 nm green laser
Z_M = 1.0                  # 1 meter from aperture/lens to wall 
PIXEL_SIZE_M = 1e-4        # 0.1 mm per pixel on wall 
    for name, path in IMAGE_FILES.items():
        print(f"\n=== Processing {name} ({path.name}) ===")
        # 1. Load & crop
        I = loadToGray(path)

        # 2. Radial analysis
        core_radius = CORE_RADII[name]
        I_masked, prof = analyzeRadial(I, core_radius, name)
        radial_profiles[name] = prof

        # 3. FFT fingerprint
        _ = fftFingerprint(I_masked, name)

        # 4. Speckle contrast & grain size (use unmasked image; core saturation is fine
        # for our approximate stats, but you can switch to I_masked if you prefer)
        Cmap = speckleContrastMap(I, block=24)
        mean_C = plotSpeckleContrast(Cmap, name)
        fwhm_px = speckleGrainSize(I, crop_size=256, name=name)
        speckle_summary[name] = (mean_C, fwhm_px)

    # Combined radial plot
    plotRadialProfiles(radial_profiles)

    # Print a small numerical summary to console
    print("\n=== Speckle summary ===")
    for name, (C, fwhm) in speckle_summary.items():
        print(f"{name:15s} : mean C ≈ {C:5.2f}, speckle FWHM ≈ {fwhm:6.1f} px")

if __name__ == "__main__":
    main()