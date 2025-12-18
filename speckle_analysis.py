import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate2d
from config import OUT_DIR

## Compute speckle contrast C = sigma/mean in non-overlapping blocks with return map
def speckleContrastMap(I: np.ndarray, block: int = 24, min_mean: float = 0.02) -> np.ndarray:
    H, W = I.shape
    Cmap = np.zeros_like(I, dtype=np.float32)
    for y in range(0, H - block, block):
        for x in range(0, W - block, block):
            patch = I[y:y+block, x:x+block]
            m = patch.mean()
            if m > min_mean:
                Cmap[y:y+block, x:x+block] = patch.std() / (m + 1e-8)
    return Cmap

## plot speckle contast map and average C over valid pixels
def plotSpeckleContrast(Cmap: np.ndarray, name: str) -> float:
    valid = Cmap[Cmap > 0]
    mean_C = float(valid.mean()) if valid.size > 0 else float("nan")

    plt.figure(figsize=(5,4))
    plt.imshow(Cmap, cmap="viridis", vmin=0, vmax=1.2)
    plt.colorbar(label="C = σ/μ")
    plt.axis("off")
    plt.title(f"Local speckle contrast – {name}\nmean C ≈ {mean_C:.2f}")
    plt.tight_layout()
    out_path = OUT_DIR / f"speckleC_{name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[speckle] Saved {name} speckle contrast map to {out_path}")
    return mean_C

## estimate speckle grain size from the fwhm of the intensity autocorrelation
def speckleGrainSize(I: np.ndarray, crop_size: int = 256, name: str = "") -> float:
    H, W = I.shape
    cy, cx = H // 2, W // 2
    half = crop_size // 2
    y0 = max(0, cy - half)
    y1 = min(H, cy + half)
    x0 = max(0, cx - half)
    x1 = min(W, cx + half)
    crop = I[y0:y1, x0:x1].copy()

    crop -= crop.mean()
    R = correlate2d(crop, crop, mode="full", boundary="fill")
    R /= R.max() + 1e-8

    Hc, Wc = R.shape
    mid_y = Hc // 2
    line_x = R[mid_y, :]
    x_axis = np.arange(len(line_x)) - len(line_x)//2

    halfmax = 0.5
    above = np.where(line_x >= halfmax)[0]
    if len(above) > 1:
        fwhm_px = (above[-1] - above[0])
    else:
        fwhm_px = float("nan")

    plt.figure(figsize=(5,4))
    plt.plot(x_axis, line_x)
    plt.axhline(halfmax, ls="--", color="red")
    plt.xlabel("Lag (pixels)")
    plt.ylabel("Normalized autocorrelation")
    plt.title(f"Autocorrelation along x – {name}\nFWHM ≈ {fwhm_px:.1f} px")
    plt.tight_layout()
    out_path = OUT_DIR / f"speckleAC_{name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[speckle] Saved {name} autocorrelation plot to {out_path}")
    return float(fwhm_px)