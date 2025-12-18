import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import windows
from config import OUT_DIR

## Compute log magnitude FFT of masked intensity image, return log10 magnitude array
def fftFingerprint(I_masked: np.ndarray, name: str) -> np.ndarray:
    H, W = I_masked.shape
    # Hann window to reduce edge artifacts
    win_x = windows.hann(W)
    win_y = windows.hann(H)
    win2d = np.outer(win_y, win_x)
    Iw = I_masked * win2d

    F = np.fft.fftshift(np.fft.fft2(Iw))
    Fmag = np.abs(F)
    Flog = np.log10(Fmag + 1e-3)

    plt.figure(figsize=(5,5))
    plt.imshow(Flog, cmap="gray")
    plt.axis("off")
    plt.title(f"log |FFT(I)| – {name}")
    plt.tight_layout()
    out_path = OUT_DIR / f"fft_{name}.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[fft] Saved {name} FFT fingerprint to {out_path}")

    return Flog