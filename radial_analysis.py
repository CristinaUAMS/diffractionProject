import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from config import OUT_DIR
from io_utils import centerAndMask

## Compute azimuthal average I(r), for each radius r avg intensity over all pixels
def radialProfile(I_masked: np.ndarray, r: np.ndarray) -> np.ndarray:
    # use integer radii as bins
    r_int = r[I_masked > 0].astype(int)
    vals = I_masked[I_masked > 0]
    r_max = int(r_int.max())
    num = np.bincount(r_int, weights=vals, minlength=r_max+1)
    den = np.bincount(r_int, minlength=r_max+1)
    prof = num / (den + 1e-8)
    return prof

## profiles: dict[name -> radialProfile array], save log-scale radial intensity plot
def plotRadialProfiles(profiles: dict) -> None:
    plt.figure(figsize=(6,4))
    for name, prof in profiles.items():
        r_axis = np.arange(len(prof))
        plt.plot(r_axis, prof, label=name)
    plt.yscale("log")
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Mean intensity (a.u.)")
    plt.title("Radial intensity profiles")
    plt.legend()
    plt.tight_layout()
    out_path = OUT_DIR / "radial_profiles.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[radial] Saved radial profile plot to {out_path}")

## wrapper: mask core, compute profile and save per-image radial plot
def analyzeRadial(I: np.ndarray, core_radius: int, name: str) -> Tuple[np.ndarray, np.ndarray]:
    I_masked, r, center = centerAndMask(I, core_radius)
    prof = radialProfile(I_masked, r)

    # individual plot
    r_axis = np.arange(len(prof))
    plt.figure(figsize=(5,4))
    plt.plot(r_axis, prof)
    plt.yscale("log")
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Mean intensity (a.u.)")
    plt.title(f"Radial profile – {name}")
    plt.tight_layout()
    out_path = OUT_DIR / f"radial_{name}.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[radial] Saved {name} radial plot to {out_path}")

    return I_masked, prof    