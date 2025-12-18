import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple
from config import OUT_DIR
from io_utils import centerAndMask
from math_metrics import encircledEnergy, halfPowerRadius


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


    # Encircled energy and half-power radius
    E = encircledEnergy(prof)
    R50 = halfPowerRadius(prof)

    # Plot encircled energy on same figure or separate
    fig, ax1 = plt.subplots(figsize=(5,4))
    r_axis = np.arange(len(prof))
    ax1.plot(r_axis, prof, label="I(r)")
    ax1.set_yscale("log")
    ax1.set_xlabel("Radius (pixels)")
    ax1.set_ylabel("Mean intensity (a.u.)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.plot(r_axis, E, color="C1", label="E(r)")
    ax2.axvline(R50, color="C1", ls="--", alpha=0.7)
    ax2.set_ylabel("Encircled energy", color="C1")
    ax2.set_ylim(0, 1.05)
    ax2.tick_params(axis="y", labelcolor="C1")

    plt.title(f"Radial profile & encircled energy – {name}\nR50 ≈ {R50:.1f} px")
    fig.tight_layout()
    out_path = OUT_DIR / f"radial_plus_energy_{name}.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"[radial] {name}: R50 ≈ {R50:.2f} pixels (saved {out_path})")

    return I_masked, prof    

