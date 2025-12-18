import cv2
import numpy as np

from pathlib import Path
from typing import Tuple
from config import TARGET_SIZE

##load color image and convert to grayscale float32 [0,1]
def loadToGray(path: Path) -> np.ndarray:
    img = cv2.imread(str(Path))
    if img is None:
        raise FileNotFoundError(f"could not find image {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray -= gray.min()
    gray /= gray.max() + 1e-8
    return gray

## Find brightest pixel, compute radial distances and mask out saturated bits
## Return I_masked: image with central disk set to zero
##                  r: radial distance array (same shape as I)
##                  (cx, cy): center coordinates

def centerAndMask(I: np.ndarray, core_radius: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int]]:
    H, W = I.shape
    cy, cx = np.unravel_index(np.argmax(I), I.shape)
    Y, X = np.indices(I.shape)
    r = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = r > core_radius
    I_masked = I * mask
    return I_masked, r, (cx, cy)
