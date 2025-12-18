from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"
OUT_DIR.mkdir(exist_ok=True)

IMAGE_FILES = {
    "aperture_only": BASE_DIR/"images/noLensPinhole.jpg",
    "lens_convex": BASE_DIR/"images/LaserLens2.jpg",
    "lens_planar": BASE_DIR/"images/planeToLaser.jpg",
}