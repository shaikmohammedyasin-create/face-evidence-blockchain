"""
demo/generate_sample.py — Generate a synthetic test face portrait for zero-setup demo testing.
"""
from pathlib import Path
import numpy as np
import cv2

def create_sample_face(out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Create canvas 400x400
    img = np.full((400, 400, 3), 240, dtype=np.uint8)

    # Head (oval)
    cv2.ellipse(img, (200, 200), (90, 120), 0, 0, 360, (200, 180, 160), -1)
    # Eyes
    cv2.circle(img, (165, 175), 12, (255, 255, 255), -1)
    cv2.circle(img, (235, 175), 12, (255, 255, 255), -1)
    cv2.circle(img, (165, 175), 6, (60, 40, 30), -1)
    cv2.circle(img, (235, 175), 6, (60, 40, 30), -1)
    # Nose
    cv2.line(img, (200, 175), (195, 215), (160, 140, 120), 3)
    cv2.line(img, (195, 215), (205, 215), (160, 140, 120), 3)
    # Mouth
    cv2.ellipse(img, (200, 250), (35, 15), 0, 0, 180, (150, 70, 70), -1)
    # Hair
    cv2.ellipse(img, (200, 130), (100, 60), 0, 180, 360, (40, 30, 20), -1)

    cv2.imwrite(str(out_path), img)
    return out_path

if __name__ == "__main__":
    dest = Path(__file__).resolve().parents[1] / "data" / "input" / "sample_portrait.png"
    create_sample_face(dest)
    print(f"Generated sample portrait at: {dest}")
