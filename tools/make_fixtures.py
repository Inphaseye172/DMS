#!/usr/bin/env python3
"""Generates deterministic test images under tests/fixtures/.

  scene_floor.png      640x360 perspective checkerboard floor + sky gradient (bottom = near)
  odd_333x197.png      odd dimensions
  tiny_1x1.png         degenerate size
  seq/frame_0000..0011 12-frame 320x180 sequence: a square moving in front of the floor
  scene_4k.png         3840x2160 version of the floor scene (for tiling / 4K stress); only with --4k

Requires Pillow + numpy. Run: python3 -I tools/make_fixtures.py [--4k]
"""
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests", "fixtures")


def floor_scene(w, h, square=None):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    horizon = h * 0.4
    img = np.zeros((h, w, 3), np.float64)
    # Sky: vertical gradient.
    t = np.clip(yy / horizon, 0, 1)
    sky = np.stack([0.45 + 0.3 * t, 0.6 + 0.25 * t, 0.95 + 0.0 * t], -1)
    # Floor: perspective checkerboard. z ~ 1/(y - horizon).
    dy = np.maximum(yy - horizon, 1e-3)
    z = (h * 0.6) / dy
    xw = (xx - w / 2) * z / (w * 0.5)
    checker = ((np.floor(xw * 2) + np.floor(z * 2)) % 2).astype(np.float64)
    shade = np.clip(1.2 / (0.4 + 0.15 * z), 0.2, 1.0)
    floor = np.stack([0.2 + 0.6 * checker, 0.2 + 0.5 * checker, 0.15 + 0.4 * checker], -1) * shade[..., None]
    mask = (yy > horizon)[..., None]
    img = np.where(mask, floor, sky)
    if square is not None:
        cx, cy, s = square
        m = (np.abs(xx - cx) < s) & (np.abs(yy - cy) < s * 1.4)
        img[m] = [0.85, 0.2, 0.15]
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def main():
    os.makedirs(os.path.join(ROOT, "seq"), exist_ok=True)
    Image.fromarray(floor_scene(640, 360)).save(os.path.join(ROOT, "scene_floor.png"), optimize=True)
    Image.fromarray(floor_scene(333, 197)).save(os.path.join(ROOT, "odd_333x197.png"), optimize=True)
    Image.fromarray(np.full((1, 1, 3), 128, np.uint8)).save(os.path.join(ROOT, "tiny_1x1.png"))
    for i in range(12):
        sq = (60 + i * 18, 120, 22)
        Image.fromarray(floor_scene(320, 180, sq)).save(os.path.join(ROOT, "seq", f"frame_{i:04d}.png"), optimize=True)
    if "--4k" in sys.argv:
        Image.fromarray(floor_scene(3840, 2160, (1900, 1500, 260))).save(os.path.join(ROOT, "scene_4k.png"), optimize=True)
    print("fixtures written to", os.path.abspath(ROOT))


if __name__ == "__main__":
    main()
