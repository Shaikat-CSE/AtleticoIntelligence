from typing import Any, Dict, Tuple

import cv2
import numpy as np


_COLOR_PALETTE_BGR: Dict[str, Tuple[int, int, int]] = {
    "White": (245, 245, 245),
    "Light Gray": (200, 200, 200),
    "Gray": (128, 128, 128),
    "Black": (28, 28, 28),
    "Beige": (170, 200, 225),
    "Brown": (60, 90, 145),
    "Red": (40, 40, 210),
    "Maroon": (45, 45, 128),
    "Pink": (170, 150, 235),
    "Orange": (40, 140, 235),
    "Yellow": (40, 225, 235),
    "Gold": (55, 180, 215),
    "Green": (55, 150, 55),
    "Lime": (60, 220, 120),
    "Teal": (120, 140, 45),
    "Cyan": (220, 220, 45),
    "Sky Blue": (210, 170, 90),
    "Blue": (200, 90, 45),
    "Navy": (120, 60, 25),
    "Purple": (140, 70, 130),
}


def _bgr_to_lab(bgr: Tuple[int, int, int]) -> np.ndarray:
    swatch = np.uint8([[list(bgr)]])
    return cv2.cvtColor(swatch, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def _bgr_to_hsv(bgr: Tuple[int, int, int]) -> np.ndarray:
    swatch = np.uint8([[list(bgr)]])
    return cv2.cvtColor(swatch, cv2.COLOR_BGR2HSV)[0, 0].astype(np.float32)


_COLOR_PALETTE_LAB: Dict[str, np.ndarray] = {
    name: _bgr_to_lab(bgr)
    for name, bgr in _COLOR_PALETTE_BGR.items()
}

_COLOR_PALETTE_HSV: Dict[str, np.ndarray] = {
    name: _bgr_to_hsv(bgr)
    for name, bgr in _COLOR_PALETTE_BGR.items()
}


def _lab_to_bgr(lab: np.ndarray) -> np.ndarray:
    swatch = np.uint8([[np.clip(lab, 0, 255).astype(np.uint8)]])
    return cv2.cvtColor(swatch, cv2.COLOR_LAB2BGR)[0, 0].astype(np.float32)


def _crop_jersey_roi(player_bbox: Any, image: np.ndarray) -> np.ndarray:
    x1, y1 = int(player_bbox.x1), int(player_bbox.y1)
    x2, y2 = int(player_bbox.x2), int(player_bbox.y2)
    x1, x2 = max(0, x1), min(image.shape[1], x2)
    y1, y2 = max(0, y1), min(image.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)

    roi_height = y2 - y1
    roi_width = x2 - x1

    jersey_top = int(y1 + roi_height * 0.18)
    jersey_bottom = int(y1 + roi_height * 0.54)
    jersey_left = int(x1 + roi_width * 0.18)
    jersey_right = int(x2 - roi_width * 0.18)

    if jersey_right <= jersey_left or jersey_bottom <= jersey_top:
        return np.empty((0, 0, 3), dtype=np.uint8)

    return image[jersey_top:jersey_bottom, jersey_left:jersey_right]


def _dominant_color_from_roi_bgr(roi: np.ndarray) -> np.ndarray:
    if roi.size == 0:
        return np.array([128, 128, 128], dtype=np.float32)

    small_roi = roi
    if roi.shape[0] > 36 or roi.shape[1] > 36:
        small_roi = cv2.resize(roi, (36, 36), interpolation=cv2.INTER_AREA)

    pixels_bgr = small_roi.reshape(-1, 3).astype(np.float32)
    pixels_hsv = cv2.cvtColor(small_roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

    hue = pixels_hsv[:, 0]
    sat = pixels_hsv[:, 1]
    val = pixels_hsv[:, 2]

    not_grass = ~((hue >= 35) & (hue <= 95) & (sat >= 55) & (val >= 35))
    not_extreme_shadow = val >= 18
    usable_mask = not_grass & not_extreme_shadow

    if int(np.count_nonzero(usable_mask)) < 24:
        usable_mask = not_extreme_shadow
    if int(np.count_nonzero(usable_mask)) < 12:
        usable_mask = np.ones(len(pixels_bgr), dtype=bool)

    usable_pixels = pixels_bgr[usable_mask]
    if len(usable_pixels) == 0:
        usable_pixels = pixels_bgr

    if len(usable_pixels) > 400:
        step = max(1, len(usable_pixels) // 400)
        usable_pixels = usable_pixels[::step][:400]

    if len(usable_pixels) < 24:
        return np.median(usable_pixels, axis=0).astype(np.float32)

    lab_pixels = cv2.cvtColor(
        np.uint8(np.clip(usable_pixels.reshape(-1, 1, 3), 0, 255)),
        cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)

    cluster_count = 3 if len(lab_pixels) >= 120 else 2
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 18, 0.5)
    _, labels, centers = cv2.kmeans(
        lab_pixels,
        cluster_count,
        None,
        criteria,
        4,
        cv2.KMEANS_PP_CENTERS
    )

    best_idx = 0
    best_score = -1.0
    for idx in range(cluster_count):
        mask = labels.ravel() == idx
        count = int(np.count_nonzero(mask))
        center_bgr = _lab_to_bgr(centers[idx])
        center_hsv = cv2.cvtColor(
            np.uint8([[np.clip(center_bgr, 0, 255).astype(np.uint8)]]),
            cv2.COLOR_BGR2HSV
        )[0, 0].astype(np.float32)

        score = float(count)
        if center_hsv[1] >= 40:
            score *= 1.08
        if 35 <= center_hsv[0] <= 95 and center_hsv[1] >= 55:
            score *= 0.65
        if center_hsv[2] <= 18 or center_hsv[2] >= 245:
            score *= 0.92

        if score > best_score:
            best_score = score
            best_idx = idx

    chosen_pixels = usable_pixels[labels.ravel() == best_idx]
    if len(chosen_pixels) == 0:
        chosen_pixels = usable_pixels

    center_bgr = _lab_to_bgr(centers[best_idx])
    median_bgr = np.median(chosen_pixels, axis=0).astype(np.float32)
    return (center_bgr * 0.65 + median_bgr * 0.35).astype(np.float32)


def extract_jersey_color_bgr(player_bbox: Any, image: np.ndarray) -> np.ndarray:
    roi = _crop_jersey_roi(player_bbox, image)
    return _dominant_color_from_roi_bgr(roi)


def _hue_distance(h1: float, h2: float) -> float:
    diff = abs(h1 - h2)
    return min(diff, 180.0 - diff)


def _neutral_name(val: int) -> str:
    if val >= 230:
        return "White"
    if val >= 180:
        return "Light Gray"
    if val >= 95:
        return "Gray"
    return "Black"


def _candidate_names_for_hsv(hue: int, sat: int, val: int) -> Tuple[str, ...]:
    if hue <= 7 or hue >= 173:
        if val < 115:
            return ("Maroon", "Red", "Brown", "Pink")
        if sat < 95 and val > 160:
            return ("Pink", "Red", "Orange", "Maroon")
        return ("Red", "Maroon", "Pink", "Orange")

    if hue <= 16:
        if val < 150:
            return ("Brown", "Orange", "Maroon", "Gold")
        return ("Orange", "Red", "Gold", "Brown", "Yellow")

    if hue <= 28:
        if sat < 95 and val > 155:
            return ("Beige", "Gold", "Yellow", "Orange")
        if val < 150:
            return ("Brown", "Gold", "Orange", "Maroon")
        return ("Gold", "Yellow", "Orange", "Beige", "Brown")

    if hue <= 36:
        if sat < 90 and val > 170:
            return ("Beige", "Yellow", "Gold", "Light Gray")
        return ("Yellow", "Gold", "Lime", "Orange", "Beige")

    if hue <= 50:
        if sat > 135 and val > 130:
            return ("Lime", "Green", "Yellow", "Teal")
        return ("Green", "Lime", "Yellow", "Teal")

    if hue <= 85:
        return ("Green", "Teal", "Lime", "Cyan")

    if hue <= 100:
        return ("Teal", "Cyan", "Green", "Sky Blue")

    if hue <= 114:
        return ("Cyan", "Sky Blue", "Teal", "Blue")

    if hue <= 130:
        if val < 105:
            return ("Navy", "Blue", "Sky Blue", "Purple")
        return ("Sky Blue", "Blue", "Cyan", "Navy")

    if hue <= 146:
        if val < 115:
            return ("Navy", "Blue", "Purple", "Sky Blue")
        return ("Blue", "Sky Blue", "Navy", "Purple")

    if hue <= 166:
        if val < 120:
            return ("Purple", "Navy", "Maroon", "Blue")
        return ("Purple", "Pink", "Blue", "Maroon")

    return ("Pink", "Purple", "Red", "Maroon")


def _weighted_color_score(name: str, lab: np.ndarray, hue: int, sat: int, val: int) -> float:
    palette_lab = _COLOR_PALETTE_LAB[name]
    palette_hsv = _COLOR_PALETTE_HSV[name]

    lab_distance = float(np.linalg.norm(lab - palette_lab))
    hue_distance = _hue_distance(float(hue), float(palette_hsv[0]))
    sat_distance = abs(float(sat) - float(palette_hsv[1]))
    val_distance = abs(float(val) - float(palette_hsv[2]))

    return (
        lab_distance * 0.045
        + hue_distance * 0.75
        + sat_distance * 0.010
        + val_distance * 0.006
    )


def get_color_name_from_bgr(bgr: Tuple[int, int, int]) -> str:
    bgr_int = tuple(int(max(0, min(255, channel))) for channel in bgr)
    swatch = np.uint8([[list(bgr_int)]])
    hsv = cv2.cvtColor(swatch, cv2.COLOR_BGR2HSV)[0, 0]
    hue, sat, val = int(hsv[0]), int(hsv[1]), int(hsv[2])

    # Handle achromatic colors first so muted kits don't fall through to raw RGB labels.
    if sat < 30:
        return _neutral_name(val)

    # Warm low-saturation kits tend to be sand/tan/cream rather than bright yellow.
    if 8 <= hue <= 30 and sat < 90 and val > 190:
        if sat < 75:
            return "Beige"

    # Very dark kits are better classified by hue family before nearest-palette matching.
    if val <= 55:
        if hue <= 10 or hue >= 170:
            return "Maroon"
        if 110 <= hue <= 150:
            return "Navy"
        if 35 <= hue <= 95:
            return "Green"
        if 10 <= hue <= 28:
            return "Brown"
        return "Black"

    if 10 <= hue <= 24 and val < 160 and sat < 170:
        return "Brown"

    if 105 <= hue <= 130 and sat > 135 and val < 145:
        return "Navy"

    lab = cv2.cvtColor(swatch, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    candidate_names = _candidate_names_for_hsv(hue, sat, val)

    best_name = candidate_names[0]
    best_distance = float("inf")
    for name in candidate_names:
        distance = _weighted_color_score(name, lab, hue, sat, val)
        if distance < best_distance:
            best_name = name
            best_distance = distance

    return best_name
