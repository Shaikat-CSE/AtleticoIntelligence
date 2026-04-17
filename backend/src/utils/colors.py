from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class JerseyColorProfile:
    color_bgr: Tuple[int, int, int]
    color_name: str
    confidence: float
    warning: Optional[str]
    usable_pixel_ratio: float
    sample_count: int


@dataclass
class TeamColorProfile:
    color_bgr: Tuple[int, int, int]
    color_name: str
    confidence: float
    warning: Optional[str]
    player_count: int


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


def _to_bgr_tuple(color: np.ndarray) -> Tuple[int, int, int]:
    return tuple(int(np.clip(round(float(channel)), 0, 255)) for channel in color)


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


def _crop_relative_region(
    player_bbox: Any,
    image: np.ndarray,
    top_ratio: float,
    bottom_ratio: float,
    left_ratio: float,
    right_ratio: float
) -> np.ndarray:
    x1, y1 = int(player_bbox.x1), int(player_bbox.y1)
    x2, y2 = int(player_bbox.x2), int(player_bbox.y2)
    x1, x2 = max(0, x1), min(image.shape[1], x2)
    y1, y2 = max(0, y1), min(image.shape[0], y2)

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)

    width = x2 - x1
    height = y2 - y1

    left = int(round(x1 + width * left_ratio))
    right = int(round(x1 + width * right_ratio))
    top = int(round(y1 + height * top_ratio))
    bottom = int(round(y1 + height * bottom_ratio))

    left = max(x1, min(left, x2))
    right = max(x1, min(right, x2))
    top = max(y1, min(top, y2))
    bottom = max(y1, min(bottom, y2))

    if right - left < 4 or bottom - top < 4:
        return np.empty((0, 0, 3), dtype=np.uint8)

    return image[top:bottom, left:right]


def _sample_jersey_rois(player_bbox: Any, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
    windows = [
        (0.18, 0.40, 0.24, 0.76, 1.25),
        (0.24, 0.56, 0.20, 0.80, 1.10),
        (0.22, 0.50, 0.34, 0.66, 1.35),
        (0.34, 0.64, 0.22, 0.78, 0.85),
    ]

    sampled_rois: List[Tuple[np.ndarray, float]] = []
    for top, bottom, left, right, weight in windows:
        roi = _crop_relative_region(player_bbox, image, top, bottom, left, right)
        if roi.size == 0:
            continue
        sampled_rois.append((roi, weight))

    if sampled_rois:
        return sampled_rois

    fallback_roi = _crop_jersey_roi(player_bbox, image)
    if fallback_roi.size == 0:
        return []

    return [(fallback_roi, 1.0)]


def _prepare_jersey_pixels(roi: np.ndarray) -> Tuple[np.ndarray, float]:
    if roi.size == 0:
        return np.empty((0, 3), dtype=np.float32), 0.0

    small_roi = roi
    if roi.shape[0] > 40 or roi.shape[1] > 40:
        small_roi = cv2.resize(roi, (40, 40), interpolation=cv2.INTER_AREA)

    pixels_bgr = small_roi.reshape(-1, 3).astype(np.float32)
    pixels_hsv = cv2.cvtColor(small_roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

    hue = pixels_hsv[:, 0]
    sat = pixels_hsv[:, 1]
    val = pixels_hsv[:, 2]

    not_grass = ~((hue >= 32) & (hue <= 96) & (sat >= 35) & (val >= 28))
    not_extreme_shadow = val >= 18
    usable_mask = not_grass & not_extreme_shadow

    usable_count = int(np.count_nonzero(usable_mask))
    if usable_count < 24:
        usable_mask = not_extreme_shadow
        usable_count = int(np.count_nonzero(usable_mask))
    if usable_count < 12:
        usable_mask = np.ones(len(pixels_bgr), dtype=bool)
        usable_count = len(pixels_bgr)

    usable_pixels = pixels_bgr[usable_mask]
    usable_ratio = usable_count / max(len(pixels_bgr), 1)

    if len(usable_pixels) > 450:
        step = max(1, len(usable_pixels) // 450)
        usable_pixels = usable_pixels[::step][:450]

    return usable_pixels, float(usable_ratio)


def _dominant_color_and_quality_from_pixels(
    usable_pixels: np.ndarray,
    usable_ratio: float
) -> Tuple[np.ndarray, float]:
    if usable_pixels.size == 0:
        return np.array([128, 128, 128], dtype=np.float32), 0.05

    if len(usable_pixels) < 24:
        median_bgr = np.median(usable_pixels, axis=0).astype(np.float32)
        quality = float(np.clip(0.25 + usable_ratio * 0.45, 0.08, 0.55))
        return median_bgr, quality

    lab_pixels = cv2.cvtColor(
        np.uint8(np.clip(usable_pixels.reshape(-1, 1, 3), 0, 255)),
        cv2.COLOR_BGR2LAB
    ).reshape(-1, 3).astype(np.float32)

    if len(lab_pixels) >= 160:
        cluster_count = 3
    elif len(lab_pixels) >= 50:
        cluster_count = 2
    else:
        cluster_count = 1

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
    best_spread = 40.0

    for idx in range(cluster_count):
        cluster_mask = labels.ravel() == idx
        count = int(np.count_nonzero(cluster_mask))
        if count == 0:
            continue

        cluster_lab = lab_pixels[cluster_mask]
        center_bgr = _lab_to_bgr(centers[idx])
        center_hsv = cv2.cvtColor(
            np.uint8([[np.clip(center_bgr, 0, 255).astype(np.uint8)]]),
            cv2.COLOR_BGR2HSV
        )[0, 0].astype(np.float32)

        spread = float(np.mean(np.linalg.norm(cluster_lab - centers[idx], axis=1)))
        score = float(count)
        if center_hsv[1] >= 35:
            score *= 1.05
        if 32 <= center_hsv[0] <= 96 and center_hsv[1] >= 45:
            score *= 0.58
        if center_hsv[2] <= 18:
            score *= 0.90
        score *= max(0.35, 1.15 - spread / 42.0)

        if score > best_score:
            best_score = score
            best_idx = idx
            best_spread = spread

    chosen_mask = labels.ravel() == best_idx
    chosen_pixels = usable_pixels[chosen_mask]
    if len(chosen_pixels) == 0:
        chosen_pixels = usable_pixels

    center_bgr = _lab_to_bgr(centers[best_idx])
    median_bgr = np.median(chosen_pixels, axis=0).astype(np.float32)
    blended_bgr = (center_bgr * 0.60 + median_bgr * 0.40).astype(np.float32)

    purity = len(chosen_pixels) / max(len(usable_pixels), 1)
    spread_score = max(0.0, 1.0 - best_spread / 36.0)
    quality = float(np.clip(
        0.45 * purity + 0.30 * usable_ratio + 0.25 * spread_score,
        0.08,
        0.98
    ))
    return blended_bgr, quality


def _dominant_color_from_roi_bgr(roi: np.ndarray) -> np.ndarray:
    usable_pixels, usable_ratio = _prepare_jersey_pixels(roi)
    dominant_bgr, _ = _dominant_color_and_quality_from_pixels(usable_pixels, usable_ratio)
    return dominant_bgr


def extract_jersey_color_bgr(player_bbox: Any, image: np.ndarray) -> np.ndarray:
    profile = extract_jersey_color_profile(player_bbox, image)
    return np.array(profile.color_bgr, dtype=np.float32)


def _weighted_medoid_index(lab_colors: np.ndarray, weights: np.ndarray, distance_scale: float) -> int:
    best_idx = 0
    best_score = -1.0

    for idx in range(len(lab_colors)):
        distances = np.linalg.norm(lab_colors - lab_colors[idx], axis=1)
        similarity = np.clip(1.0 - distances / max(distance_scale, 1.0), 0.0, 1.0)
        score = float(np.sum(similarity * weights))
        if score > best_score:
            best_score = score
            best_idx = idx

    return best_idx


def _jersey_warning(
    confidence: float,
    consensus_score: float,
    usable_ratio: float,
    sample_count: int
) -> Optional[str]:
    if confidence < 0.42:
        return "Low-confidence jersey color. Heavy shadow, blur, or occlusion may affect the label."
    if consensus_score < 0.52:
        return "Mixed jersey colors detected. Stripes, lighting, or overlap may make the label approximate."
    if usable_ratio < 0.35 or sample_count < 2:
        return "Limited clean jersey pixels were available, so this color is approximate."
    return None


def extract_jersey_color_profile(player_bbox: Any, image: np.ndarray) -> JerseyColorProfile:
    sampled_rois = _sample_jersey_rois(player_bbox, image)
    if not sampled_rois:
        fallback_bgr = (128, 128, 128)
        return JerseyColorProfile(
            color_bgr=fallback_bgr,
            color_name=get_color_name_from_bgr(fallback_bgr),
            confidence=0.10,
            warning="Could not isolate a clean jersey region.",
            usable_pixel_ratio=0.0,
            sample_count=0
        )

    sample_records = []
    for roi, roi_weight in sampled_rois:
        usable_pixels, usable_ratio = _prepare_jersey_pixels(roi)
        if len(usable_pixels) == 0:
            continue

        dominant_bgr, sample_confidence = _dominant_color_and_quality_from_pixels(usable_pixels, usable_ratio)
        bgr_tuple = _to_bgr_tuple(dominant_bgr)
        sample_records.append({
            "bgr": dominant_bgr,
            "bgr_tuple": bgr_tuple,
            "lab": _bgr_to_lab(bgr_tuple),
            "weight": float(roi_weight),
            "confidence": float(sample_confidence),
            "usable_ratio": float(usable_ratio),
        })

    if not sample_records:
        fallback_bgr = (128, 128, 128)
        return JerseyColorProfile(
            color_bgr=fallback_bgr,
            color_name=get_color_name_from_bgr(fallback_bgr),
            confidence=0.10,
            warning="Could not isolate a clean jersey region.",
            usable_pixel_ratio=0.0,
            sample_count=0
        )

    weights = np.array(
        [record["weight"] * max(0.25, record["confidence"]) for record in sample_records],
        dtype=np.float32
    )
    lab_colors = np.array([record["lab"] for record in sample_records], dtype=np.float32)
    medoid_idx = _weighted_medoid_index(lab_colors, weights, distance_scale=32.0)
    distances = np.linalg.norm(lab_colors - lab_colors[medoid_idx], axis=1)

    keep_mask = distances <= 24.0
    minimum_kept = 2 if len(sample_records) >= 3 else 1
    if int(np.count_nonzero(keep_mask)) < minimum_kept:
        keep_mask = np.zeros(len(sample_records), dtype=bool)
        ordered_indices = np.argsort(distances)
        keep_mask[ordered_indices[:minimum_kept]] = True

    kept_weights = weights[keep_mask]
    kept_labs = lab_colors[keep_mask]
    if kept_labs.size == 0:
        kept_weights = weights
        kept_labs = lab_colors
        keep_mask = np.ones(len(sample_records), dtype=bool)

    final_lab = np.average(kept_labs, axis=0, weights=kept_weights)
    final_bgr = _to_bgr_tuple(_lab_to_bgr(final_lab))

    kept_distances = distances[keep_mask]
    consensus_score = float(np.average(
        np.clip(1.0 - kept_distances / 28.0, 0.0, 1.0),
        weights=kept_weights
    ))
    mean_quality = float(np.average(
        np.array([record["confidence"] for record in sample_records], dtype=np.float32)[keep_mask],
        weights=kept_weights
    ))
    usable_ratio = float(np.average(
        np.array([record["usable_ratio"] for record in sample_records], dtype=np.float32)[keep_mask],
        weights=kept_weights
    ))
    confidence = float(np.clip(
        0.45 * mean_quality + 0.35 * consensus_score + 0.20 * usable_ratio,
        0.08,
        0.99
    ))

    return JerseyColorProfile(
        color_bgr=final_bgr,
        color_name=get_color_name_from_bgr(final_bgr),
        confidence=confidence,
        warning=_jersey_warning(confidence, consensus_score, usable_ratio, len(sample_records)),
        usable_pixel_ratio=usable_ratio,
        sample_count=len(sample_records)
    )


def _team_warning(
    confidence: float,
    spread_score: float,
    inlier_ratio: float
) -> Optional[str]:
    if confidence < 0.50:
        return "Team color is approximate. Player samples were inconsistent across the frame."
    if spread_score < 0.55:
        return "This team cluster contains mixed jersey shades. Stripes, shadows, or overlap may affect the label."
    if inlier_ratio < 0.68:
        return "Some player color samples were excluded as outliers before computing the team color."
    return None


def extract_team_color_profile(players: List[Any], image: np.ndarray) -> TeamColorProfile:
    player_profiles = [extract_jersey_color_profile(player, image) for player in players]
    if not player_profiles:
        fallback_bgr = (128, 128, 128)
        return TeamColorProfile(
            color_bgr=fallback_bgr,
            color_name=get_color_name_from_bgr(fallback_bgr),
            confidence=0.10,
            warning="No player color samples were available.",
            player_count=0
        )

    lab_colors = np.array(
        [_bgr_to_lab(profile.color_bgr) for profile in player_profiles],
        dtype=np.float32
    )
    weights = np.array(
        [max(0.30, profile.confidence) for profile in player_profiles],
        dtype=np.float32
    )
    medoid_idx = _weighted_medoid_index(lab_colors, weights, distance_scale=38.0)
    distances = np.linalg.norm(lab_colors - lab_colors[medoid_idx], axis=1)

    if len(player_profiles) >= 4:
        adaptive_threshold = max(18.0, float(np.percentile(distances, 70)))
    else:
        adaptive_threshold = 24.0

    keep_mask = distances <= adaptive_threshold
    minimum_kept = 2 if len(player_profiles) >= 2 else 1
    if int(np.count_nonzero(keep_mask)) < minimum_kept:
        keep_mask = np.zeros(len(player_profiles), dtype=bool)
        ordered_indices = np.argsort(distances)
        keep_mask[ordered_indices[:minimum_kept]] = True

    kept_weights = weights[keep_mask]
    kept_labs = lab_colors[keep_mask]
    final_lab = np.average(kept_labs, axis=0, weights=kept_weights)
    final_bgr = _to_bgr_tuple(_lab_to_bgr(final_lab))

    kept_distances = distances[keep_mask]
    spread = float(np.average(kept_distances, weights=kept_weights))
    spread_score = max(0.0, 1.0 - spread / 28.0)
    base_confidence = float(np.average(
        np.array([profile.confidence for profile in player_profiles], dtype=np.float32)[keep_mask],
        weights=kept_weights
    ))
    inlier_ratio = float(np.sum(kept_weights) / max(np.sum(weights), 1e-6))
    confidence = float(np.clip(
        0.45 * base_confidence + 0.30 * spread_score + 0.25 * inlier_ratio,
        0.10,
        0.99
    ))

    return TeamColorProfile(
        color_bgr=final_bgr,
        color_name=get_color_name_from_bgr(final_bgr),
        confidence=confidence,
        warning=_team_warning(confidence, spread_score, inlier_ratio),
        player_count=len(player_profiles)
    )


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

    if sat < 30:
        return _neutral_name(val)

    if 8 <= hue <= 30 and sat < 90 and val > 190:
        if sat < 75:
            return "Beige"

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
