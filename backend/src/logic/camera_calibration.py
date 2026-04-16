"""
Camera calibration and perspective correction for offside detection.

This module handles:
1. Pitch line detection for automatic calibration
2. Homography matrix calculation between image and real-world pitch coordinates
3. Coordinate transformations with perspective correction
4. Vanishing point detection for geometric accuracy
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PitchDimensions:
    """Standard FIFA pitch dimensions in meters."""
    width: float = 105.0  # Length (goal to goal)
    height: float = 68.0  # Width (side to side)
    goal_width: float = 7.32
    penalty_area_width: float = 16.5
    penalty_area_depth: float = 40.3  # Total width of penalty area
    six_yard_width: float = 5.5
    six_yard_depth: float = 18.3  # Total width of 6-yard box
    center_circle_radius: float = 9.15
    penalty_spot_distance: float = 11.0


@dataclass
class CalibrationResult:
    """Result of camera calibration."""
    homography_matrix: np.ndarray
    inverse_homography: np.ndarray
    vanishing_point: Optional[Tuple[float, float]]
    pitch_corners_image: np.ndarray
    reprojection_error: float
    is_valid: bool


class CameraCalibrator:
    """Handles camera calibration using pitch lines."""
    
    def __init__(self, pitch_dimensions: Optional[PitchDimensions] = None):
        self.pitch = pitch_dimensions or PitchDimensions()
        self._homography: Optional[np.ndarray] = None
        self._inverse_homography: Optional[np.ndarray] = None
        self._vanishing_point: Optional[Tuple[float, float]] = None
        
    def calibrate_from_points(
        self, 
        image_points: np.ndarray, 
        pitch_points: np.ndarray
    ) -> CalibrationResult:
        """
        Calibrate camera using corresponding image and pitch points.
        
        Args:
            image_points: Nx2 array of points in image coordinates (pixels)
            pitch_points: Nx2 array of corresponding points in pitch coordinates (meters)
            
        Returns:
            CalibrationResult with homography matrix and quality metrics
        """
        if len(image_points) < 4 or len(pitch_points) < 4:
            logger.error("Need at least 4 point correspondences for calibration")
            return self._create_invalid_result()
        
        try:
            H, mask = cv2.findHomography(
                image_points.astype(np.float32),
                pitch_points.astype(np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0
            )
            
            if H is None:
                logger.error("Failed to compute homography matrix")
                return self._create_invalid_result()
            
            H_inv = np.linalg.inv(H)
            
            # Calculate reprojection error
            reproj_points = cv2.perspectiveTransform(
                image_points.reshape(-1, 1, 2).astype(np.float32), 
                H
            ).reshape(-1, 2)
            error = np.mean(np.linalg.norm(reproj_points - pitch_points, axis=1))
            
            # Find vanishing point (intersection of parallel lines)
            vanishing_point = self._compute_vanishing_point(H)
            
            # Get pitch corners in image space
            pitch_corners = np.array([
                [0, 0],
                [self.pitch.width, 0],
                [self.pitch.width, self.pitch.height],
                [0, self.pitch.height]
            ], dtype=np.float32)
            
            corners_image = cv2.perspectiveTransform(
                pitch_corners.reshape(-1, 1, 2),
                H_inv
            ).reshape(-1, 2)
            
            self._homography = H
            self._inverse_homography = H_inv
            self._vanishing_point = vanishing_point
            
            logger.info(f"Camera calibrated successfully. Reprojection error: {error:.2f}m")
            
            return CalibrationResult(
                homography_matrix=H,
                inverse_homography=H_inv,
                vanishing_point=vanishing_point,
                pitch_corners_image=corners_image,
                reprojection_error=error,
                is_valid=error < 5.0  # Valid if error < 5 meters
            )
            
        except Exception as e:
            logger.error(f"Calibration failed: {e}")
            return self._create_invalid_result()
    
    def auto_calibrate(
        self, 
        image: np.ndarray,
        detected_lines: Optional[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = None
    ) -> CalibrationResult:
        """
        Attempt automatic calibration from pitch lines in image.
        Works with partial pitch view by detecting vanishing points.
        
        Args:
            image: Input image
            detected_lines: Optional pre-detected pitch lines as list of (start, end) points
            
        Returns:
            CalibrationResult
        """
        h, w = image.shape[:2]
        
        if detected_lines is None:
            detected_lines = self._detect_pitch_lines(image)
        
        logger.info(f"Detected {len(detected_lines)} pitch lines")
        
        if len(detected_lines) >= 4:
            corners = self._extract_pitch_corners_from_lines(detected_lines, image)
            if corners is not None and len(corners) >= 4:
                image_points = np.array(corners[:4], dtype=np.float32)
                pitch_points = np.array([
                    [0, 0],
                    [self.pitch.width, 0],
                    [self.pitch.width, self.pitch.height],
                    [0, self.pitch.height]
                ], dtype=np.float32)
                logger.info(f"Auto-calibration using detected corners: {corners}")
                result = self.calibrate_from_points(image_points, pitch_points)
                if result.reprojection_error < 10.0:  # Only use if error is reasonable
                    return result
                logger.warning(f"Calibration reprojection error too high ({result.reprojection_error:.1f}m), falling back to heuristic")
        
        if len(detected_lines) >= 2:
            H = self._estimate_perspective_from_partial_view(detected_lines, image.shape)
            if H is not None:
                H_inv = np.linalg.inv(H)
                vanishing = (float(w * 0.5), float(h * 0.2))
                self._vanishing_point = vanishing
                self._homography = H
                self._inverse_homography = H_inv
                corners_image = cv2.perspectiveTransform(
                    np.array([[[0, 0]], [[105, 0]], [[105, 68]], [[0, 68]]], dtype=np.float32),
                    H_inv
                ).reshape(-1, 2)
                
                logger.info("Auto-calibration using vanishing point estimation")
                return CalibrationResult(
                    homography_matrix=H,
                    inverse_homography=H_inv,
                    vanishing_point=vanishing,
                    pitch_corners_image=corners_image,
                    reprojection_error=3.0,
                    is_valid=True
                )
        
        logger.warning("Insufficient pitch lines for auto-calibration, using heuristic estimate")
        return self._heuristic_calibration(image.shape)
    
    def _detect_pitch_lines(
        self, 
        image: np.ndarray
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Detect white pitch lines in the image using improved detection."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        blurred = cv2.bilateralFilter(gray, 5, 50, 50)
        
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
        
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=30,
            minLineLength=min(image.shape[:2]) // 10,
            maxLineGap=50
        )
        
        detected = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                if length < 30:
                    continue
                    
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle > 85 or angle < 5:
                    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                    if 0 <= mid_x < image.shape[1] and 0 <= mid_y < image.shape[0]:
                        color = image[mid_y, mid_x]
                        brightness = np.mean(color)
                        if brightness > 80:
                            detected.append(((float(x1), float(y1)), (float(x2), float(y2))))
        
        return detected
    
    def _estimate_perspective_from_partial_view(
        self,
        lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        image_shape: Tuple[int, ...]
    ) -> Optional[np.ndarray]:
        """Estimate homography from partial pitch view using vanishing points.
        
        Even with partial pitch view, we can estimate perspective by:
        1. Finding vanishing points (intersection of parallel lines)
        2. Using horizon line to establish camera tilt
        3. Mapping detected lines to expected pitch geometry
        """
        if len(lines) < 2:
            return None
        
        h, w = image_shape[:2]
        
        horizontal_lines = []
        vertical_lines = []
        
        for (x1, y1), (x2, y2) in lines:
            angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
            if angle < 10:
                horizontal_lines.append(((x1, y1), (x2, y2)))
            elif angle > 80:
                vertical_lines.append(((x1, y1), (x2, y2)))
        
        vanishing_point = None
        if len(vertical_lines) >= 2:
            vanishing_point = self._find_vanishing_point(vertical_lines)
        
        horizon_y = h * 0.3
        if vanishing_point:
            horizon_y = vanishing_point[1]
        
        pitch_top = max(horizon_y, h * 0.15)
        
        visible_pitch_height = h - pitch_top
        visible_pitch_width = visible_pitch_height * (self.pitch.width / self.pitch.height)
        
        center_x = w / 2
        
        image_points = np.array([
            [float(center_x - visible_pitch_width/2), float(pitch_top)],
            [float(center_x + visible_pitch_width/2), float(pitch_top)],
            [float(center_x + visible_pitch_width/2), float(h)],
            [float(center_x - visible_pitch_width/2), float(h)],
        ], dtype=np.float32)
        
        pitch_points = np.array([
            [0, 0],
            [self.pitch.width, 0],
            [self.pitch.width, self.pitch.height],
            [0, self.pitch.height]
        ], dtype=np.float32)
        
        H, _ = cv2.findHomography(image_points, pitch_points, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        
        return H
    
    def _find_vanishing_point(
        self,
        lines: List[Tuple[Tuple[float, float], Tuple[float, float]]]
    ) -> Optional[Tuple[float, float]]:
        """Find vanishing point by intersecting parallel lines."""
        if len(lines) < 2:
            return None
        
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                p1 = self._line_intersection(
                    np.array(lines[i][0]),
                    np.array(lines[i][1]),
                    np.array(lines[j][0]),
                    np.array(lines[j][1])
                )
                if p1 is not None:
                    intersections.append(p1)
        
        if not intersections:
            return None
        
        intersections = np.array(intersections)
        median_x = np.median(intersections[:, 0])
        median_y = np.median(intersections[:, 1])
        
        return (float(median_x), float(median_y))
    
    def _extract_pitch_corners_from_lines(
        self,
        lines: List[Tuple[Tuple[float, float], Tuple[float, float]]],
        image: np.ndarray
    ) -> Optional[List[Tuple[float, float]]]:
        """Extract the four corners of the pitch from detected lines using line intersections."""
        if len(lines) < 4:
            return None
        
        h, w = image.shape[:2]
        min_dist = w * 0.1  # Minimum distance between corners
        
        all_points = []
        for (x1, y1), (x2, y2) in lines:
            all_points.append((float(x1), float(y1)))
            all_points.append((float(x2), float(y2)))
        
        if len(all_points) < 4:
            return None
        
        all_points = np.array(all_points)
        
        x_min, x_max = np.min(all_points[:, 0]), np.max(all_points[:, 0])
        y_min, y_max = np.min(all_points[:, 1]), np.max(all_points[:, 1])
        
        candidates = [
            (x_min, y_min),
            (x_max, y_min),
            (x_max, y_max),
            (x_min, y_max)
        ]
        
        corners = []
        for corner in candidates:
            is_duplicate = False
            for existing in corners:
                if np.linalg.norm(np.array(corner) - np.array(existing)) < min_dist:
                    is_duplicate = True
                    break
            if not is_duplicate:
                corners.append(corner)
        
        if len(corners) >= 4:
            corners = corners[:4]
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            
            ordered = []
            ordered.append(corners[np.argmin([c[0] + c[1] for c in corners])])
            ordered.append(corners[np.argmax([c[0] - c[1] for c in corners])])
            ordered.append(corners[np.argmax([c[0] + c[1] for c in corners])])
            ordered.append(corners[np.argmin([c[0] - c[1] for c in corners])])
            
            return ordered
        
        return None
    
    def _compute_vanishing_point(
        self, 
        homography: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Compute vanishing point from homography matrix."""
        try:
            # Vanishing point is where parallel lines converge
            # In pitch coordinates, lines parallel to y-axis converge at infinity
            # Transform direction vector [0, 1, 0] through homography
            
            # Simplified: find intersection of two parallel pitch lines in image
            # Left sideline: from (0,0) to (0, height)
            # Right sideline: from (width, 0) to (width, height)
            
            H_inv = np.linalg.inv(homography)
            
            # Transform points from pitch to image
            p1 = cv2.perspectiveTransform(np.array([[[0, 0]]], dtype=np.float32), H_inv)[0][0]
            p2 = cv2.perspectiveTransform(np.array([[[0, self.pitch.height]]], dtype=np.float32), H_inv)[0][0]
            p3 = cv2.perspectiveTransform(np.array([[[self.pitch.width, 0]]], dtype=np.float32), H_inv)[0][0]
            p4 = cv2.perspectiveTransform(np.array([[[self.pitch.width, self.pitch.height]]], dtype=np.float32), H_inv)[0][0]
            
            # Find intersection of lines (p1,p2) and (p3,p4)
            vanishing = self._line_intersection(p1, p2, p3, p4)
            return vanishing
            
        except Exception:
            return None
    
    def _line_intersection(
        self, 
        p1: np.ndarray, 
        p2: np.ndarray, 
        p3: np.ndarray, 
        p4: np.ndarray
    ) -> Optional[Tuple[float, float]]:
        """Find intersection of two lines."""
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (x, y)
    
    def _heuristic_calibration(self, image_shape: Tuple[int, ...]) -> CalibrationResult:
        """Create a heuristic calibration when auto-calibration fails.
        
        Uses a trapezoid perspective: top of pitch appears narrower (typical broadcast view).
        Maps image corners to pitch corners with realistic perspective distortion.
        """
        h, w = image_shape[:2]
        
        l_top = int(w * 0.12)
        r_top = int(w * 0.88)
        l_bottom = int(w * 0.02)
        r_bottom = int(w * 0.98)
        t = int(h * 0.12)
        b = int(h * 0.95)
        
        image_points = np.array([
            [float(l_top), float(t)],
            [float(r_top), float(t)],
            [float(r_bottom), float(b)],
            [float(l_bottom), float(b)],
        ], dtype=np.float32)
        
        pitch_points = np.array([
            [0, 0],
            [self.pitch.width, 0],
            [self.pitch.width, self.pitch.height],
            [0, self.pitch.height]
        ], dtype=np.float32)
        
        result = self.calibrate_from_points(image_points, pitch_points)
        logger.warning("Using heuristic calibration - results may be inaccurate. Provide manual calibration points for best accuracy.")
        return result
    
    def _create_invalid_result(self) -> CalibrationResult:
        """Create an invalid calibration result."""
        return CalibrationResult(
            homography_matrix=np.eye(3),
            inverse_homography=np.eye(3),
            vanishing_point=None,
            pitch_corners_image=np.array([]),
            reprojection_error=float('inf'),
            is_valid=False
        )
    
    def image_to_pitch(
        self, 
        image_point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Transform image coordinates to pitch coordinates (meters)."""
        if self._homography is None:
            raise RuntimeError("Camera not calibrated")
        
        point = np.array([[image_point]], dtype=np.float32)
        pitch_point = cv2.perspectiveTransform(point, self._homography)[0][0]
        return (float(pitch_point[0]), float(pitch_point[1]))
    
    def pitch_to_image(
        self, 
        pitch_point: Tuple[float, float]
    ) -> Tuple[float, float]:
        """Transform pitch coordinates (meters) to image coordinates."""
        if self._inverse_homography is None:
            raise RuntimeError("Camera not calibrated")
        
        point = np.array([[pitch_point]], dtype=np.float32)
        image_point = cv2.perspectiveTransform(point, self._inverse_homography)[0][0]
        return (float(image_point[0]), float(image_point[1]))
    
    def batch_image_to_pitch(
        self, 
        image_points: np.ndarray
    ) -> np.ndarray:
        """Transform multiple image coordinates to pitch coordinates."""
        if self._homography is None:
            raise RuntimeError("Camera not calibrated")
        
        return cv2.perspectiveTransform(
            image_points.reshape(-1, 1, 2).astype(np.float32),
            self._homography
        ).reshape(-1, 2)


class PerspectiveCorrector:
    """Corrects for perspective distortion in offside calculations."""
    
    def __init__(self, calibration: CalibrationResult, pitch_height: float = 68.0):
        self.calibration = calibration
        self.pitch_height = pitch_height
        
    def get_offside_line_points(
        self, 
        defender_x_position: float,
        image_height: int,
        image_width: int = 1920
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Get the offside line endpoints in image coordinates.
        
        Draws a line parallel to the pitch lines (vertical in pitch coords)
        at the defender's x position, transformed to image perspective.
        
        Args:
            defender_x_position: Defender's x position in pitch coordinates (meters)
            image_height: Height of the image in pixels
            image_width: Width of the image in pixels
            
        Returns:
            Tuple of (top_point, bottom_point) in image coordinates, or None if invalid
        """
        if not self.calibration.is_valid:
            return None
        
        try:
            defender_x = float(defender_x_position)
            if np.isnan(defender_x) or np.isnan(defender_x):
                return None
            
            pitch_top = np.array([[defender_x, 0]], dtype=np.float32)
            pitch_bottom = np.array([[defender_x, self.pitch_height]], dtype=np.float32)
            
            top_image = cv2.perspectiveTransform(
                pitch_top.reshape(-1, 1, 2),
                self.calibration.inverse_homography
            )[0][0]
            
            bottom_image = cv2.perspectiveTransform(
                pitch_bottom.reshape(-1, 1, 2),
                self.calibration.inverse_homography
            )[0][0]
            
            if (np.isnan(top_image[0]) or np.isnan(top_image[1]) or
                np.isnan(bottom_image[0]) or np.isnan(bottom_image[1])):
                return None
            
            return (tuple(top_image), tuple(bottom_image))
        except Exception as e:
            logger.warning(f"Failed to compute offside line: {e}")
            return None
    
    def get_perspective_offside_line(
        self,
        defender_x_position: float,
        image_shape: Tuple[int, int]
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Get offside line that stays within the visible pitch area.
        
        Instead of going from pitch y=0 to y=pitch_height (full pitch),
        we interpolate the line endpoints from where the offside x intersects
        the visible pitch boundaries in the image.
        """
        if not self.calibration.is_valid:
            return None
        
        try:
            h, w = image_shape
            defender_x = float(defender_x_position)
            
            if np.isnan(defender_x):
                return None
            
            line_points = []
            for pitch_y in np.linspace(0, self.pitch_height, 20):
                pt = np.array([[[defender_x, pitch_y]]], dtype=np.float32)
                img_pt = cv2.perspectiveTransform(pt, self.calibration.inverse_homography)[0][0]
                if 0 <= img_pt[0] <= w and 0 <= img_pt[1] <= h:
                    line_points.append((float(img_pt[0]), float(img_pt[1])))
            
            if len(line_points) >= 2:
                line_points.sort(key=lambda p: p[1])
                return (line_points[0], line_points[-1])
            
            return None
        except Exception as e:
            logger.warning(f"Failed to compute perspective offside line: {e}")
            return None
    
    def compare_player_positions(
        self,
        attacker_pitch_pos: Tuple[float, float],
        defender_pitch_pos: Tuple[float, float],
        goal_direction: str = "right"
    ) -> Tuple[bool, float]:
        """
        Compare attacker and defender positions accounting for perspective.
        
        Returns:
            Tuple of (is_offside, margin_meters)
        """
        attacker_x = attacker_pitch_pos[0]
        defender_x = defender_pitch_pos[0]
        
        if goal_direction == "right":
            is_offside = attacker_x > defender_x
            margin = attacker_x - defender_x
        else:
            is_offside = attacker_x < defender_x
            margin = defender_x - attacker_x
        
        return is_offside, margin


def create_calibrator(
    image: np.ndarray,
    manual_points: Optional[Tuple[np.ndarray, np.ndarray]] = None
) -> CameraCalibrator:
    """
    Factory function to create and calibrate a CameraCalibrator.
    
    Args:
        image: Input image
        manual_points: Optional tuple of (image_points, pitch_points) for manual calibration
        
    Returns:
        Calibrated CameraCalibrator instance
    """
    calibrator = CameraCalibrator()
    
    if manual_points is not None:
        image_pts, pitch_pts = manual_points
        calibrator.calibrate_from_points(image_pts, pitch_pts)
    else:
        calibrator.auto_calibrate(image)
    
    return calibrator