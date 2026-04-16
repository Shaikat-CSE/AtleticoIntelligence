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
            # Calculate homography using RANSAC for robustness
            H, mask = cv2.findHomography(
                image_points.astype(np.float32),
                pitch_points.astype(np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=3.0
            )
            
            if H is None:
                logger.error("Failed to compute homography matrix")
                return self._create_invalid_result()
            
            # Calculate inverse homography
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
        
        Args:
            image: Input image
            detected_lines: Optional pre-detected pitch lines as list of (start, end) points
            
        Returns:
            CalibrationResult
        """
        if detected_lines is None:
            detected_lines = self._detect_pitch_lines(image)
        
        if len(detected_lines) < 4:
            logger.warning("Insufficient pitch lines for auto-calibration, using default")
            return self._default_calibration(image.shape)
        
        # Try to identify key pitch lines (sidelines, goal lines, center line)
        h, w = image.shape[:2]
        
        # Use heuristic: assume image shows roughly centered view
        # Map image corners to pitch corners as initial estimate
        image_points = np.array([
            [w * 0.2, h * 0.3],   # Top-left (left sideline, far)
            [w * 0.8, h * 0.3],   # Top-right (right sideline, far)
            [w * 0.8, h * 0.8],   # Bottom-right (right sideline, near)
            [w * 0.2, h * 0.8],   # Bottom-left (left sideline, near)
        ], dtype=np.float32)
        
        pitch_points = np.array([
            [0, 0],
            [self.pitch.width, 0],
            [self.pitch.width, self.pitch.height],
            [0, self.pitch.height]
        ], dtype=np.float32)
        
        return self.calibrate_from_points(image_points, pitch_points)
    
    def _detect_pitch_lines(
        self, 
        image: np.ndarray
    ) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Detect white pitch lines in the image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line transform
        lines = cv2.HoughLinesP(
            edges, 
            rho=1, 
            theta=np.pi/180, 
            threshold=100,
            minLineLength=min(image.shape[:2]) // 4,
            maxLineGap=20
        )
        
        detected = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Filter by color - pitch lines are white/bright
                mid_x, mid_y = int((x1 + x2) / 2), int((y1 + y2) / 2)
                if 0 <= mid_x < image.shape[1] and 0 <= mid_y < image.shape[0]:
                    color = image[mid_y, mid_x]
                    brightness = np.mean(color)
                    if brightness > 150:  # Bright line
                        detected.append(((x1, y1), (x2, y2)))
        
        return detected
    
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
    
    def _default_calibration(self, image_shape: Tuple[int, ...]) -> CalibrationResult:
        """Create a default calibration when auto-calibration fails."""
        h, w = image_shape[:2]
        
        # Assume standard broadcast view
        image_points = np.array([
            [w * 0.1, h * 0.2],
            [w * 0.9, h * 0.2],
            [w * 0.9, h * 0.9],
            [w * 0.1, h * 0.9],
        ], dtype=np.float32)
        
        pitch_points = np.array([
            [0, 0],
            [self.pitch.width, 0],
            [self.pitch.width, self.pitch.height],
            [0, self.pitch.height]
        ], dtype=np.float32)
        
        result = self.calibrate_from_points(image_points, pitch_points)
        logger.warning("Using default calibration - results may be inaccurate")
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
    
    def __init__(self, calibration: CalibrationResult):
        self.calibration = calibration
        
    def get_offside_line_points(
        self, 
        defender_x_position: float,
        image_height: int
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Get the offside line endpoints in image coordinates.
        
        Args:
            defender_x_position: Defender's x position in pitch coordinates (meters)
            image_height: Height of the image in pixels
            
        Returns:
            Tuple of (top_point, bottom_point) in image coordinates
        """
        if not self.calibration.is_valid or len(self.calibration.pitch_corners_image) == 0:
            # Fallback to vertical line
            return ((defender_x_position, 0), (defender_x_position, image_height))
        
        try:
            # Get pitch height from corners
            pitch_height = self.calibration.pitch_corners_image.mean(axis=0)[1] if len(self.calibration.pitch_corners_image.shape) > 1 else 68.0
            
            # Get top and bottom of offside line using homography
            top_pitch = np.array([[defender_x_position, 0]], dtype=np.float32)
            bottom_pitch = np.array([[defender_x_position, pitch_height]], dtype=np.float32)
            
            top_image = cv2.perspectiveTransform(
                top_pitch.reshape(-1, 1, 2),
                self.calibration.inverse_homography
            )[0][0]
            
            bottom_image = cv2.perspectiveTransform(
                bottom_pitch.reshape(-1, 1, 2),
                self.calibration.inverse_homography
            )[0][0]
            
            return (tuple(top_image), tuple(bottom_image))
        except Exception as e:
            logger.warning(f"Failed to compute offside line: {e}, using fallback")
            return ((defender_x_position, 0), (defender_x_position, image_height))
    
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