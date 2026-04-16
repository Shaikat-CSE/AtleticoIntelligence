"""
Offside analysis with perspective-corrected geometric calculations.

This module uses camera calibration and homography to accurately determine
offside positions in real-world pitch coordinates.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import logging

from ..detection import BoundingBox
from .camera_calibration import CameraCalibrator, CalibrationResult, PerspectiveCorrector

logger = logging.getLogger(__name__)


@dataclass
class PlayerPosition:
    """Player position with both image and pitch coordinates."""
    player_id: int
    bounding_box: BoundingBox
    image_position: Tuple[float, float]  # (x, y) in pixels
    pitch_position: Optional[Tuple[float, float]]  # (x, y) in meters
    team: str  # "attacking" or "defending"


@dataclass
class OffsideAnalysisResult:
    """Result of offside analysis with geometric accuracy."""
    decision: str  # "OFFSIDE", "ONSIDE", or "UNKNOWN"
    attacker: BoundingBox
    second_last_defender: BoundingBox
    confidence: float
    attacker_pitch_pos: Optional[Tuple[float, float]]
    defender_pitch_pos: Optional[Tuple[float, float]]
    offside_margin_meters: float  # How far offside/onside in meters
    offside_line_image: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    calibration_quality: str  # "good", "poor", "fallback"


class GeometricOffsideAnalyzer:
    """
    Analyze offside using geometric transformations with perspective correction.
    """
    
    def __init__(
        self, 
        tolerance_meters: float = 0.5,  # ~50cm tolerance (typical for VAR)
        goal_direction: str = "right",
        calibrator: Optional[CameraCalibrator] = None
    ):
        self.tolerance_meters = tolerance_meters
        self.goal_direction = goal_direction
        self.calibrator = calibrator
        
    def analyze(
        self, 
        team1: List[BoundingBox], 
        team2: List[BoundingBox],
        image: Optional[np.ndarray] = None
    ) -> OffsideAnalysisResult:
        """
        Analyze offside with perspective-corrected geometry.
        
        Args:
            team1: List of players in team 1
            team2: List of players in team 2
            image: Optional image for camera calibration
            
        Returns:
            OffsideAnalysisResult with geometrically accurate decision
        """
        if not team1 or not team2:
            logger.warning("Insufficient players for offside analysis")
            return self._create_unknown_result(None, None)
        
        # Calibrate camera if image provided and not already calibrated
        if image is not None and self.calibrator is None:
            self.calibrator = CameraCalibrator()
            calibration = self.calibrator.auto_calibrate(image)
        elif self.calibrator is not None:
            # Use provided calibrator
            calibration = CalibrationResult(
                homography_matrix=self.calibrator._homography if self.calibrator._homography is not None else np.eye(3),
                inverse_homography=self.calibrator._inverse_homography if self.calibrator._inverse_homography is not None else np.eye(3),
                vanishing_point=None,
                pitch_corners_image=np.array([]),
                reprojection_error=0.0,
                is_valid=self.calibrator._homography is not None
            )
        else:
            calibration = None
        
        # Determine attacking and defending teams
        attacking_team, defending_team = self._determine_teams(team1, team2)
        
        # Get attacking player (furthest toward goal)
        attacker = self._find_attacking_player(attacking_team)
        if attacker is None:
            logger.warning("No attacker found")
            return self._create_unknown_result(None, None)
        
        # Get defending players sorted by distance to goal
        defenders_sorted = self._sort_defenders_by_goal_distance(defending_team)
        if len(defenders_sorted) < 2:
            logger.warning("Not enough defenders (need at least 2)")
            return self._create_unknown_result(attacker, None)
        
        # Second-last defender
        second_last_defender = defenders_sorted[-2]
        
        # Transform positions to pitch coordinates
        attacker_pitch_pos = self._get_foot_position_pitch(attacker, calibration)
        defender_pitch_pos = self._get_foot_position_pitch(second_last_defender, calibration)
        
        # Calculate offside using geometric comparison
        if attacker_pitch_pos is not None and defender_pitch_pos is not None:
            is_offside, margin = self._compare_positions_geometric(
                attacker_pitch_pos, 
                defender_pitch_pos
            )
            calibration_quality = "good" if calibration and calibration.is_valid else "fallback"
        else:
            # Fallback to simple image-space comparison (less accurate)
            is_offside, margin = self._compare_positions_image_space(
                attacker.foot_position,
                second_last_defender.foot_position
            )
            calibration_quality = "poor"
        
        decision = "OFFSIDE" if is_offside else "ONSIDE"
        confidence = self._calculate_confidence(attacker, second_last_defender, calibration)
        
        # Get offside line for visualization
        offside_line = None
        if calibration and calibration.is_valid and defender_pitch_pos is not None:
            corrector = PerspectiveCorrector(calibration)
            h = image.shape[0] if image is not None else 720
            offside_line = corrector.get_offside_line_points(defender_pitch_pos[0], h)
        
        logger.info(
            f"Offside decision: {decision}, "
            f"margin: {margin:.2f}m, "
            f"calibration: {calibration_quality}"
        )
        
        return OffsideAnalysisResult(
            decision=decision,
            attacker=attacker,
            second_last_defender=second_last_defender,
            confidence=confidence,
            attacker_pitch_pos=attacker_pitch_pos,
            defender_pitch_pos=defender_pitch_pos,
            offside_margin_meters=margin,
            offside_line_image=offside_line,
            calibration_quality=calibration_quality
        )
    
    def _determine_teams(
        self, 
        team1: List[BoundingBox], 
        team2: List[BoundingBox]
    ) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        """Determine which team is attacking based on average position."""
        if not team1 or not team2:
            return team1, team2
        
        # Calculate average x position of each team
        team1_avg_x = np.mean([p.foot_position[0] for p in team1])
        team2_avg_x = np.mean([p.foot_position[0] for p in team2])
        
        if self.goal_direction == "right":
            # Attacking team is closer to right side (higher x)
            return (team1, team2) if team1_avg_x > team2_avg_x else (team2, team1)
        else:
            # Attacking team is closer to left side (lower x)
            return (team1, team2) if team1_avg_x < team2_avg_x else (team2, team1)
    
    def _find_attacking_player(
        self, 
        attacking_team: List[BoundingBox]
    ) -> Optional[BoundingBox]:
        """Find the player furthest toward the opponent's goal."""
        if not attacking_team:
            return None
        
        if self.goal_direction == "right":
            return max(attacking_team, key=lambda p: p.foot_position[0])
        else:
            return min(attacking_team, key=lambda p: p.foot_position[0])
    
    def _sort_defenders_by_goal_distance(
        self, 
        defending_team: List[BoundingBox]
    ) -> List[BoundingBox]:
        """Sort defenders by their distance from the goal they're defending."""
        if self.goal_direction == "right":
            # Goal on right, defenders closer to right have lower x
            return sorted(defending_team, key=lambda p: p.foot_position[0], reverse=True)
        else:
            # Goal on left, defenders closer to left have higher x
            return sorted(defending_team, key=lambda p: p.foot_position[0])
    
    def _get_foot_position_pitch(
        self, 
        player: BoundingBox,
        calibration: Optional[CalibrationResult]
    ) -> Optional[Tuple[float, float]]:
        """Transform player's foot position to pitch coordinates."""
        if calibration is None or not calibration.is_valid:
            return None
        
        try:
            foot_image = player.foot_position
            foot_pitch = self._transform_point(foot_image, calibration.homography_matrix)
            return foot_pitch
        except Exception as e:
            logger.warning(f"Failed to transform player position: {e}")
            return None
    
    def _transform_point(
        self, 
        image_point: Tuple[float, float],
        homography: np.ndarray
    ) -> Tuple[float, float]:
        """Transform a point using homography matrix."""
        point = np.array([image_point[0], image_point[1], 1.0])
        transformed = homography @ point
        # Normalize by w
        w = transformed[2]
        if abs(w) < 1e-10:
            return (0.0, 0.0)
        return (float(transformed[0] / w), float(transformed[1] / w))
    
    def _compare_positions_geometric(
        self,
        attacker_pitch_pos: Tuple[float, float],
        defender_pitch_pos: Tuple[float, float]
    ) -> Tuple[bool, float]:
        """
        Compare positions in pitch coordinates (meters).
        
        Returns:
            Tuple of (is_offside, margin_in_meters)
        """
        attacker_x = attacker_pitch_pos[0]
        defender_x = defender_pitch_pos[0]
        
        if self.goal_direction == "right":
            margin = attacker_x - defender_x
            is_offside = margin > self.tolerance_meters
        else:
            margin = defender_x - attacker_x
            is_offside = margin > self.tolerance_meters
        
        return is_offside, margin
    
    def _compare_positions_image_space(
        self,
        attacker_image_pos: Tuple[float, float],
        defender_image_pos: Tuple[float, float]
    ) -> Tuple[bool, float]:
        """
        Fallback comparison in image space (pixels).
        Less accurate but works without calibration.
        """
        attacker_x = attacker_image_pos[0]
        defender_x = defender_image_pos[0]
        
        if self.goal_direction == "right":
            margin_pixels = attacker_x - defender_x
            is_offside = margin_pixels > 10  # ~10 pixels tolerance
        else:
            margin_pixels = defender_x - attacker_x
            is_offside = margin_pixels > 10
        
        # Rough conversion to meters for reporting (assume ~100m = image width)
        margin_meters = margin_pixels / 10.0  # Very rough estimate
        
        return is_offside, margin_meters
    
    def _calculate_confidence(
        self,
        attacker: BoundingBox,
        defender: BoundingBox,
        calibration: Optional[CalibrationResult]
    ) -> float:
        """Calculate confidence score based on detection quality and calibration."""
        # Base confidence from detection scores
        base_confidence = (attacker.confidence + defender.confidence) / 2
        
        # Calibration quality factor
        if calibration is None:
            calib_factor = 0.6  # Reduced confidence without calibration
        elif calibration.is_valid:
            # Better calibration = higher confidence
            error = calibration.reprojection_error
            calib_factor = max(0.5, 1.0 - (error / 10.0))
        else:
            calib_factor = 0.5
        
        # Combine factors
        confidence = base_confidence * 0.7 + calib_factor * 0.3
        return min(1.0, max(0.0, confidence))
    
    def _create_unknown_result(
        self,
        attacker: Optional[BoundingBox],
        defender: Optional[BoundingBox]
    ) -> OffsideAnalysisResult:
        """Create an unknown result when analysis fails."""
        return OffsideAnalysisResult(
            decision="UNKNOWN",
            attacker=attacker or BoundingBox(0, 0, 0, 0, 0, 0, "unknown"),
            second_last_defender=defender or BoundingBox(0, 0, 0, 0, 0, 0, "unknown"),
            confidence=0.0,
            attacker_pitch_pos=None,
            defender_pitch_pos=None,
            offside_margin_meters=0.0,
            offside_line_image=None,
            calibration_quality="failed"
        )


# Backward compatibility function
def analyze_offside(
    team1: List[BoundingBox], 
    team2: List[BoundingBox], 
    goal_direction: str = "right",
    image: Optional[np.ndarray] = None,
    calibrator: Optional[CameraCalibrator] = None
) -> OffsideAnalysisResult:
    """
    Analyze offside with geometric corrections.
    
    This is the main entry point - replaces the old analyze_offside function.
    """
    analyzer = GeometricOffsideAnalyzer(
        tolerance_meters=0.5,
        goal_direction=goal_direction,
        calibrator=calibrator
    )
    return analyzer.analyze(team1, team2, image)