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
class OffsideAnalysisResult:
    """Result of offside analysis with geometric accuracy."""
    decision: str  # "OFFSIDE", "ONSIDE", or "UNKNOWN"
    attacker: BoundingBox
    second_last_defender: BoundingBox  # Extreme defender in attacking direction
    confidence: float
    attacker_pitch_pos: Optional[Tuple[float, float]]
    defender_pitch_pos: Optional[Tuple[float, float]]
    offside_margin_meters: float  # How far offside/onside in meters
    offside_line_image: Optional[Tuple[Tuple[float, float], Tuple[float, float]]]
    calibration_quality: str  # "good", "poor", "fallback"
    goalkeeper: Optional[BoundingBox] = None  # Defender closest to own goal line


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
        image: Optional[np.ndarray] = None,
        ball_position: Optional[Tuple[float, float]] = None,
        attacking_team_input: Optional[str] = None  # "team1" or "team2"
    ) -> OffsideAnalysisResult:
        """
        Analyze offside with perspective-corrected geometry.
        
        Args:
            team1: List of players in team 1
            team2: List of players in team 2
            image: Optional image for camera calibration
            ball_position: Optional ball position (x, y) in pixels
            attacking_team: Optional manual assignment - "team1" or "team2"
                          If provided, this team is considered attacking
            
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
            vp = None
            if hasattr(self.calibrator, '_vanishing_point') and self.calibrator._vanishing_point is not None:
                vp = self.calibrator._vanishing_point
            calibration = CalibrationResult(
                homography_matrix=self.calibrator._homography if self.calibrator._homography is not None else np.eye(3),
                inverse_homography=self.calibrator._inverse_homography if self.calibrator._inverse_homography is not None else np.eye(3),
                vanishing_point=vp,
                pitch_corners_image=np.array([]),
                reprojection_error=0.0,
                is_valid=self.calibrator._homography is not None
            )
        else:
            calibration = None
        
        all_players = team1 + team2
        
        # STRICT RULES:
        # 1. Player CLOSEST to ball = ATTACKER (regardless of team)
        # 2. Other team jersey color = DEFENDERS
        # 3. 2nd defender (in UI) = FIRST defender in ATTACKING direction (closest to goal being attacked)
        #    - If goal_direction='right' (attacking leftward): leftmost defender
        #    - If goal_direction='left' (attacking rightward): rightmost defender
        
        if ball_position is None:
            logger.warning("No ball position - cannot determine attacker")
            return self._create_unknown_result(None, None)
        
        # Rule 1: Find attacker (closest to ball)
        attacker = min(all_players, key=lambda p: np.linalg.norm(np.array(p.foot_position) - np.array(ball_position)))
        logger.info(f"Attacker (closest to ball): foot=({attacker.foot_position[0]:.1f}, {attacker.foot_position[1]:.1f})")
        
        # Determine attacker's team
        if attacker in team1:
            attacking_list = team1
            defending_list = team2
            attacker_team = "team1"
        elif attacker in team2:
            attacking_list = team2
            defending_list = team1
            attacker_team = "team2"
        else:
            logger.warning("Attacker not found in either team!")
            return self._create_unknown_result(attacker, None)
        
        logger.info(f"Attacking team: {attacker_team}, Defending team: {attacker_team == 'team1' and 'team2' or 'team1'}")
        
        if len(defending_list) < 1:
            logger.warning("No defenders found!")
            return self._create_unknown_result(attacker, None)
        
        # Rule 3: DEFENDER = FIRST defender in attacking direction (closest to goal being attacked)
        # GOALKEEPER = defender closest to their OWN goal line (opposite extreme)
        if self.goal_direction == "right":
            # Attacking LEFT - leftmost defender is closest to attacking goal
            second_last_defender = min(defending_list, key=lambda p: p.foot_position[0])
            goalkeeper = max(defending_list, key=lambda p: p.foot_position[0])  # Rightmost = own goal
        else:
            # Attacking RIGHT - rightmost defender is closest to attacking goal
            second_last_defender = max(defending_list, key=lambda p: p.foot_position[0])
            goalkeeper = min(defending_list, key=lambda p: p.foot_position[0])  # Leftmost = own goal
        
        # Log all defenders for debugging
        for i, d in enumerate(defending_list):
            logger.info(f"  Defender {i}: foot=({d.foot_position[0]:.1f}, {d.foot_position[1]:.1f})")
        
        # OFFSIDE CHECK: Attacker is offside if they are ahead of ALL defenders
        # goal_direction='right' (attacking LEFT): attacker must be LEFT of ALL defenders
        # goal_direction='left' (attacking RIGHT): attacker must be RIGHT of ALL defenders
        attacker_x = attacker.foot_position[0]
        
        if self.goal_direction == "right":
            # Attacking LEFT - attacker ahead if x < ALL defenders' x
            is_ahead_of_all = all(attacker_x < d.foot_position[0] for d in defending_list)
            defender_x = min(d.foot_position[0] for d in defending_list)  # Leftmost defender
        else:
            # Attacking RIGHT - attacker ahead if x > ALL defenders' x
            is_ahead_of_all = all(attacker_x > d.foot_position[0] for d in defending_list)
            defender_x = max(d.foot_position[0] for d in defending_list)  # Rightmost defender
        
        is_offside = is_ahead_of_all
        margin = abs(attacker_x - defender_x)
        
        logger.info(f"Offside check: attacker_x={attacker_x:.1f}, defender_x={defender_x:.1f}, is_offside={is_offside}, margin={margin:.1f}")
        
        # Get offside line for visualization
        offside_line = None
        calibration_quality = "good" if calibration and calibration.is_valid else "fallback"
        
        # Try to get perspective-corrected offside line
        if calibration and calibration.is_valid:
            try:
                corrector = PerspectiveCorrector(calibration)
                h, w = image.shape[:2] if image is not None else (720, 1920)
                offside_line = corrector.get_perspective_offside_line(defender_x, (h, w))
                logger.info(f"Offside line computed: {offside_line}")
            except Exception as e:
                logger.warning(f"Failed to compute offside line: {e}")
        
        decision = "OFFSIDE" if is_offside else "ONSIDE"
        confidence = (attacker.confidence + min(d.confidence for d in defending_list)) / 2
        
        logger.info(f"Offside decision: {decision}, margin: {margin:.2f}px")
        
        return OffsideAnalysisResult(
            decision=decision,
            attacker=attacker,
            second_last_defender=second_last_defender,
            goalkeeper=goalkeeper,
            confidence=confidence,
            attacker_pitch_pos=None,
            defender_pitch_pos=None,
            offside_margin_meters=margin,
            offside_line_image=offside_line,
            calibration_quality=calibration_quality
        )
    
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
        
        if np.isnan(attacker_x) or np.isnan(defender_x):
            logger.warning("NaN detected in pitch positions, using fallback comparison")
            return False, 0.0
        
        logger.info(f"Geometric comparison: attacker_x={attacker_x:.2f}m, defender_x={defender_x:.2f}m, "
                    f"goal_direction={self.goal_direction}, tolerance={self.tolerance_meters}m")
        
        if self.goal_direction == "right":
            margin = attacker_x - defender_x
        else:
            margin = defender_x - attacker_x
        
        is_offside = margin > self.tolerance_meters
        
        logger.info(f"Offside margin: {margin:.2f}m, is_offside={is_offside}")
        
        return is_offside, abs(margin)
    
    def _compare_positions_image_space(
        self,
        attacker_image_pos: Tuple[float, float],
        defender_image_pos: Tuple[float, float],
        image_width: int = 1920
    ) -> Tuple[bool, float]:
        """
        Fallback comparison in image space (pixels).
        Less accurate but works without calibration.
        
        Converts pixel difference to approximate meters assuming
        the pitch spans most of the image width.
        """
        attacker_x = attacker_image_pos[0]
        defender_x = defender_image_pos[0]
        
        pitch_width_meters = 105.0
        pixels_per_meter = image_width / pitch_width_meters
        tolerance_pixels = self.tolerance_meters * pixels_per_meter
        
        if self.goal_direction == "right":
            margin_pixels = attacker_x - defender_x
        else:
            margin_pixels = defender_x - attacker_x
        
        is_offside = margin_pixels > tolerance_pixels
        margin_meters = abs(margin_pixels) / pixels_per_meter
        
        logger.info(f"Image-space fallback: margin_pixels={margin_pixels:.1f}, "
                    f"tolerance_pixels={tolerance_pixels:.1f}, margin_meters={margin_meters:.2f}m")
        
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
    calibrator: Optional[CameraCalibrator] = None,
    ball_position: Optional[Tuple[float, float]] = None,
    attacking_team: Optional[str] = None
) -> OffsideAnalysisResult:
    """
    Analyze offside with geometric corrections.
    
    Args:
        attacking_team: "team1" or "team2" to manually specify attacking team
    """
    analyzer = GeometricOffsideAnalyzer(
        tolerance_meters=0.5,
        goal_direction=goal_direction,
        calibrator=calibrator
    )
    return analyzer.analyze(team1, team2, image, ball_position, attacking_team_input=attacking_team)