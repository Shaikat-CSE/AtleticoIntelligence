import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from ..detection import BoundingBox
from ..logic import OffsideAnalysisResult


class PitchVisualizer:
    """Visualize offside analysis results on video frames."""
    
    def __init__(
        self,
        output_dir: str = "output/annotated",
        bbox_thickness: int = 2,
        attacker_color: Tuple[int, int, int] = (255, 0, 0),  # Red
        defender_color: Tuple[int, int, int] = (0, 0, 255),  # Blue
        ball_color: Tuple[int, int, int] = (0, 255, 0),      # Green
        text_color: Tuple[int, int, int] = (255, 255, 255),  # White
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bbox_thickness = bbox_thickness
        self.attacker_color = attacker_color
        self.defender_color = defender_color
        self.ball_color = ball_color
        self.text_color = text_color

    def annotate_frame(
        self,
        image: np.ndarray,
        detection_result,
        offside_result: OffsideAnalysisResult,
        team1: List[BoundingBox],
        team2: List[BoundingBox],
        output_filename: str = "annotated_frame.jpg"
    ) -> str:
        """
        Annotate frame with detection and offside analysis results.
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]

        # Draw all detected players
        for player in detection_result.players:
            self._draw_player_bbox(annotated, player, team1, team2)

        # Draw ball if detected
        if detection_result.ball:
            self._draw_ball_bbox(annotated, detection_result.ball)

        # Draw offside analysis results
        if offside_result and offside_result.decision != "UNKNOWN":
            self._draw_offside_analysis(annotated, offside_result, h, w)

        # Add calibration quality indicator
        if offside_result:
            quality_text = f"Cal: {offside_result.calibration_quality.upper()}"
            quality_color = {
                "good": (0, 255, 0),
                "poor": (0, 165, 255),
                "fallback": (0, 0, 255),
                "failed": (128, 128, 128)
            }.get(offside_result.calibration_quality, (128, 128, 128))
            
            cv2.putText(
                annotated, 
                quality_text, 
                (w - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                quality_color, 
                2
            )

        # Save annotated image
        output_path = Path(output_filename)
        if not output_path.is_absolute():
            output_path = self.output_dir / output_filename

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        return str(output_path)

    def _draw_player_bbox(
        self,
        image: np.ndarray,
        player: BoundingBox,
        team1: List[BoundingBox],
        team2: List[BoundingBox],
    ):
        """Draw bounding box for a player with team identification."""
        # Determine team
        if player in team1:
            color = self.attacker_color
            label = "ATT"
        elif player in team2:
            color = self.defender_color
            label = "DEF"
        else:
            color = (128, 128, 128)  # Gray for unknown
            label = "?"

        # Draw bounding box
        x1, y1, x2, y2 = int(player.x1), int(player.y1), int(player.x2), int(player.y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.bbox_thickness)

        # Draw label
        cv2.putText(
            image, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            color, 
            1
        )

        # Draw foot position marker
        foot_x, foot_y = int(player.foot_position[0]), int(player.foot_position[1])
        cv2.circle(image, (foot_x, foot_y), 5, color, -1)

    def _draw_ball_bbox(self, image: np.ndarray, ball: BoundingBox):
        """Draw bounding box for the ball."""
        x1, y1, x2, y2 = int(ball.x1), int(ball.y1), int(ball.x2), int(ball.y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), self.ball_color, self.bbox_thickness)
        cv2.putText(
            image, 
            "BALL", 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            self.ball_color, 
            1
        )

    def _draw_offside_analysis(
        self, 
        image: np.ndarray, 
        offside_result: OffsideAnalysisResult,
        image_height: int,
        image_width: int
    ):
        """Draw offside line and decision on the image."""
        
        # Highlight attacker
        attacker = offside_result.attacker
        ax, ay = int(attacker.center[0]), int(attacker.center[1])
        cv2.circle(image, (ax, ay), 20, self.attacker_color, 3)
        cv2.putText(
            image,
            "ATTACKER",
            (ax - 40, ay - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.attacker_color,
            2
        )

        # Highlight second-last defender
        defender = offside_result.second_last_defender
        dx, dy = int(defender.center[0]), int(defender.center[1])
        cv2.circle(image, (dx, dy), 20, self.defender_color, 3)
        cv2.putText(
            image,
            "DEFENDER",
            (dx - 40, dy - 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.defender_color,
            2
        )

        # Draw decision text
        decision = offside_result.decision
        confidence = offside_result.confidence
        margin = offside_result.offside_margin_meters
        
        # Position text at top-left
        text_y = 50
        
        # Decision with color coding
        if decision == "OFFSIDE":
            decision_color = (0, 0, 255)  # Red
        else:
            decision_color = (0, 255, 0)  # Green
            
        cv2.putText(
            image,
            f"{decision}",
            (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            decision_color,
            3
        )
        
        # Confidence and margin
        cv2.putText(
            image,
            f"Conf: {confidence:.2f}",
            (20, text_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.text_color,
            2
        )
        
        margin_text = f"Margin: {margin:.2f}m"
        if decision == "OFFSIDE":
            margin_text = f"+{margin:.2f}m offside"
        elif decision == "ONSIDE":
            margin_text = f"{-margin:.2f}m onside"
            
        cv2.putText(
            image,
            margin_text,
            (20, text_y + 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.text_color,
            2
        )

    def annotate_from_positions(
        self,
        image: np.ndarray,
        attacker_pos: Tuple[float, float],
        defender_pos: Tuple[float, float],
        ball_pos: Optional[Tuple[float, float]] = None,
        offside_line_x: Optional[float] = None,
        offside_line_top: Optional[Tuple[float, float]] = None,
        offside_line_bottom: Optional[Tuple[float, float]] = None,
        decision: str = "UNKNOWN",
        all_attacking: Optional[List[Dict[str, float]]] = None,
        all_defending: Optional[List[Dict[str, float]]] = None,
        output_filename: str = "annotated_frame.jpg"
    ) -> str:
        """
        Annotate frame from raw positions (used by legacy code).
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]

        # Draw all attacking players
        if all_attacking:
            for p in all_attacking:
                px, py = int(p["x"]), int(p["y"])
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(annotated, (px, py), 15, self.attacker_color, 2)
                    cv2.putText(
                        annotated, 
                        "ATT", 
                        (px - 15, py - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        self.attacker_color, 
                        1
                    )

        # Draw all defending players
        if all_defending:
            for p in all_defending:
                px, py = int(p["x"]), int(p["y"])
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(annotated, (px, py), 15, self.defender_color, 2)
                    cv2.putText(
                        annotated, 
                        "DEF", 
                        (px - 15, py - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        self.defender_color, 
                        1
                    )

        # Draw attacker
        ax, ay = int(attacker_pos[0]), int(attacker_pos[1])
        if 0 <= ax < w and 0 <= ay < h:
            cv2.circle(annotated, (ax, ay), 20, self.attacker_color, 3)
            cv2.putText(
                annotated, 
                "ATTACKER", 
                (ax - 40, ay - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                self.attacker_color, 
                2
            )

        # Draw defender
        dx, dy = int(defender_pos[0]), int(defender_pos[1])
        if 0 <= dx < w and 0 <= dy < h:
            cv2.circle(annotated, (dx, dy), 20, self.defender_color, 3)
            cv2.putText(
                annotated, 
                "2ND DEF", 
                (dx - 40, dy - 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                self.defender_color, 
                2
            )

        # Draw ball
        if ball_pos:
            bx, by = int(ball_pos[0]), int(ball_pos[1])
            if 0 <= bx < w and 0 <= by < h:
                cv2.circle(annotated, (bx, by), 10, self.ball_color, 2)
                cv2.putText(
                    annotated, 
                    "BALL", 
                    (bx - 20, by - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    self.ball_color, 
                    1
                )

        # Draw decision
        decision_color = (0, 0, 255) if decision == "OFFSIDE" else (0, 255, 0)
        cv2.putText(
            annotated, 
            decision, 
            (20, 50), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1.5, 
            decision_color, 
            3
        )

        # Save
        output_path = Path(output_filename)
        if not output_path.is_absolute():
            output_path = self.output_dir / output_filename

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        return str(output_path)


def annotate_frame(
    image: np.ndarray,
    detection_result,
    offside_result: Optional[OffsideAnalysisResult],
    team1: List[BoundingBox],
    team2: List[BoundingBox],
    output_filename: str = "annotated_frame.jpg"
) -> str:
    """Convenience function to annotate a frame."""
    visualizer = PitchVisualizer()
    return visualizer.annotate_frame(
        image, detection_result, offside_result, team1, team2, output_filename
    )


def annotate_from_llm(
    image: np.ndarray,
    attacker_pos: Tuple[float, float],
    defender_pos: Tuple[float, float],
    ball_pos: Optional[Tuple[float, float]] = None,
    offside_line_x: Optional[float] = None,
    offside_line_top: Optional[Tuple[float, float]] = None,
    offside_line_bottom: Optional[Tuple[float, float]] = None,
    decision: str = "UNKNOWN",
    all_attacking: List[Dict[str, float]] = None,
    all_defending: List[Dict[str, float]] = None,
    output_filename: str = "annotated_frame.jpg"
) -> str:
    """
    Legacy function for backward compatibility.
    Uses annotate_from_positions internally.
    """
    visualizer = PitchVisualizer()
    return visualizer.annotate_from_positions(
        image, attacker_pos, defender_pos, ball_pos, offside_line_x,
        offside_line_top, offside_line_bottom,
        decision, all_attacking, all_defending, output_filename
    )