import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

from ..detection import BoundingBox
from ..logic import OffsideAnalysisResult, GoalCheckResult


def _write_image_or_raise(output_path: Path, image: np.ndarray) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise OSError(f"Failed to write annotated image to {output_path}")
    return str(output_path)


class PitchVisualizer:
    def __init__(
        self,
        output_dir: str = "output/annotated",
        bbox_thickness: int = 2,
        attacker_color: Tuple[int, int, int] = (255, 0, 0),
        defender_color: Tuple[int, int, int] = (0, 0, 255),
        goalkeeper_color: Tuple[int, int, int] = (0, 165, 255),
        referee_color: Tuple[int, int, int] = (192, 192, 192),
        ball_color: Tuple[int, int, int] = (0, 255, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.bbox_thickness = bbox_thickness
        self.attacker_color = attacker_color
        self.defender_color = defender_color
        self.goalkeeper_color = goalkeeper_color
        self.referee_color = referee_color
        self.ball_color = ball_color
        self.text_color = text_color

    def annotate_frame(
        self,
        image: np.ndarray,
        detection_result,
        offside_result: OffsideAnalysisResult,
        team1: List[BoundingBox],
        team2: List[BoundingBox],
        output_filename: str = "annotated_frame.jpg",
        attacking_team: str = "team1"
    ) -> str:
        annotated = image.copy()
        h, w = annotated.shape[:2]

        for player in detection_result.players:
            self._draw_player_bbox(annotated, player, team1, team2, attacking_team)

        if offside_result and offside_result.goalkeeper:
            self._draw_special_bbox(annotated, offside_result.goalkeeper, self.goalkeeper_color, "GK")

        for referee in getattr(detection_result, "referees", []):
            self._draw_special_bbox(annotated, referee, self.referee_color, "REF")

        if detection_result.ball:
            self._draw_ball_bbox(annotated, detection_result.ball)

        if offside_result and offside_result.decision != "UNKNOWN":
            self._draw_offside_analysis(annotated, offside_result, h, w)

        output_path = Path(output_filename)
        if not output_path.is_absolute():
            output_path = self.output_dir / output_filename

        return _write_image_or_raise(output_path, annotated)

    def annotate_goal_check(
        self,
        image: np.ndarray,
        detection_result,
        goal_result: GoalCheckResult,
        output_filename: str = "goal_check.jpg"
    ) -> str:
        annotated = image.copy()
        image_height, _ = annotated.shape[:2]

        if goal_result.ball is not None:
            self._draw_ball_bbox(annotated, goal_result.ball)

        if goal_result.goal_line_x is not None:
            line_x = int(round(goal_result.goal_line_x))
            line_color = (0, 255, 255) if goal_result.goal_line_source == "goalpost+line-assumption" else (0, 200, 255)
            cv2.line(annotated, (line_x, 0), (line_x, image_height - 1), line_color, 3)
            cv2.putText(
                annotated,
                "GOAL LINE",
                (max(10, line_x - 70), 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                line_color,
                2
            )

        if goal_result.goalpost_x is not None:
            post_x = int(round(goal_result.goalpost_x))
            post_color = (255, 255, 255)
            cv2.line(annotated, (post_x, 0), (post_x, image_height - 1), post_color, 2)
            cv2.putText(
                annotated,
                "POST",
                (max(10, post_x - 30), min(image_height - 12, 56)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                post_color,
                2
            )

        decision_color = (0, 255, 0)
        if goal_result.decision == "NO GOAL":
            decision_color = (0, 0, 255)
        elif goal_result.decision == "UNKNOWN":
            decision_color = (0, 215, 255)

        cv2.putText(
            annotated,
            goal_result.decision,
            (20, 46),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.4,
            decision_color,
            3
        )
        cv2.putText(
            annotated,
            f"Conf: {goal_result.confidence:.2f}",
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.text_color,
            2
        )
        cv2.putText(
            annotated,
            f"Checked goal: {goal_result.goal_direction.upper()}",
            (20, 108),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.text_color,
            2
        )
        cv2.putText(
            annotated,
            f"Margin: {goal_result.goal_margin_pixels:+.1f}px",
            (20, 138),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.text_color,
            2
        )
        cv2.putText(
            annotated,
            f"Geometry: {goal_result.goal_line_source}",
            (20, 168),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            self.text_color,
            2
        )

        output_path = Path(output_filename)
        if not output_path.is_absolute():
            output_path = self.output_dir / output_filename

        return _write_image_or_raise(output_path, annotated)

    def _draw_player_bbox(
        self,
        image: np.ndarray,
        player: BoundingBox,
        team1: List[BoundingBox],
        team2: List[BoundingBox],
        attacking_team: str = "team1"
    ):
        is_attacking = (attacking_team == "team1" and player in team1) or (attacking_team == "team2" and player in team2)
        
        if is_attacking:
            color = self.attacker_color
            label = "ATT"
        elif player in team1 or player in team2:
            color = self.defender_color
            label = "DEF"
        else:
            color = (128, 128, 128)
            label = "?"

        x1, y1, x2, y2 = int(player.x1), int(player.y1), int(player.x2), int(player.y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.bbox_thickness)

        cv2.putText(
            image, 
            label, 
            (x1, y1 - 5), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.5, 
            color, 
            1
        )

        foot_x, foot_y = int(player.foot_position[0]), int(player.foot_position[1])
        cv2.circle(image, (foot_x, foot_y), 5, color, -1)

    def _draw_special_bbox(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        color: Tuple[int, int, int],
        label: str
    ):
        x1, y1, x2, y2 = int(bbox.x1), int(bbox.y1), int(bbox.x2), int(bbox.y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.bbox_thickness)
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1
        )
        foot_x, foot_y = int(bbox.foot_position[0]), int(bbox.foot_position[1])
        cv2.circle(image, (foot_x, foot_y), 5, color, -1)

    def _draw_ball_bbox(self, image: np.ndarray, ball: BoundingBox):
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

        decision = offside_result.decision
        confidence = offside_result.confidence
        margin = offside_result.offside_margin_pixels
        
        text_y = 50
        
        if decision == "OFFSIDE":
            decision_color = (0, 0, 255)
        else:
            decision_color = (0, 255, 0)
            
        cv2.putText(
            image,
            f"{decision}",
            (20, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            decision_color,
            3
        )
        
        cv2.putText(
            image,
            f"Conf: {confidence:.2f}",
            (20, text_y + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            self.text_color,
            2
        )
        
        margin_text = f"Margin: {margin:.1f}px"
        if decision == "OFFSIDE":
            margin_text = f"+{margin:.1f}px offside"
        elif decision == "ONSIDE":
            margin_text = f"-{margin:.1f}px onside"
            
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
        annotated = image.copy()
        h, w = annotated.shape[:2]

        if all_attacking:
            for p in all_attacking:
                px, py = int(p["x"]), int(p["y"])
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(annotated, (px, py), 15, self.attacker_color, 2)
                    cv2.putText(annotated, "ATT", (px - 15, py - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.attacker_color, 1)

        if all_defending:
            for p in all_defending:
                px, py = int(p["x"]), int(p["y"])
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(annotated, (px, py), 15, self.defender_color, 2)
                    cv2.putText(annotated, "DEF", (px - 15, py - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.defender_color, 1)

        ax, ay = int(attacker_pos[0]), int(attacker_pos[1])
        if 0 <= ax < w and 0 <= ay < h:
            cv2.circle(annotated, (ax, ay), 20, self.attacker_color, 3)
            cv2.putText(annotated, "ATTACKER", (ax - 40, ay - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.attacker_color, 2)

        dx, dy = int(defender_pos[0]), int(defender_pos[1])
        if 0 <= dx < w and 0 <= dy < h:
            cv2.circle(annotated, (dx, dy), 20, self.defender_color, 3)
            cv2.putText(annotated, "2ND DEF", (dx - 40, dy - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.defender_color, 2)

        if ball_pos:
            bx, by = int(ball_pos[0]), int(ball_pos[1])
            if 0 <= bx < w and 0 <= by < h:
                cv2.circle(annotated, (bx, by), 10, self.ball_color, 2)
                cv2.putText(annotated, "BALL", (bx - 20, by - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ball_color, 1)

        decision_color = (0, 0, 255) if decision == "OFFSIDE" else (0, 255, 0)
        cv2.putText(annotated, decision, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, decision_color, 3)

        output_path = Path(output_filename)
        if not output_path.is_absolute():
            output_path = self.output_dir / output_filename

        return _write_image_or_raise(output_path, annotated)


def annotate_frame(
    image: np.ndarray,
    detection_result,
    offside_result: Optional[OffsideAnalysisResult],
    team1: List[BoundingBox],
    team2: List[BoundingBox],
    output_filename: str = "annotated_frame.jpg",
    attacking_team: str = "team1"
) -> str:
    visualizer = PitchVisualizer()
    return visualizer.annotate_frame(
        image, detection_result, offside_result, team1, team2, output_filename, attacking_team
    )


def annotate_goal_check(
    image: np.ndarray,
    detection_result,
    goal_result: GoalCheckResult,
    output_filename: str = "goal_check.jpg"
) -> str:
    visualizer = PitchVisualizer()
    return visualizer.annotate_goal_check(
        image, detection_result, goal_result, output_filename
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
    visualizer = PitchVisualizer()
    return visualizer.annotate_from_positions(
        image, attacker_pos, defender_pos, ball_pos, offside_line_x,
        offside_line_top, offside_line_bottom,
        decision, all_attacking, all_defending, output_filename
    )
