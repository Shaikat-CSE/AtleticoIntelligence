from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ..detection import BoundingBox, DetectionResult


@dataclass
class GoalCheckResult:
    decision: str
    confidence: float
    explanation: str
    goal_direction: str
    goal_line_x: Optional[float]
    goal_line_confidence: float
    goal_line_source: str
    goalpost_x: Optional[float]
    goalpost_confidence: float
    goalpost_source: str
    goal_margin_pixels: float
    ball: Optional[BoundingBox]


class GoalLineAnalyzer:
    def analyze(
        self,
        image: np.ndarray,
        detection_result: DetectionResult,
        goal_direction: str
    ) -> GoalCheckResult:
        if goal_direction not in {"left", "right"}:
            raise ValueError("goal_direction must be 'left' or 'right'")

        image_height, image_width = image.shape[:2]
        ball = self._select_ball_for_goal_side(
            list(getattr(detection_result, "ball_candidates", [])),
            detection_result.ball,
            goal_direction,
            image_width
        )

        if ball is None:
            return GoalCheckResult(
                decision="UNKNOWN",
                confidence=0.12,
                explanation="Ball could not be detected, so goal-line review is unavailable for this frame.",
                goal_direction=goal_direction,
                goal_line_x=None,
                goal_line_confidence=0.0,
                goal_line_source="unavailable",
                goalpost_x=None,
                goalpost_confidence=0.0,
                goalpost_source="unavailable",
                goal_margin_pixels=0.0,
                ball=None
            )

        goal_line_x, goal_line_confidence, goal_line_source, goalpost_x, goalpost_confidence, goalpost_source = self._estimate_goal_line_x(
            image,
            goal_direction,
            ball
        )

        signed_margin = self._goal_margin_pixels(ball, goal_line_x, goal_direction)
        ball_size = max(ball.width, ball.height, 1.0)
        decision_tolerance = max(2.0, ball_size * 0.08)

        if abs(signed_margin) < decision_tolerance and goal_line_confidence < 0.55:
            decision = "UNKNOWN"
        else:
            decision = "GOAL" if signed_margin >= 0 else "NO GOAL"

        confidence = self._decision_confidence(
            ball_confidence=ball.confidence,
            line_confidence=goal_line_confidence,
            signed_margin=signed_margin,
            ball_size=ball_size,
            decision=decision,
            goal_line_source=goal_line_source
        )

        explanation = self._build_explanation(
            decision=decision,
            signed_margin=signed_margin,
            goal_direction=goal_direction,
            goal_line_source=goal_line_source,
            goal_line_confidence=goal_line_confidence
        )

        return GoalCheckResult(
            decision=decision,
            confidence=confidence,
            explanation=explanation,
            goal_direction=goal_direction,
            goal_line_x=goal_line_x,
            goal_line_confidence=goal_line_confidence,
            goal_line_source=goal_line_source,
            goalpost_x=goalpost_x,
            goalpost_confidence=goalpost_confidence,
            goalpost_source=goalpost_source,
            goal_margin_pixels=signed_margin,
            ball=ball
        )

    def _select_ball_for_goal_side(
        self,
        candidates: List[BoundingBox],
        selected_ball: Optional[BoundingBox],
        goal_direction: str,
        image_width: int
    ) -> Optional[BoundingBox]:
        working_candidates = list(candidates)
        if selected_ball is not None and all(selected_ball != candidate for candidate in working_candidates):
            working_candidates.append(selected_ball)

        if not working_candidates:
            return selected_ball

        def score(candidate: BoundingBox) -> float:
            source = candidate.source or ""
            side_bias = (
                1.0 - (candidate.center[0] / max(image_width, 1))
                if goal_direction == "left"
                else candidate.center[0] / max(image_width, 1)
            )
            ball_size = max(candidate.width, candidate.height, 1.0)
            score_value = candidate.confidence
            if source == "football":
                score_value += 0.04
            elif source.startswith("generic"):
                score_value += 0.02
            score_value += side_bias * 0.08
            if ball_size >= 8:
                score_value += 0.03
            return score_value

        return max(working_candidates, key=score)

    def _estimate_goal_line_x(
        self,
        image: np.ndarray,
        goal_direction: str,
        ball: BoundingBox
    ) -> Tuple[float, float, str, Optional[float], float, str]:
        _, image_width = image.shape[:2]
        white_mask = self._build_goal_side_white_mask(image, goal_direction, ball)
        goalpost = self._detect_goalpost_candidate(white_mask, goal_direction, ball)
        candidate_lines = self._detect_goal_line_candidates(white_mask, goal_direction, ball)

        if goalpost is not None:
            goalpost_x, goalpost_confidence, goalpost_width = goalpost
            line_from_post = self._goal_line_from_goalpost(goalpost_x, goalpost_width, goal_direction)

            if candidate_lines:
                compatible_lines = [
                    (line_x, score)
                    for line_x, score in candidate_lines
                    if abs(line_x - line_from_post) <= max(ball.width * 3.0, 18.0)
                ]
                if compatible_lines:
                    weights = np.array(
                        [goalpost_confidence * 1.35] + [score for _, score in compatible_lines],
                        dtype=np.float32
                    )
                    xs = np.array(
                        [line_from_post] + [line_x for line_x, _ in compatible_lines],
                        dtype=np.float32
                    )
                    goal_line_x = float(np.average(xs, weights=weights))
                    line_confidence = float(np.clip(
                        goalpost_confidence * 0.55 + compatible_lines[0][1] / 1.35 * 0.45,
                        0.35,
                        0.96
                    ))
                    return (
                        goal_line_x,
                        line_confidence,
                        "goalpost+line-assumption",
                        float(goalpost_x),
                        float(goalpost_confidence),
                        "goalpost-detection"
                    )

            return (
                float(line_from_post),
                float(np.clip(goalpost_confidence * 0.82, 0.30, 0.90)),
                "goalpost-assumption",
                float(goalpost_x),
                float(goalpost_confidence),
                "goalpost-detection"
            )

        if candidate_lines:
            top_score = candidate_lines[0][1]
            kept_lines = [
                (line_x, score)
                for line_x, score in candidate_lines
                if score >= top_score * 0.82
            ]
            weights = np.array([score for _, score in kept_lines], dtype=np.float32)
            xs = np.array([line_x for line_x, _ in kept_lines], dtype=np.float32)
            goal_line_x = float(np.average(xs, weights=weights))
            line_confidence = float(np.clip(top_score / 1.35, 0.30, 0.95))
            return goal_line_x, line_confidence, "line-assumption", None, 0.0, "unavailable"

        fallback_x = self._fallback_goal_line_x(goal_direction, image_width)
        return fallback_x, 0.22, "side-assumption", None, 0.0, "unavailable"

    def _build_goal_side_white_mask(
        self,
        image: np.ndarray,
        goal_direction: str,
        ball: BoundingBox
    ) -> np.ndarray:
        _, image_width = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        white_mask = (
            (hsv[:, :, 1] <= 85) &
            (hsv[:, :, 2] >= 145)
        ).astype(np.uint8) * 255

        grass_mask = (
            (hsv[:, :, 0] >= 32) &
            (hsv[:, :, 0] <= 96) &
            (hsv[:, :, 1] >= 35) &
            (hsv[:, :, 2] >= 28)
        )
        white_mask[grass_mask] = 0

        if goal_direction == "left":
            search_limit = int(min(max(ball.center[0] + image_width * 0.10, image_width * 0.24), image_width * 0.56))
            side_mask = np.zeros_like(white_mask)
            side_mask[:, :max(search_limit, 1)] = 255
        else:
            search_start = int(max(min(ball.center[0] - image_width * 0.10, image_width * 0.76), image_width * 0.44))
            side_mask = np.zeros_like(white_mask)
            side_mask[:, min(search_start, image_width - 1):] = 255

        mask = cv2.bitwise_and(white_mask, side_mask)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    def _detect_goal_line_candidates(
        self,
        white_mask: np.ndarray,
        goal_direction: str,
        ball: BoundingBox
    ) -> List[Tuple[float, float]]:
        image_height, image_width = white_mask.shape[:2]

        edges = cv2.Canny(white_mask, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            threshold=40,
            minLineLength=max(30, int(image_height * 0.14)),
            maxLineGap=max(10, int(image_height * 0.05))
        )

        if lines is None:
            return []

        candidates: List[Tuple[float, float]] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx = abs(x2 - x1)
            dy = abs(y2 - y1)
            length = float(np.hypot(dx, dy))
            if length < max(30.0, image_height * 0.14):
                continue
            if dy < max(22.0, image_height * 0.10):
                continue

            verticality = dy / max(length, 1.0)
            if verticality < 0.82:
                continue

            line_x = (x1 + x2) / 2.0
            if goal_direction == "left" and line_x > image_width * 0.5:
                continue
            if goal_direction == "right" and line_x < image_width * 0.5:
                continue

            side_score = (
                1.0 - min(line_x / max(image_width * 0.45, 1.0), 1.0)
                if goal_direction == "left"
                else min((line_x - image_width * 0.55) / max(image_width * 0.45, 1.0), 1.0)
            )

            ball_distance = abs(line_x - ball.center[0])
            ball_proximity = max(0.0, 1.0 - ball_distance / max(image_width * 0.22, 1.0))

            ball_side_bonus = 0.0
            if goal_direction == "left" and line_x <= ball.center[0] + max(ball.width * 0.75, 4.0):
                ball_side_bonus = 0.15
            if goal_direction == "right" and line_x >= ball.center[0] - max(ball.width * 0.75, 4.0):
                ball_side_bonus = 0.15

            score = (
                (length / max(image_height, 1.0)) * 0.70
                + verticality * 0.35
                + side_score * 0.20
                + ball_proximity * 0.18
                + ball_side_bonus
            )
            candidates.append((float(line_x), float(score)))

        candidates.sort(key=lambda item: item[1], reverse=True)
        return candidates

    def _detect_goalpost_candidate(
        self,
        white_mask: np.ndarray,
        goal_direction: str,
        ball: BoundingBox
    ) -> Optional[Tuple[float, float, float]]:
        image_height, image_width = white_mask.shape[:2]
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(white_mask, connectivity=8)

        best_candidate = None
        best_score = -1.0

        for label_idx in range(1, component_count):
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            area = int(stats[label_idx, cv2.CC_STAT_AREA])

            if area < 35 or width <= 0 or height <= 0:
                continue

            aspect_ratio = height / max(width, 1.0)
            density = area / max(width * height, 1.0)
            if height < max(26.0, image_height * 0.10):
                continue
            if aspect_ratio < 1.8:
                continue
            if density < 0.16:
                continue

            x_center = x + width / 2.0
            if goal_direction == "left":
                if x_center > image_width * 0.56:
                    continue
                side_score = 1.0 - min(x_center / max(image_width * 0.50, 1.0), 1.0)
                ball_relation = max(0.0, 1.0 - max(x_center - ball.center[0], 0.0) / max(image_width * 0.18, 1.0))
            else:
                if x_center < image_width * 0.44:
                    continue
                side_score = min((x_center - image_width * 0.44) / max(image_width * 0.50, 1.0), 1.0)
                ball_relation = max(0.0, 1.0 - max(ball.center[0] - x_center, 0.0) / max(image_width * 0.18, 1.0))

            top_extension = max(0.0, 1.0 - y / max(image_height * 0.75, 1.0))
            normalized_height = min(height / max(image_height * 0.40, 1.0), 1.0)
            aspect_score = min(aspect_ratio / 8.0, 1.0)

            score = (
                normalized_height * 0.34
                + aspect_score * 0.16
                + density * 0.16
                + side_score * 0.18
                + ball_relation * 0.10
                + top_extension * 0.06
            )

            if score > best_score:
                best_score = score
                best_candidate = (float(x_center), float(np.clip(score, 0.20, 0.95)), float(width))

        return best_candidate

    def _goal_line_from_goalpost(
        self,
        goalpost_x: float,
        goalpost_width: float,
        goal_direction: str
    ) -> float:
        offset = max(goalpost_width * 0.22, 1.0)
        if goal_direction == "left":
            return float(goalpost_x + offset)
        return float(goalpost_x - offset)

    def _fallback_goal_line_x(
        self,
        goal_direction: str,
        image_width: int
    ) -> float:
        if goal_direction == "left":
            return float(image_width * 0.04)
        return float(image_width * 0.96)

    def _goal_margin_pixels(
        self,
        ball: BoundingBox,
        goal_line_x: float,
        goal_direction: str
    ) -> float:
        if goal_direction == "left":
            return float(goal_line_x - ball.x2)
        return float(ball.x1 - goal_line_x)

    def _decision_confidence(
        self,
        ball_confidence: float,
        line_confidence: float,
        signed_margin: float,
        ball_size: float,
        decision: str,
        goal_line_source: str
    ) -> float:
        margin_factor = min(abs(signed_margin) / max(ball_size, 1.0), 1.5) / 1.5

        if decision == "UNKNOWN":
            confidence = 0.18 + ball_confidence * 0.22 + line_confidence * 0.22
        else:
            confidence = (
                0.28
                + ball_confidence * 0.28
                + line_confidence * 0.24
                + margin_factor * 0.20
            )

        if goal_line_source == "side-assumption":
            confidence *= 0.84

        return float(np.clip(confidence, 0.10, 0.99))

    def _build_explanation(
        self,
        decision: str,
        signed_margin: float,
        goal_direction: str,
        goal_line_source: str,
        goal_line_confidence: float
    ) -> str:
        checked_side = "left" if goal_direction == "left" else "right"

        if decision == "UNKNOWN":
            return (
                f"Goal check is inconclusive for the {checked_side} goal. "
                f"The ball or goal line could not be localized with enough confidence."
            )

        margin_text = f"{abs(signed_margin):.1f}px"
        line_note = (
            "using a detected goalpost plus nearby goal-line markings"
            if goal_line_source == "goalpost+line-assumption"
            else "using the detected goalpost as the goal-line anchor"
            if goal_line_source == "goalpost-assumption"
            else "using detected goal-line markings"
            if goal_line_source == "line-assumption"
            else "using a side-based goal-line assumption near the image edge"
        )

        if decision == "GOAL":
            return (
                f"GOAL for the {checked_side} side. The full ball is estimated to be "
                f"{margin_text} beyond the goal line, {line_note}. "
                f"Goal-line confidence: {goal_line_confidence:.2f}."
            )

        return (
            f"NO GOAL for the {checked_side} side. The ball is estimated to be "
            f"{margin_text} short of fully crossing the goal line, {line_note}. "
            f"Goal-line confidence: {goal_line_confidence:.2f}."
        )


def analyze_goal_check(
    image: np.ndarray,
    detection_result: DetectionResult,
    goal_direction: str
) -> GoalCheckResult:
    analyzer = GoalLineAnalyzer()
    return analyzer.analyze(image, detection_result, goal_direction)
