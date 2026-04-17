from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Tuple
from pathlib import Path
import cv2
import numpy as np
import uuid
import logging

from ..detection import create_detector, BoundingBox
from ..logic import separate_teams, analyze_offside, TeamInfo
from ..utils import (
    extract_team_color_profile,
    extract_jersey_color_profile,
)
from ..visualization import annotate_frame, generate_offside_svg

router = APIRouter()
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
OUTPUT_DIR = BASE_DIR / "output"
ANNOTATED_DIR = OUTPUT_DIR / "annotated"

ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)


class Position(BaseModel):
    x: float
    y: float


class TeamInfoResponse(BaseModel):
    team_id: str
    color_name: str
    color_bgr: tuple
    player_count: int
    color_confidence: Optional[float] = None
    color_warning: Optional[str] = None


class GoalkeeperInfo(BaseModel):
    position: Position
    color_name: str
    color_bgr: tuple
    color_confidence: Optional[float] = None
    color_warning: Optional[str] = None
    team: Optional[str] = None
    source: Optional[str] = None


class OffsideResponse(BaseModel):
    decision: str
    confidence: float
    attacker_position: Position
    defender_position: Position
    attacker_foot: Position
    defender_foot: Position
    explanation: str
    annotated_image_url: Optional[str] = None
    svg_url: Optional[str] = None
    offside_margin_pixels: float
    attacking_team: str
    defending_team: str
    team1_info: Optional[TeamInfoResponse] = None
    team2_info: Optional[TeamInfoResponse] = None


class TeamDetectionResponse(BaseModel):
    team1_info: Optional[TeamInfoResponse] = None
    team2_info: Optional[TeamInfoResponse] = None
    goalkeeper: Optional[GoalkeeperInfo] = None
    player_count: int
    ball_detected: bool


class DetectorState:
    def __init__(self):
        self._detector = None
    
    def get(self):
        if self._detector is None:
            logger.info("Initializing YOLO detector...")
            self._detector = create_detector("uisikdag/yolo-v8-football-players-detection", confidence_threshold=0.25, ball_confidence_threshold=0.01)
        return self._detector


detector_state = DetectorState()

def _compute_team_info_from_players(players: List[BoundingBox], image: np.ndarray):
    if not players:
        return None

    team_profile = extract_team_color_profile(players, image)

    print(
        f"[_compute_team_info] players={len(players)}, color_bgr={team_profile.color_bgr}, "
        f"color_name={team_profile.color_name}, confidence={team_profile.confidence:.2f}"
    )

    return TeamInfo(
        players=list(players),
        color_bgr=team_profile.color_bgr,
        color_name=team_profile.color_name,
        goalkeeper=None,
        color_confidence=team_profile.confidence,
        color_warning=team_profile.warning
    )
def _find_extreme_player(
    team_players: List[BoundingBox],
    goal_direction: str,
    toward_goal: bool
) -> Optional[BoundingBox]:
    if not team_players:
        return None

    if goal_direction == "right":
        return max(team_players, key=lambda player: player.foot_position[0]) if toward_goal else min(team_players, key=lambda player: player.foot_position[0])
    return min(team_players, key=lambda player: player.foot_position[0]) if toward_goal else max(team_players, key=lambda player: player.foot_position[0])


def _is_plausible_defending_goalkeeper(
    candidate: BoundingBox,
    defending_team_players: List[BoundingBox],
    goal_direction: str,
    image_width: int
) -> bool:
    extreme_defender = _find_extreme_player(defending_team_players, goal_direction, toward_goal=True)
    if extreme_defender is None:
        return True

    candidate_x = candidate.foot_position[0]
    extreme_x = extreme_defender.foot_position[0]
    margin = max(image_width * 0.12, 80)

    if goal_direction == "right":
        return candidate_x >= extreme_x - margin
    return candidate_x <= extreme_x + margin


def _is_on_defending_half(
    candidate: BoundingBox,
    goal_direction: str,
    image_width: int
) -> bool:
    candidate_x = candidate.foot_position[0]
    midpoint_x = image_width / 2.0
    if goal_direction == "right":
        return candidate_x >= midpoint_x
    return candidate_x <= midpoint_x


def _goalkeeper_central_gap(
    candidate: BoundingBox,
    team_players: List[BoundingBox],
    team_id: str,
    attacking_team: str,
    goal_direction: str,
    image_width: int
) -> float:
    if not team_players:
        goal_edge = _goal_edge_for_team(team_id, attacking_team, goal_direction)
        goal_x = 0.0 if goal_edge == "left" else float(image_width)
        return abs(candidate.foot_position[0] - goal_x)

    goal_edge = _goal_edge_for_team(team_id, attacking_team, goal_direction)
    extreme_x = min(player.foot_position[0] for player in team_players) if goal_edge == "left" else max(player.foot_position[0] for player in team_players)
    candidate_x = candidate.foot_position[0]

    if goal_edge == "left":
        return max(0.0, candidate_x - extreme_x)
    return max(0.0, extreme_x - candidate_x)


def _is_ball_like_goalkeeper_candidate(
    candidate: Optional[BoundingBox],
    ball: Optional[BoundingBox]
) -> bool:
    if candidate is None or ball is None:
        return False

    center_distance = float(np.linalg.norm(np.array(candidate.center) - np.array(ball.center)))
    foot_distance = float(np.linalg.norm(np.array(candidate.foot_position) - np.array(ball.center)))
    similar_scale = (
        candidate.height <= max(ball.height * 4.0, 36.0)
        and candidate.width <= max(ball.width * 4.0, 36.0)
    )

    if not similar_scale:
        return False

    return (
        center_distance <= max(18.0, ball.width * 2.5)
        or foot_distance <= max(14.0, ball.width * 2.0)
    )


def _goal_edge_for_team(team_id: str, attacking_team: str, goal_direction: str) -> str:
    if team_id == attacking_team:
        return "left" if goal_direction == "right" else "right"
    return goal_direction


def _select_goalkeeper_candidate(
    candidates: List[BoundingBox],
    team_id: str,
    attacking_team: str,
    goal_direction: str
) -> Optional[BoundingBox]:
    valid_candidates = [candidate for candidate in candidates if candidate is not None]
    if not valid_candidates:
        return None

    goal_edge = _goal_edge_for_team(team_id, attacking_team, goal_direction)
    if goal_edge == "left":
        return min(valid_candidates, key=lambda player: player.foot_position[0])
    return max(valid_candidates, key=lambda player: player.foot_position[0])


def _select_ball_for_selected_attack(
    ball_candidates: List[BoundingBox],
    attacking_players: List[BoundingBox],
    defending_goalkeeper: Optional[BoundingBox]
) -> Optional[BoundingBox]:
    if not ball_candidates:
        return None

    if not attacking_players:
        best_ball = max(ball_candidates, key=lambda ball: ball.confidence)
        logger.info(
            f"[Ball] No attacking players available; using highest-confidence candidate "
            f"(source={best_ball.source or 'unknown'}, conf={best_ball.confidence:.2f})"
        )
        return best_ball

    ranked_candidates = []
    soft_ranked_candidates = []

    for ball in ball_candidates:
        ball_anchor = np.array(ball.foot_position)
        closest_attacker = min(
            attacking_players,
            key=lambda player: np.linalg.norm(ball_anchor - np.array(player.foot_position))
        )
        attacker_dist = np.linalg.norm(ball_anchor - np.array(closest_attacker.foot_position))
        horizontal_gap = abs(ball.foot_position[0] - closest_attacker.foot_position[0])
        vertical_gap = abs(ball.foot_position[1] - closest_attacker.foot_position[1])
        attacker_height = max(closest_attacker.height, 1.0)
        square_ratio = min(ball.width, ball.height) / max(ball.width, ball.height, 1.0)
        source = ball.source or ""

        soft_score = ball.confidence
        if source == "football":
            soft_score += 0.04
        elif source.startswith("generic"):
            soft_score += 0.025
        soft_score += square_ratio * 0.05
        if attacker_dist <= max(attacker_height * 0.28, ball.width * 6, 14):
            soft_score += 0.05

        goalkeeper_dist = None
        if defending_goalkeeper is not None:
            goalkeeper_dist = np.linalg.norm(ball_anchor - np.array(defending_goalkeeper.foot_position))

        soft_score -= min(attacker_dist / attacker_height, 2.0) * 0.06
        soft_score -= min(vertical_gap / attacker_height, 2.0) * 0.06
        soft_score -= min(horizontal_gap / attacker_height, 2.0) * 0.03
        if goalkeeper_dist is not None and attacker_height > 0:
            soft_score += min(goalkeeper_dist / attacker_height, 2.0) * 0.02

        soft_ranked_candidates.append((soft_score, ball))

        max_attacker_dist = max(attacker_height * 1.15, ball.width * 18, 54)
        if attacker_dist > max_attacker_dist:
            logger.info(
                f"[Ball] Candidate rejected: too far from attacking player "
                f"(dist={attacker_dist:.1f}, max={max_attacker_dist:.1f}, source={ball.source or 'unknown'})"
            )
            continue

        max_vertical_gap = max(attacker_height * 0.95, 46)
        if vertical_gap > max_vertical_gap:
            logger.info(
                f"[Ball] Candidate rejected: too far vertically from attacking foot "
                f"(gap={vertical_gap:.1f}, max={max_vertical_gap:.1f}, source={ball.source or 'unknown'})"
            )
            continue

        if defending_goalkeeper is not None:
            goalkeeper_margin = max(20, ball.width * 2.5)
            if goalkeeper_dist + goalkeeper_margin < attacker_dist and attacker_dist > max(attacker_height * 0.45, 24):
                logger.info(
                    f"[Ball] Candidate rejected: closer to defending goalkeeper than attacker "
                    f"(gk_dist={goalkeeper_dist:.1f}, attacker_dist={attacker_dist:.1f}, source={ball.source or 'unknown'})"
                )
                continue

        score = soft_score
        if attacker_dist <= max(attacker_height * 0.75, ball.width * 12, 32):
            score += 0.05
        if vertical_gap <= max(attacker_height * 0.55, 28):
            score += 0.03

        ranked_candidates.append((score, ball))

    if ranked_candidates:
        best_score, best_ball = max(ranked_candidates, key=lambda item: item[0])
        logger.info(
            f"[Ball] Selected team-aware candidate: source={best_ball.source or 'unknown'}, "
            f"conf={best_ball.confidence:.2f}, score={best_score:.2f}"
        )
        return best_ball

    if soft_ranked_candidates:
        best_score, best_ball = max(soft_ranked_candidates, key=lambda item: item[0])
        logger.info(
            f"[Ball] Falling back to soft-ranked candidate: source={best_ball.source or 'unknown'}, "
            f"conf={best_ball.confidence:.2f}, score={best_score:.2f}"
        )
        return best_ball

    logger.info("[Ball] No team-aware ball candidate survived final validation")
    return None


def _validate_ball_for_selected_attack(
    ball: Optional[BoundingBox],
    attacking_players: List[BoundingBox],
    defending_goalkeeper: Optional[BoundingBox]
) -> Optional[BoundingBox]:
    if ball is None:
        return None

    return _select_ball_for_selected_attack([ball], attacking_players, defending_goalkeeper)


def _is_distinct_third_color_goalkeeper(candidate: Optional[BoundingBox]) -> bool:
    return candidate is not None and (candidate.source or "").startswith("third-color")


@router.post("/detect-teams", response_model=TeamDetectionResponse)
async def detect_teams(image_file: UploadFile = File(...)):
    """
    Step 1: Upload image -> Detect Teams
    
    Returns team colors for user selection.
    """
    contents = await image_file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")
    
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    detector = detector_state.get()
    detection_result = detector.detect(image)
    
    logger.info(
        f"[Team Detection] Found {len(detection_result.players)} players, "
        f"{len(detection_result.goalkeepers)} goalkeepers, "
        f"{len(detection_result.referees)} referees, "
        f"ball={detection_result.ball is not None}"
    )
    
    if len(detection_result.players) < 2:
        raise HTTPException(status_code=422, detail=f"Need at least 2 players to detect teams. Found: {len(detection_result.players)}")
    
    team1, team2, detected_goalkeeper = separate_teams(detection_result.players, image)

    if not team1 or not team2:
        raise HTTPException(status_code=422, detail="Could not separate teams into two groups")
    
    team1_info = _compute_team_info_from_players(team1, image)
    team2_info = _compute_team_info_from_players(team2, image)
    
    if team1_info is None or team2_info is None:
        raise HTTPException(status_code=422, detail="Could not compute team colors")
    
    logger.info(
        f"[Team Detection] Team 1 (left): {len(team1)} players, color={team1_info.color_name} "
        f"RGB{team1_info.color_bgr}, confidence={team1_info.color_confidence:.2f}"
    )
    logger.info(
        f"[Team Detection] Team 2 (right): {len(team2)} players, color={team2_info.color_name} "
        f"RGB{team2_info.color_bgr}, confidence={team2_info.color_confidence:.2f}"
    )
    
    goalkeeper_info = None
    if detected_goalkeeper:
        logger.info(
            f"[Team Detection] Singleton third-color player isolated from team colors: "
            f"foot_pos={detected_goalkeeper.foot_position}"
        )
        gk_profile = extract_jersey_color_profile(detected_goalkeeper, image)
        logger.info(
            f"[Team Detection] Third-color singleton: {gk_profile.color_name} RGB{gk_profile.color_bgr}, "
            f"excluded from both team color palettes"
        )
        
        goalkeeper_info = GoalkeeperInfo(
            position=Position(x=float(detected_goalkeeper.foot_position[0]), y=float(detected_goalkeeper.foot_position[1])),
            color_name=gk_profile.color_name,
            color_bgr=gk_profile.color_bgr,
            color_confidence=gk_profile.confidence,
            color_warning=gk_profile.warning,
            team=None,
            source=detected_goalkeeper.source or "third-color-singleton"
        )
    elif len(detection_result.goalkeepers) == 1:
        detected_yolo_goalkeeper = detection_result.goalkeepers[0]
        gk_profile = extract_jersey_color_profile(detected_yolo_goalkeeper, image)
        logger.info(
            f"[Team Detection] Detector goalkeeper candidate: {gk_profile.color_name} RGB{gk_profile.color_bgr}, "
            f"source={detected_yolo_goalkeeper.source or 'yolo'}"
        )

        goalkeeper_info = GoalkeeperInfo(
            position=Position(
                x=float(detected_yolo_goalkeeper.foot_position[0]),
                y=float(detected_yolo_goalkeeper.foot_position[1])
            ),
            color_name=gk_profile.color_name,
            color_bgr=gk_profile.color_bgr,
            color_confidence=gk_profile.confidence,
            color_warning=gk_profile.warning,
            team=None,
            source=detected_yolo_goalkeeper.source or "yolo"
        )
    
    def convert_color(color):
        if color is None:
            return (0, 0, 0)
        return (int(color[0]), int(color[1]), int(color[2]))
    
    return TeamDetectionResponse(
        team1_info=TeamInfoResponse(
            team_id="team1",
            color_name=str(team1_info.color_name),
            color_bgr=convert_color(team1_info.color_bgr),
            player_count=int(len(team1_info.players)),
            color_confidence=float(team1_info.color_confidence) if team1_info.color_confidence is not None else None,
            color_warning=team1_info.color_warning
        ),
        team2_info=TeamInfoResponse(
            team_id="team2",
            color_name=str(team2_info.color_name),
            color_bgr=convert_color(team2_info.color_bgr),
            player_count=int(len(team2_info.players)),
            color_confidence=float(team2_info.color_confidence) if team2_info.color_confidence is not None else None,
            color_warning=team2_info.color_warning
        ),
        goalkeeper=goalkeeper_info,
        player_count=int(len(detection_result.players)),
        ball_detected=bool(detection_result.ball is not None)
    )


@router.post("/analyze-offside", response_model=OffsideResponse)
async def analyze_offside_endpoint(
    image_file: UploadFile = File(...),
    attacking_team: str = Form(...),
    goal_direction: str = Form("right")
):
    """
    Step 3: Click "Analyze Offside" -> Full analysis with user's selection
    
    User provides:
    - attacking_team: "team1" or "team2"
    - goal_direction: "left" or "right" (direction of the goal being attacked)
    """
    if attacking_team not in ["team1", "team2"]:
        raise HTTPException(status_code=400, detail="attacking_team must be 'team1' or 'team2'")
    
    if goal_direction not in ["left", "right"]:
        raise HTTPException(status_code=400, detail="goal_direction must be 'left' or 'right'")
    
    contents = await image_file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")
    
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    h, w = image.shape[:2]
    logger.info(
        f"[Analysis] Image shape: {image.shape}, attacking_team={attacking_team}, "
        f"goal_direction={goal_direction}"
    )

    detector = detector_state.get()
    detection_result = detector.detect(image)

    logger.info(
        f"[Detection] Found {len(detection_result.players)} players, "
        f"{len(detection_result.goalkeepers)} goalkeepers, "
        f"{len(detection_result.referees)} referees, "
        f"ball={detection_result.ball is not None}"
    )

    if len(detection_result.players) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 players for offside analysis")

    team1, team2, detected_goalkeeper = separate_teams(detection_result.players, image)

    logger.info(f"[Teams] Team 1: {len(team1)} players, Team 2: {len(team2)} players")

    if not team1 or not team2:
        raise HTTPException(status_code=422, detail="Could not separate teams")

    team1_info = _compute_team_info_from_players(team1, image)
    team2_info = _compute_team_info_from_players(team2, image)

    if len(detection_result.players) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 players for offside analysis")

    if not team1 or not team2:
        raise HTTPException(status_code=422, detail="Could not separate teams")
    
    if team1_info:
        logger.info(
            f"[Teams] Team1 color: {team1_info.color_name} RGB{team1_info.color_bgr}, "
            f"confidence={team1_info.color_confidence:.2f}"
        )
    if team2_info:
        logger.info(
            f"[Teams] Team2 color: {team2_info.color_name} RGB{team2_info.color_bgr}, "
            f"confidence={team2_info.color_confidence:.2f}"
        )
    
    team1_goalkeeper = None
    team2_goalkeeper = None

    if detection_result.goalkeepers and team1_info and team2_info:
        team_goalkeeper_candidates = {"team1": [], "team2": []}
        plausibility_margin = max(w * 0.12, 80)
        for gk in detection_result.goalkeepers:
            team1_gap = _goalkeeper_central_gap(gk, team1, "team1", attacking_team, goal_direction, w)
            team2_gap = _goalkeeper_central_gap(gk, team2, "team2", attacking_team, goal_direction, w)
            team1_plausible = team1_gap <= plausibility_margin
            team2_plausible = team2_gap <= plausibility_margin

            if not team1_plausible and not team2_plausible:
                logger.info(
                    f"[Goalkeeper] Rejected YOLO goalkeeper candidate as implausible for both teams "
                    f"(team1_gap={team1_gap:.1f}, team2_gap={team2_gap:.1f})"
                )
                continue

            if team1_plausible and not team2_plausible:
                team_goalkeeper_candidates["team1"].append(gk)
                logger.info(f"[Goalkeeper] Assigned to team1 (own-goal gap: {team1_gap:.1f})")
            elif team2_plausible and not team1_plausible:
                team_goalkeeper_candidates["team2"].append(gk)
                logger.info(f"[Goalkeeper] Assigned to team2 (own-goal gap: {team2_gap:.1f})")
            elif team1_gap <= team2_gap:
                team_goalkeeper_candidates["team1"].append(gk)
                logger.info(
                    f"[Goalkeeper] Assigned to team1 by lower own-goal gap "
                    f"(team1_gap={team1_gap:.1f}, team2_gap={team2_gap:.1f})"
                )
            else:
                team_goalkeeper_candidates["team2"].append(gk)
                logger.info(
                    f"[Goalkeeper] Assigned to team2 by lower own-goal gap "
                    f"(team1_gap={team1_gap:.1f}, team2_gap={team2_gap:.1f})"
                )
        team1_goalkeeper = _select_goalkeeper_candidate(
            team_goalkeeper_candidates["team1"],
            "team1",
            attacking_team,
            goal_direction
        )
        team2_goalkeeper = _select_goalkeeper_candidate(
            team_goalkeeper_candidates["team2"],
            "team2",
            attacking_team,
            goal_direction
        )
        if not team1_goalkeeper and not team2_goalkeeper and len(detection_result.goalkeepers) >= 2:
            team1_goalkeeper = detection_result.goalkeepers[0]
            team2_goalkeeper = detection_result.goalkeepers[1]
            logger.info(f"[Goalkeeper] Using first two goalkeepers for teams")
    elif detection_result.goalkeepers:
        logger.warning(f"[Goalkeeper] Could not match goalkeepers to teams (team1_info or team2_info missing)")
        if len(detection_result.goalkeepers) >= 2:
            team1_goalkeeper = detection_result.goalkeepers[0]
            team2_goalkeeper = detection_result.goalkeepers[1]
    distinct_goalkeeper = detected_goalkeeper
    if distinct_goalkeeper is None:
        distinct_goalkeeper = next(
            (goalkeeper for goalkeeper in detection_result.goalkeepers if _is_distinct_third_color_goalkeeper(goalkeeper)),
            None
        )

    if _is_distinct_third_color_goalkeeper(distinct_goalkeeper):
        logger.info(
            f"[Goalkeeper] Distinct third-color player auto-assigned to defending team: "
            f"foot_pos={distinct_goalkeeper.foot_position}, attacking_team={attacking_team}"
        )
        if attacking_team == "team1":
            team2_goalkeeper = distinct_goalkeeper
        else:
            team1_goalkeeper = distinct_goalkeeper
    
    if team1_goalkeeper is None:
        team1_goalkeeper = _find_extreme_player(
            team1,
            goal_direction,
            toward_goal=(attacking_team != "team1")
        )
    if team2_goalkeeper is None:
        team2_goalkeeper = _find_extreme_player(
            team2,
            goal_direction,
            toward_goal=(attacking_team != "team2")
        )

    if attacking_team == "team1":
        attacking_players = [player for player in team1 if player != team1_goalkeeper]
        defending_goalkeeper = team2_goalkeeper
    else:
        attacking_players = [player for player in team2 if player != team2_goalkeeper]
        defending_goalkeeper = team1_goalkeeper

    if not attacking_players:
        attacking_players = team1 if attacking_team == "team1" else team2

    ball_candidates = list(getattr(detection_result, "ball_candidates", []))
    if detection_result.ball and all(detection_result.ball != candidate for candidate in ball_candidates):
        ball_candidates.append(detection_result.ball)
    detection_result.ball = _select_ball_for_selected_attack(
        ball_candidates,
        attacking_players,
        defending_goalkeeper
    )

    if attacking_team == "team1":
        if _is_ball_like_goalkeeper_candidate(team2_goalkeeper, detection_result.ball):
            logger.info("[Goalkeeper] Defending goalkeeper overlaps selected ball; falling back to extreme defending player")
            team2_goalkeeper = _find_extreme_player(team2, goal_direction, toward_goal=True)
            defending_goalkeeper = team2_goalkeeper
            detection_result.ball = _select_ball_for_selected_attack(
                ball_candidates,
                attacking_players,
                defending_goalkeeper
            )
    else:
        if _is_ball_like_goalkeeper_candidate(team1_goalkeeper, detection_result.ball):
            logger.info("[Goalkeeper] Defending goalkeeper overlaps selected ball; falling back to extreme defending player")
            team1_goalkeeper = _find_extreme_player(team1, goal_direction, toward_goal=True)
            defending_goalkeeper = team1_goalkeeper
            detection_result.ball = _select_ball_for_selected_attack(
                ball_candidates,
                attacking_players,
                defending_goalkeeper
            )

    ball_pos = detection_result.ball.foot_position if detection_result.ball else None
    if ball_pos:
        logger.info(f"[Ball] Ball position: ({ball_pos[0]:.0f}, {ball_pos[1]:.0f})")
    
    offside_result = analyze_offside(
        team1=team1,
        team2=team2,
        ball_position=ball_pos,
        attacking_team_input=attacking_team,
        team1_info=team1_info,
        team2_info=team2_info,
        goal_direction=goal_direction,
        team1_goalkeeper=team1_goalkeeper,
        team2_goalkeeper=team2_goalkeeper
    )
    
    file_id = str(uuid.uuid4())[:8]
    
    annotated_filename = f"annotated_{file_id}.jpg"
    annotated_path = ANNOTATED_DIR / annotated_filename
    
    result_path = annotate_frame(
        image, detection_result, offside_result, team1, team2,
        output_filename=str(annotated_path),
        attacking_team=offside_result.attacking_team
    )
    logger.info(f"[Output] Annotated image saved: {result_path}")
    annotated_url = f"/annotated/{annotated_filename}"
    
    svg_filename = f"pitch_{file_id}_{goal_direction}_{int(uuid.uuid4().time)}.svg"
    svg_path = OUTPUT_DIR / svg_filename
    
    scale_x_svg = 1050 / w
    scale_y_svg = 680 / h
    
    attacker_foot = offside_result.attacker.foot_position
    defender_foot = offside_result.second_last_defender.foot_position
    goalkeeper_foot = offside_result.goalkeeper.foot_position if offside_result.goalkeeper else (0, 0)
    
    attacker_pos = (attacker_foot[0] * scale_x_svg, attacker_foot[1] * scale_y_svg)
    defender_pos = (defender_foot[0] * scale_x_svg, defender_foot[1] * scale_y_svg)
    goalkeeper_pos = (goalkeeper_foot[0] * scale_x_svg, goalkeeper_foot[1] * scale_y_svg)
    
    excluded_positions = [attacker_foot, defender_foot, goalkeeper_foot]
    team1_positions = []
    team2_positions = []
    for p in team1:
        if p.foot_position not in excluded_positions:
            team1_positions.append((p.foot_position[0] * scale_x_svg, p.foot_position[1] * scale_y_svg))
    for p in team2:
        if p.foot_position not in excluded_positions:
            team2_positions.append((p.foot_position[0] * scale_x_svg, p.foot_position[1] * scale_y_svg))
    
    ball_pos_scaled = None
    if ball_pos is not None:
        ball_pos_scaled = (ball_pos[0] * scale_x_svg, ball_pos[1] * scale_y_svg)
    
    offside_line_x_val = defender_pos[0]
    
    svg_content = generate_offside_svg(
        attacker_pos=attacker_pos,
        defender_pos=defender_pos,
        goalkeeper_pos=goalkeeper_pos,
        ball_pos=ball_pos_scaled,
        offside_line_x=offside_line_x_val,
        offside_line_top=None,
        offside_line_bottom=None,
        decision=offside_result.decision,
        team1_positions=team1_positions if team1_positions else None,
        team2_positions=team2_positions if team2_positions else None,
        image_width=w,
        image_height=h,
        output_path=str(svg_path),
        attacking_team=attacking_team,
        goal_direction=goal_direction
    )
    logger.info(f"[Output] SVG saved: {svg_path}")
    svg_url = f"/pitch/{svg_filename}"
    
    def safe_float(val):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return 0.0
        return val
    
    def convert_color(color):
        if color is None:
            return (0, 0, 0)
        return (int(color[0]), int(color[1]), int(color[2]))
    
    structured_data = {
        "decision": offside_result.decision,
        "attacker_position": {"x": safe_float(offside_result.attacker.center[0]), "y": safe_float(offside_result.attacker.center[1])},
        "defender_position": {"x": safe_float(offside_result.second_last_defender.center[0]), "y": safe_float(offside_result.second_last_defender.center[1])},
        "attacker_foot": {"x": safe_float(offside_result.attacker.foot_position[0]), "y": safe_float(offside_result.attacker.foot_position[1])},
        "defender_foot": {"x": safe_float(offside_result.second_last_defender.foot_position[0]), "y": safe_float(offside_result.second_last_defender.foot_position[1])},
        "confidence": safe_float(offside_result.confidence),
        "offside_margin_pixels": safe_float(offside_result.offside_margin_pixels),
        "attacking_team": offside_result.attacking_team,
        "defending_team": offside_result.defending_team,
        "team1_info": {
            "team_id": "team1",
            "color_name": str(team1_info.color_name) if team1_info else "Unknown",
            "color_bgr": convert_color(team1_info.color_bgr) if team1_info else (0, 0, 0),
            "player_count": int(len(team1)),
            "color_confidence": safe_float(team1_info.color_confidence) if team1_info else None,
            "color_warning": team1_info.color_warning if team1_info else None
        } if team1_info else None,
        "team2_info": {
            "team_id": "team2",
            "color_name": str(team2_info.color_name) if team2_info else "Unknown",
            "color_bgr": convert_color(team2_info.color_bgr) if team2_info else (0, 0, 0),
            "player_count": int(len(team2)),
            "color_confidence": safe_float(team2_info.color_confidence) if team2_info else None,
            "color_warning": team2_info.color_warning if team2_info else None
        } if team2_info else None
    }
    
    explanation = f"{structured_data['decision']} - Attacker ahead by {structured_data['offside_margin_pixels']:.1f} pixels" if structured_data['decision'] != "UNKNOWN" else "Analysis could not be completed"
    
    return OffsideResponse(
        decision=structured_data["decision"],
        confidence=structured_data["confidence"],
        attacker_position=Position(**structured_data["attacker_position"]),
        defender_position=Position(**structured_data["defender_position"]),
        attacker_foot=Position(**structured_data["attacker_foot"]),
        defender_foot=Position(**structured_data["defender_foot"]),
        explanation=explanation,
        annotated_image_url=annotated_url,
        svg_url=svg_url,
        offside_margin_pixels=structured_data["offside_margin_pixels"],
        attacking_team=structured_data["attacking_team"],
        defending_team=structured_data["defending_team"],
        team1_info=TeamInfoResponse(**structured_data["team1_info"]) if structured_data.get("team1_info") else None,
        team2_info=TeamInfoResponse(**structured_data["team2_info"]) if structured_data.get("team2_info") else None
    )


@router.post("/generate-visual")
async def generate_visual(
    attacker_x: float = Form(...),
    attacker_y: float = Form(...),
    defender_x: float = Form(...),
    defender_y: float = Form(...),
    decision: str = Form(...),
    ball_x: Optional[float] = Form(None),
    ball_y: Optional[float] = Form(None),
    goal_direction: str = Form("right"),
    attacking_team: str = Form("team1")
):
    """Generate SVG visualization from positions."""
    svg_content = generate_offside_svg(
        attacker_pos=(attacker_x, attacker_y),
        defender_pos=(defender_x, defender_y),
        goalkeeper_pos=(0, 0),
        ball_pos=(ball_x, ball_y) if ball_x and ball_y else None,
        offside_line_x=defender_x,
        decision=decision,
        attacking_team=attacking_team,
        goal_direction=goal_direction
    )
    
    return {"svg_content": svg_content}
