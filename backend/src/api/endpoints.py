from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import cv2
import numpy as np
import uuid
import logging

from ..detection import create_detector, BoundingBox
from ..logic import separate_teams, analyze_offside, get_team_info, TeamInfo
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
    
    logger.info(f"[Team Detection] Found {len(detection_result.players)} players, {len(detection_result.goalkeepers)} goalkeepers, ball={detection_result.ball is not None}")
    
    if len(detection_result.players) < 2:
        raise HTTPException(status_code=422, detail=f"Need at least 2 players to detect teams. Found: {len(detection_result.players)}")
    
    team1_info, team2_info = get_team_info(detection_result.players, image)
    
    if team1_info is None or team2_info is None:
        raise HTTPException(status_code=422, detail="Could not separate teams into two groups")
    
    def convert_color(color):
        if color is None:
            return (0, 0, 0)
        return (int(color[0]), int(color[1]), int(color[2]))
    
    return TeamDetectionResponse(
        team1_info=TeamInfoResponse(
            team_id="team1",
            color_name=str(team1_info.color_name),
            color_bgr=convert_color(team1_info.color_bgr),
            player_count=int(len(team1_info.players))
        ),
        team2_info=TeamInfoResponse(
            team_id="team2",
            color_name=str(team2_info.color_name),
            color_bgr=convert_color(team2_info.color_bgr),
            player_count=int(len(team2_info.players))
        ),
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
    logger.info(f"[Analysis] Image shape: {image.shape}, attacking_team={attacking_team}, goal_direction={goal_direction}")
    
    detector = detector_state.get()
    detection_result = detector.detect(image)
    
    logger.info(f"[Detection] Found {len(detection_result.players)} players, {len(detection_result.goalkeepers)} goalkeepers, ball={detection_result.ball is not None}")
    
    if len(detection_result.players) < 2:
        raise HTTPException(status_code=422, detail="Need at least 2 players for offside analysis")
    
    team1, team2 = separate_teams(detection_result.players, image)
    team1_info, team2_info = get_team_info(detection_result.players, image)
    
    logger.info(f"[Teams] Team 1: {len(team1)} players, Team 2: {len(team2)} players")
    
    if not team1 or not team2:
        raise HTTPException(status_code=422, detail="Could not separate teams")
    
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
        goal_direction=goal_direction
    )
    
    file_id = str(uuid.uuid4())[:8]
    
    annotated_filename = f"annotated_{file_id}.jpg"
    annotated_path = ANNOTATED_DIR / annotated_filename
    
    result_path = annotate_frame(
        image, detection_result, offside_result, team1, team2,
        output_filename=str(annotated_path)
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
            "player_count": int(len(team1))
        } if team1_info else None,
        "team2_info": {
            "team_id": "team2",
            "color_name": str(team2_info.color_name) if team2_info else "Unknown",
            "color_bgr": convert_color(team2_info.color_bgr) if team2_info else (0, 0, 0),
            "player_count": int(len(team2))
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
