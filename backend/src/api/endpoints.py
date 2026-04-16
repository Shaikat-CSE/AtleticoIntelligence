from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import cv2
import numpy as np
import uuid
import logging

from ..detection import create_detector, BoundingBox
from ..logic import separate_teams, analyze_offside, CameraCalibrator, create_calibrator
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


class OffsideResponse(BaseModel):
    decision: str
    confidence: float
    attacker_position: Position
    defender_position: Position
    attacker_foot: Position
    defender_foot: Position
    explanation: str
    annotated_image_url: str
    svg_url: Optional[str] = None
    calibration_quality: str
    offside_margin_meters: float


class VisualizationRequest(BaseModel):
    attacker_position: Position
    defender_position: Position
    ball_position: Optional[Position] = None
    offside_line_x: Optional[float] = None
    decision: str
    generate_explanation: bool = True


class VisualizationResponse(BaseModel):
    svg_content: str
    explanation: Optional[str] = None


class DetectorState:
    """Simple state management for detector."""
    def __init__(self):
        self._detector = None
    
    def get(self):
        if self._detector is None:
            logger.info("Initializing YOLO detector...")
            self._detector = create_detector("yolov8n.pt", confidence_threshold=0.25, ball_confidence_threshold=0.05)
        return self._detector


detector_state = DetectorState()


def process_frame_analysis(
    image: np.ndarray, 
    goal_direction: str = "right",
    calibrator: Optional[CameraCalibrator] = None,
    attacking_team: Optional[str] = None
) -> dict:
    """
    Process a frame through the geometric offside detection pipeline.
    
    Args:
        image: Input image
        goal_direction: Direction of opponent's goal ("left" or "right")
        calibrator: Optional pre-calibrated camera
        attacking_team: Optional "team1" or "team2" to manually specify attacking team
        
    Returns:
        Dictionary with analysis results and file paths
    """
    h, w = image.shape[:2]
    logger.info(f"[Analysis] Image shape: {image.shape}")
    
    # Step 1: Detect players and ball
    detector = detector_state.get()
    detection_result = detector.detect(image)
    
    logger.info(
        f"[Detection] Found {len(detection_result.players)} players, "
        f"ball={detection_result.ball is not None}"
    )
    
    if len(detection_result.players) < 2:
        logger.warning("[Analysis] Insufficient players detected")
        return {
            "error": "Need at least 2 players for offside analysis",
            "structured_data": {
                "decision": "UNKNOWN",
                "attacker_position": {"x": 0, "y": 0},
                "defender_position": {"x": 0, "y": 0},
                "attacker_foot": {"x": 0, "y": 0},
                "defender_foot": {"x": 0, "y": 0},
                "confidence": 0.0,
                "calibration_quality": "failed",
                "offside_margin_meters": 0.0
            },
            "annotated_url": None,
            "svg_url": None,
            "file_id": None
        }
    
    # Step 2: Separate teams by jersey color
    team1, team2 = separate_teams(detection_result.players, image)
    logger.info(f"[Teams] Team 1: {len(team1)} players, Team 2: {len(team2)} players")
    
    if not team1 or not team2:
        logger.warning("[Analysis] Could not separate teams")
        return {
            "error": "Could not separate teams",
            "structured_data": {
                "decision": "UNKNOWN",
                "attacker_position": {"x": 0, "y": 0},
                "defender_position": {"x": 0, "y": 0},
                "attacker_foot": {"x": 0, "y": 0},
                "defender_foot": {"x": 0, "y": 0},
                "confidence": 0.0,
                "calibration_quality": "failed",
                "offside_margin_meters": 0.0
            },
            "annotated_url": None,
            "svg_url": None,
            "file_id": None
        }
    
    # Step 3: Calibrate camera if not provided
    if calibrator is None:
        calibrator = create_calibrator(image)
        logger.info("[Calibration] Auto-calibrated camera")
    
    # Get ball position if detected
    ball_pos = detection_result.ball.foot_position if detection_result.ball else None
    if ball_pos:
        logger.info(f"[Ball] Ball position: ({ball_pos[0]:.0f}, {ball_pos[1]:.0f})")
    
    # Step 4: Geometric offside analysis with perspective correction
    offside_result = analyze_offside(
        team1, 
        team2, 
        goal_direction=goal_direction,
        image=image,
        calibrator=calibrator,
        ball_position=ball_pos,
        attacking_team=attacking_team
    )
    
    file_id = str(uuid.uuid4())[:8]
    
    # Step 5: Generate annotated image
    annotated_filename = f"annotated_{file_id}.jpg"
    annotated_path = ANNOTATED_DIR / annotated_filename
    
    result_path = annotate_frame(
        image, detection_result, offside_result, team1, team2,
        output_filename=str(annotated_path)
    )
    logger.info(f"[Output] Annotated image saved: {result_path}")
    annotated_url = f"/annotated/{annotated_filename}"
    
    # Step 6: Generate SVG visualization
    import time
    svg_filename = f"pitch_{file_id}_{goal_direction}_{int(time.time())}.svg"
    svg_path = OUTPUT_DIR / svg_filename
    
    # Use RAW YOLO foot positions for drawing attacker/defender in SVG
    # Scale image coords to SVG viewBox (scale by ratio of SVG size to image size)
    svg_w = 1050  # SVG viewBox width (pitch_width * 10)
    svg_h = 680   # SVG viewBox height (pitch_height * 10)
    scale_x_svg = svg_w / w
    scale_y_svg = svg_h / h
    
    # Raw YOLO positions for attacker/defender/goalkeeper markers
    attacker_foot = offside_result.attacker.foot_position
    defender_foot = offside_result.second_last_defender.foot_position
    goalkeeper_foot = offside_result.goalkeeper.foot_position if offside_result.goalkeeper else (0, 0)
    attacker_pos = (attacker_foot[0] * scale_x_svg, attacker_foot[1] * scale_y_svg)
    defender_pos = (defender_foot[0] * scale_x_svg, defender_foot[1] * scale_y_svg)
    goalkeeper_pos = (goalkeeper_foot[0] * scale_x_svg, goalkeeper_foot[1] * scale_y_svg)
    
    # Get team positions in SVG coordinates (raw YOLO positions)
    # Exclude attacker, defender, and goalkeeper to avoid double drawing
    excluded_positions = [attacker_foot, defender_foot, goalkeeper_foot]
    team1_positions = []
    team2_positions = []
    for p in team1:
        if p.foot_position not in excluded_positions:
            team1_positions.append((p.foot_position[0] * scale_x_svg, p.foot_position[1] * scale_y_svg))
    for p in team2:
        if p.foot_position not in excluded_positions:
            team2_positions.append((p.foot_position[0] * scale_x_svg, p.foot_position[1] * scale_y_svg))
    
    # Ball position in SVG coordinates (raw YOLO)
    ball_pos_scaled = None
    if ball_pos is not None:
        ball_pos_scaled = (ball_pos[0] * scale_x_svg, ball_pos[1] * scale_y_svg)
    
    # Offside line: drawn at attacker's X position (player with ball)
    offside_line_x_val = attacker_pos[0]
    
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
        attacking_team=attacking_team or "team1",
        goal_direction=goal_direction
    )
    logger.info(f"[Output] SVG saved: {svg_path}")
    svg_url = f"/pitch/{svg_filename}"
    
    # Step 7: Prepare structured data for response
    def safe_float(val):
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return 0.0
        return val
    
    structured_data = {
        "decision": offside_result.decision,
        "attacker_position": {"x": safe_float(offside_result.attacker.center[0]), "y": safe_float(offside_result.attacker.center[1])},
        "defender_position": {"x": safe_float(offside_result.second_last_defender.center[0]), "y": safe_float(offside_result.second_last_defender.center[1])},
        "attacker_foot": {"x": safe_float(offside_result.attacker.foot_position[0]), "y": safe_float(offside_result.attacker.foot_position[1])},
        "defender_foot": {"x": safe_float(offside_result.second_last_defender.foot_position[0]), "y": safe_float(offside_result.second_last_defender.foot_position[1])},
        "confidence": safe_float(offside_result.confidence),
        "calibration_quality": offside_result.calibration_quality,
        "offside_margin_meters": safe_float(offside_result.offside_margin_meters)
    }
    
    return {
        "structured_data": structured_data,
        "annotated_url": annotated_url,
        "svg_url": svg_url,
        "svg_content": svg_content,
        "file_id": file_id
    }


@router.post("/analyze-frame", response_model=OffsideResponse)
async def analyze_frame(
    image_file: UploadFile = File(...),
    goal_direction: str = Form("right"),
    attacking_team: Optional[str] = Form(None),
):
    """
    Analyze a single frame for offside using geometric perspective correction.
    
    This endpoint uses:
    1. YOLOv8 for player/ball detection
    2. K-means clustering for team separation
    3. Camera calibration and homography for perspective correction
    4. Geometric calculations in real-world pitch coordinates
    
    Use attacking_team to manually specify which team is attacking.
    """
    # Validate input
    contents = await image_file.read()
    if len(contents) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")
    
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    logger.info(f"[API] goal_direction={goal_direction}, attacking_team={attacking_team}")
    
    # Process frame through geometric pipeline
    result = process_frame_analysis(image, goal_direction, attacking_team=attacking_team)
    
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    
    structured_data = result["structured_data"]
    
    return OffsideResponse(
        decision=structured_data["decision"],
        confidence=structured_data["confidence"],
        attacker_position=Position(**structured_data["attacker_position"]),
        defender_position=Position(**structured_data["defender_position"]),
        attacker_foot=Position(**structured_data["attacker_foot"]),
        defender_foot=Position(**structured_data["defender_foot"]),
        explanation=f"{structured_data['decision']} - Margin: {structured_data['offside_margin_meters']:.2f}m",
        annotated_image_url=result["annotated_url"],
        svg_url=result["svg_url"],
        calibration_quality=structured_data["calibration_quality"],
        offside_margin_meters=structured_data["offside_margin_meters"]
    )


@router.post("/generate-visual", response_model=VisualizationResponse)
async def generate_visual(request: VisualizationRequest):
    """Generate SVG visualization from structured positions."""
    svg_content = generate_offside_svg(
        attacker_pos=(request.attacker_position.x, request.attacker_position.y),
        defender_pos=(request.defender_position.x, request.defender_position.y),
        ball_pos=(request.ball_position.x, request.ball_position.y) if request.ball_position else None,
        offside_line_x=request.offside_line_x,
        decision=request.decision
    )
    
    return VisualizationResponse(svg_content=svg_content, explanation=None)


@router.post("/analyze-with-calibration", response_model=OffsideResponse)
async def analyze_with_calibration(
    image_file: UploadFile = File(...),
    goal_direction: str = "right",
    pitch_top_left_x: Optional[float] = None,
    pitch_top_left_y: Optional[float] = None,
    pitch_top_right_x: Optional[float] = None,
    pitch_top_right_y: Optional[float] = None,
    pitch_bottom_right_x: Optional[float] = None,
    pitch_bottom_right_y: Optional[float] = None,
    pitch_bottom_left_x: Optional[float] = None,
    pitch_bottom_left_y: Optional[float] = None
):
    """
    Analyze offside with manual camera calibration for higher accuracy.
    
    Provide the image coordinates of the four pitch corners for precise calibration.
    """
    contents = await image_file.read()
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 50MB)")
    
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
    
    # Check if all calibration points provided
    calibrator = None
    if all(v is not None for v in [
        pitch_top_left_x, pitch_top_left_y,
        pitch_top_right_x, pitch_top_right_y,
        pitch_bottom_right_x, pitch_bottom_right_y,
        pitch_bottom_left_x, pitch_bottom_left_y
    ]):
        # Manual calibration with provided points
        image_points = np.array([
            [pitch_top_left_x, pitch_top_left_y],
            [pitch_top_right_x, pitch_top_right_y],
            [pitch_bottom_right_x, pitch_bottom_right_y],
            [pitch_bottom_left_x, pitch_bottom_left_y]
        ], dtype=np.float32)
        
        pitch_points = np.array([
            [0, 0],
            [105, 0],
            [105, 68],
            [0, 68]
        ], dtype=np.float32)
        
        calibrator = create_calibrator(image, (image_points, pitch_points))
        logger.info("[Calibration] Using manual calibration points")
    
    # Process with calibration
    result = process_frame_analysis(image, goal_direction, calibrator)
    
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    
    structured_data = result["structured_data"]
    
    return OffsideResponse(
        decision=structured_data["decision"],
        confidence=structured_data["confidence"],
        attacker_position=Position(**structured_data["attacker_position"]),
        defender_position=Position(**structured_data["defender_position"]),
        attacker_foot=Position(**structured_data["attacker_foot"]),
        defender_foot=Position(**structured_data["defender_foot"]),
        explanation=f"{structured_data['decision']} - Margin: {structured_data['offside_margin_meters']:.2f}m",
        annotated_image_url=result["annotated_url"],
        svg_url=result["svg_url"],
        calibration_quality=structured_data["calibration_quality"],
        offside_margin_meters=structured_data["offside_margin_meters"]
    )