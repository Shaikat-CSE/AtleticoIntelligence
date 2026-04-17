import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import cv2

from src.detection import create_detector
from src.logic import separate_teams, analyze_offside
from src.visualization import annotate_frame, generate_offside_svg, generate_llm_explanation


def demo():
    print("=" * 60)
    print("AtleticoIntelligence - Offside Detection Demo v2.0")
    print("=" * 60)
    print("\nNOTE: Using geometric perspective correction (LLM removed)")

    print("\n[1] Loading YOLOv8 detector...")
    detector = create_detector("uisikdag/yolo-v8-football-players-detection", confidence_threshold=0.25)
    print("    Detector loaded successfully")

    image_path = "test_frame.jpg"
    if not Path(image_path).exists():
        print(f"\n    Note: Place a football frame at '{image_path}' to test detection")
        print("    Using mock image for demo...")

        # Create a mock football pitch image
        image = np.ones((720, 1280, 3), dtype=np.uint8) * 50
        # Green field
        cv2.rectangle(image, (0, 100), (1280, 620), (34, 139, 34), -1)
        # White lines
        cv2.rectangle(image, (50, 150), (1230, 570), (255, 255, 255), 3)
        cv2.line(image, (640, 150), (640, 570), (255, 255, 255), 3)
        cv2.circle(image, (640, 360), 100, (255, 255, 255), 3)
        
        # Add mock players (red team attacking from left)
        cv2.rectangle(image, (300, 300), (350, 400), (0, 0, 200), -1)  # Attacker
        cv2.rectangle(image, (450, 280), (500, 380), (200, 0, 0), -1)  # Defender 1
        cv2.rectangle(image, (500, 320), (550, 420), (200, 0, 0), -1)  # Defender 2 (2nd last)
        cv2.rectangle(image, (550, 300), (600, 400), (200, 0, 0), -1)  # Defender 3 (GK)
        
        # Add mock ball near the attacker
        cv2.circle(image, (320, 405), 12, (255, 255, 255), -1)  # White ball
        
        detection_result = detector.detect(image)
    else:
        print(f"\n[2] Loading image: {image_path}")
        image = cv2.imread(image_path)
        print(f"    Image shape: {image.shape}")

        print("\n[3] Running detection...")
        detection_result = detector.detect(image)

    print(f"    Detected {len(detection_result.players)} players")
    print(f"    Detected ball: {detection_result.ball is not None}")

    if len(detection_result.players) < 2:
        print("\n    Warning: Not enough players detected for offside analysis")
        print("    Make sure the image contains visible football players")
        return

    print("\n[4] Separating teams by jersey color...")
    team1, team2 = separate_teams(detection_result.players, image)
    print(f"    Team 1: {len(team1)} players")
    print(f"    Team 2: {len(team2)} players")

    if not team1 or not team2:
        print("\n    Warning: Could not separate teams by color")
        print("    Assigning all players to one team for demo")
        team1 = detection_result.players[:len(detection_result.players)//2]
        team2 = detection_result.players[len(detection_result.players)//2:]

    print("\n[5] Calibrating camera and analyzing offside...")
    print("    - Computing homography matrix")
    print("    - Transforming to pitch coordinates")
    print("    - Comparing positions in metric space")
    
    offside_result = analyze_offside(
        team1, team2, 
        goal_direction="right",
        image=image
    )

    if offside_result and offside_result.decision != "UNKNOWN":
        print(f"\n    Decision: {offside_result.decision}")
        print(f"    Confidence: {offside_result.confidence:.2f}")
        print(f"    Calibration Quality: {offside_result.calibration_quality}")
        
        if offside_result.attacker_pitch_pos:
            print(f"    Attacker pitch position: ({offside_result.attacker_pitch_pos[0]:.2f}m, {offside_result.attacker_pitch_pos[1]:.2f}m)")
            print(f"    Defender pitch position: ({offside_result.defender_pitch_pos[0]:.2f}m, {offside_result.defender_pitch_pos[1]:.2f}m)")
            print(f"    Offside margin: {offside_result.offside_margin_meters:.2f}m")
        else:
            print(f"    Attacker image position: ({offside_result.attacker.foot_position[0]:.1f}, {offside_result.attacker.foot_position[1]:.1f})")
            print(f"    Defender image position: ({offside_result.second_last_defender.foot_position[0]:.1f}, {offside_result.second_last_defender.foot_position[1]:.1f})")

        structured_data = {
            "decision": offside_result.decision,
            "attacker_position": {"x": offside_result.attacker.foot_position[0], "y": offside_result.attacker.foot_position[1]},
            "defender_position": {"x": offside_result.second_last_defender.foot_position[0], "y": offside_result.second_last_defender.foot_position[1]},
            "confidence": offside_result.confidence,
            "calibration_quality": offside_result.calibration_quality,
            "offside_margin_meters": offside_result.offside_margin_meters
        }

        print("\n[6] Generating LLM explanation...")
        print("    (LLM used ONLY for text, NOT for detection)")
        explanation = generate_llm_explanation(structured_data)
        print(f"    {explanation}")

        print("\n[7] Generating visualizations...")
        output_dir = Path(__file__).parent / "output"
        output_dir.mkdir(exist_ok=True)

        annotated_path = output_dir / "demo_annotated.jpg"
        annotate_frame(
            image, detection_result, offside_result, team1, team2,
            output_filename=str(annotated_path)
        )
        print(f"    Annotated image: {annotated_path}")

        # Use pitch coordinates for SVG if available
        if offside_result.attacker_pitch_pos:
            attacker_pos = offside_result.attacker_pitch_pos
            defender_pos = offside_result.defender_pitch_pos
        else:
            attacker_pos = offside_result.attacker.foot_position
            defender_pos = offside_result.second_last_defender.foot_position
            
        svg_path = output_dir / "demo_pitch.svg"
        generate_offside_svg(
            attacker_pos=attacker_pos,
            defender_pos=defender_pos,
            ball_pos=detection_result.ball.foot_position if detection_result.ball else None,
            offside_line_x=defender_pos[0] if offside_result.decision == "OFFSIDE" else None,
            offside_line_top=offside_result.offside_line_image[0] if offside_result.offside_line_image else None,
            offside_line_bottom=offside_result.offside_line_image[1] if offside_result.offside_line_image else None,
            decision=offside_result.decision,
            image_width=image.shape[1],
            image_height=image.shape[0],
            output_path=str(svg_path)
        )
        print(f"    SVG pitch: {svg_path}")

    else:
        print("    Could not determine offside (insufficient players)")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    demo()