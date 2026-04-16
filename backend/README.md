# AtleticoIntelligence - AI-Powered Offside Review System

Real-time offside detection from a single frame using YOLOv8 and geometric perspective correction.

## What's New in v2.0

**MAJOR ARCHITECTURAL CHANGE:** The LLM Vision detection pipeline has been **REMOVED** and replaced with proper geometric calculations.

### Why?
- LLMs cannot accurately perform geometric calculations
- Perspective correction requires mathematical transformations, not pattern matching
- Detection accuracy improved significantly with homography-based approach

**LLMs are now used ONLY for generating explanations, NOT for detection.**

## Project Structure

```
backend/
├── config.yaml                 # Configuration file
├── requirements.txt            # Python dependencies
├── demo.py                     # Demo script
├── README.md
├── models/                     # YOLO models (auto-downloaded)
├── output/
│   └── annotated/              # Generated annotations
└── src/
    ├── main.py                 # FastAPI entry point
    ├── api/
    │   └── endpoints.py        # API endpoints
    ├── detection/
    │   └── yolo_detector.py    # YOLOv8 detector
    ├── logic/
    │   ├── camera_calibration.py  # NEW: Homography-based calibration
    │   ├── team_separation.py     # Team separation by jersey color
    │   └── offside_analyzer.py    # NEW: Geometric offside analysis
    ├── visualization/
    │   ├── annotator.py           # Frame annotation with OpenCV
    │   ├── svg_generator.py       # Top-down pitch SVG
    │   ├── llm_integration.py     # LLM explanations only
    │   └── vision_analyzer.py     # DEPRECATED: Do not use
    └── utils/
        └── config.py            # Configuration loader
```

## How It Works

### Detection Pipeline (v2.0)

1. **Detection**: YOLOv8 detects players (class 0) and ball (class 32)
2. **Team Separation**: K-means clustering separates teams by jersey color
3. **Camera Calibration**: Automatic homography calculation from pitch lines or manual points
4. **Coordinate Transformation**: Image pixels -> Real-world pitch coordinates (meters)
5. **Offside Analysis**: Geometric comparison in pitch space with 50cm tolerance
6. **Visualization**: Annotated frame + SVG pitch diagram
7. **Explanation**: LLM generates human-readable explanation (post-detection only)

### Key Improvements

| Aspect | v1.0 (LLM) | v2.0 (Geometric) |
|--------|------------|------------------|
| **Offside Line** | Vertical (incorrect) | Perspective-corrected |
| **Position Comparison** | Pixel space | Metric space (meters) |
| **Accuracy** | Poor (guessing) | High (mathematical) |
| **Consistency** | Variable | Deterministic |
| **Speed** | Slow (API calls) | Fast (local computation) |

## Installation

```bash
cd backend
pip install -r requirements.txt
```

YOLOv8 model will be auto-downloaded on first run.

## Running the Server

```bash
cd backend

# Set Gemini API key (optional, for explanations only)
export GEMINI_API_KEY="your-key"

# Run server
python -m src.main
```

Server runs at `http://localhost:8000`

## API Endpoints

### POST /api/v1/analyze-frame

Upload an image frame for offside analysis with automatic camera calibration.

```bash
curl -X POST "http://localhost:8000/api/v1/analyze-frame" \
  -F "image_file=@frame.jpg" \
  -F "goal_direction=right"
```

Response:
```json
{
  "decision": "OFFSIDE",
  "confidence": 0.92,
  "attacker_position": {"x": 650.0, "y": 125.0},
  "defender_position": {"x": 540.0, "y": 125.0},
  "attacker_foot": {"x": 650.0, "y": 200.0},
  "defender_foot": {"x": 540.0, "y": 200.0},
  "explanation": "The attacker is 0.85m offside...",
  "annotated_image_url": "/annotated/annotated_abc123.jpg",
  "svg_url": "/output/pitch_abc123.svg",
  "calibration_quality": "good",
  "offside_margin_meters": 0.85
}
```

### POST /api/v1/analyze-with-calibration

Analyze with manual camera calibration for higher accuracy. Provide the 4 pitch corner coordinates.

```bash
curl -X POST "http://localhost:8000/api/v1/analyze-with-calibration" \
  -F "image_file=@frame.jpg" \
  -F "goal_direction=right" \
  -F "pitch_top_left_x=100" \
  -F "pitch_top_left_y=50" \
  -F "pitch_top_right_x=1180" \
  -F "pitch_top_right_y=50" \
  -F "pitch_bottom_right_x=1220" \
  -F "pitch_bottom_right_y=670" \
  -F "pitch_bottom_left_x=60" \
  -F "pitch_bottom_left_y=670"
```

### POST /api/v1/generate-visual

Generate SVG visualization from structured JSON.

```bash
curl -X POST "http://localhost:8000/api/v1/generate-visual" \
  -H "Content-Type: application/json" \
  -d '{
    "attacker_position": {"x": 65, "y": 34},
    "defender_position": {"x": 54, "y": 34},
    "decision": "OFFSIDE"
  }'
```

## Configuration

Edit `config.yaml` to customize:

```yaml
offside:
  tolerance_meters: 0.5  # VAR-style 50cm tolerance

camera_calibration:
  enabled: true
  auto_calibrate: true
  reprojection_threshold: 5.0  # Max error in meters

llm:
  use_for_explanations: true   # Generate human-readable text
  use_for_detection: false     # NEVER enable - deprecated
```

## Running Demo

```bash
cd backend
python demo.py
```

Place a football frame as `test_frame.jpg` in the backend folder.

## Calibration Quality Indicator

The system shows calibration quality in annotations:

- **GREEN** (`good`): Reprojection error < 2m
- **ORANGE** (`poor`): Reprojection error 2-5m  
- **RED** (`fallback`): Auto-calibration failed, using default
- **GRAY** (`failed`): Could not calibrate

For best results, use manual calibration with known pitch corner positions.

## Migration from v1.0

**Removed endpoints:**
- `POST /api/v1/analyze-frame-vision` - Removed (LLM vision)
- `POST /api/v1/analyze-frame-hybrid` - Removed (LLM + YOLO)

**New endpoints:**
- `POST /api/v1/analyze-with-calibration` - Manual calibration

**Response changes:**
- Added `calibration_quality` field
- Added `offside_margin_meters` field
- Offside line is now perspective-corrected

## Technical Details

### Camera Calibration

Uses homography transformation to map image coordinates to real-world pitch coordinates:

```
H = findHomography(image_points, pitch_points)
pitch_pos = H * image_pos
```

### Offside Calculation

1. Transform attacker and defender foot positions to pitch space
2. Compare x-coordinates in meters
3. Apply 50cm tolerance (VAR standard)
4. Account for goal direction (left/right)

### Why No LLM for Detection?

- LLMs are **pattern matchers**, not **geometric calculators**
- Cannot accurately judge 3D spatial relationships from 2D images
- Confidence scores are arbitrary (how confident the LLM is in its guess)
- No mathematical basis for perspective correction
- Inconsistent results across similar frames

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at `http://localhost:3000` and proxies API calls to `http://localhost:8000`.

## Future Improvements

- [ ] Pose estimation for accurate foot positions (replace bbox bottoms)
- [ ] Multiple camera triangulation
- [ ] Temporal analysis across frames
- [ ] Semi-automatic pitch corner detection
- [ ] Player tracking for consistent team assignment

## License

MIT