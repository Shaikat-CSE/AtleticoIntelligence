# AtleticoIntelligence

AI-powered soccer offside review system for match officials with geometric perspective correction.

## What's New in v2.0

**MAJOR CHANGE:** Replaced LLM Vision detection with geometric calculations using camera calibration and homography transforms. This provides deterministic, mathematically accurate offside decisions instead of AI guessing.

**Key improvements:**
- ✅ Perspective-corrected offside line (not just vertical)
- ✅ Real-world pitch coordinates (meters, not pixels)
- ✅ 50cm VAR tolerance standard
- ✅ Deterministic results (same input = same output)
- ✅ Faster (no API calls for detection)
- ✅ Offside margin in meters
- ✅ Calibration quality indicator

**Removed:** LLM Vision and Hybrid detection modes (fundamentally flawed for geometric problems)

## Overview

A human-in-the-loop system where officials select a frame from video, and AI analyzes it for offside decisions using computer vision and geometric calculations.

**How it works:**
1. Official uploads video or image
2. Scrub video to desired frame (or use image directly)
3. Click "Review This Frame" or "Analyze This Image"
4. **YOLOv8** detects players and ball
5. **K-means** clustering separates teams by jersey color
6. **Camera calibration** computes homography matrix from pitch lines
7. **Geometric transformation** converts pixels to pitch coordinates (meters)
8. **Offside analysis** compares positions in real-world space
9. Returns OFFSIDE/ONSIDE verdict with perspective-corrected visualization
10. **LLM** generates human-readable explanation (post-detection only)

## Quick Start

### 1. Start Backend

```bash
cd backend

pip install -r requirements.txt

set GEMINI_API_KEY=your-key   # Windows (optional, for explanations only)
export GEMINI_API_KEY=your-key  # Mac/Linux (optional, for explanations only)

python -m src.main
```

Backend runs at `http://localhost:8000`

### 2. Start Frontend (separate terminal)

```bash
cd frontend

npm install

npm run dev
```

Frontend runs at `http://localhost:3000`

### 3. Test

1. Open `http://localhost:3000`
2. Upload a football image (player(s) + ball)
3. Click "Analyze This Image"
4. View result with calibration quality and offside margin

## Project Structure

```
AtleticoIntelligence/
├── backend/
│   ├── src/
│   │   ├── api/endpoints.py          # FastAPI endpoints
│   │   ├── detection/yolo_detector.py    # YOLOv8 detection
│   │   ├── logic/
│   │   │   ├── camera_calibration.py  # NEW: Homography calibration
│   │   │   ├── team_separation.py     # Jersey color clustering
│   │   │   └── offside_analyzer.py    # Geometric offside logic
│   │   └── visualization/
│   │       ├── annotator.py           # OpenCV annotations
│   │       ├── svg_generator.py       # Top-down pitch SVG
│   │       ├── llm_integration.py     # Explanations only
│   │       └── vision_analyzer.py     # DEPRECATED
│   └── config.yaml
└── frontend/
    └── src/
        ├── components/
        │   ├── VideoPlayer.jsx     # Upload + frame capture
        │   ├── VerdictDisplay.jsx  # Result overlay with new fields
        │   └── SVGViewer.jsx       # Pitch diagram
        ├── pages/
        │   ├── MatchConsole.jsx    # Main page
        │   └── IncidentDetail.jsx
        └── services/api.js         # API client
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/analyze-frame` | POST | Upload image → get offside verdict (auto calibration) |
| `/api/v1/analyze-with-calibration` | POST | Upload image with manual calibration points |
| `/api/v1/generate-visual` | POST | Generate SVG pitch diagram |
| `/health` | GET | Health check |

**Removed endpoints:**
- ~~`/api/v1/analyze-frame-vision`~~ (LLM Vision - removed)
- ~~`/api/v1/analyze-frame-hybrid`~~ (Hybrid - removed)

### Analyze Frame

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
  "attacker_foot": {"x": 650.0, "y": 200.0},
  "defender_foot": {"x": 540.0, "y": 200.0},
  "annotated_image_url": "/annotated/annotated_abc123.jpg",
  "svg_url": "/pitch_abc123.svg",
  "explanation": "The attacker is 0.85m offside...",
  "calibration_quality": "good",
  "offside_margin_meters": 0.85
}
```

### Analyze with Manual Calibration

For higher accuracy, provide the 4 pitch corner coordinates:

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

## Architecture v2.0

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Browser   │────▶│  FastAPI    │────▶│  YOLOv8     │
│   (React)   │◀────│  Backend    │◀────│  Detector   │
└─────────────┘     └─────────────┘     └─────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │ Camera Calibrator│
                    │  (Homography)    │
                    └─────────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │ Geometric Offside│
                    │  Analyzer (meters)│
                    └─────────────────┘
                            │
                            ▼
                    ┌─────────────────┐
                    │   LLM (Gemini)  │
                    │ (explanation ONLY)│
                    └─────────────────┘
```

## Detection Pipeline v2.0

1. **Image Input** → YOLOv8 detects players (class 0) and ball (class 32)
2. **Team Separation** → K-means clustering on jersey colors
3. **Camera Calibration** → Auto-detect pitch lines or use defaults
4. **Homography** → Compute H matrix for pixel-to-meter transformation
5. **Position Extraction** → Transform foot positions to pitch coordinates
6. **Offside Analysis** → Compare in metric space with 50cm tolerance
7. **Decision** → `attacker_x > defender_x + 0.5m → OFFSIDE`
8. **Visualization** → Perspective-corrected offside line
9. **Explanation** → LLM generates human-readable text

## Configuration

`backend/config.yaml`:

```yaml
detection:
  model_path: "yolov8n.pt"
  confidence_threshold: 0.25

offside:
  tolerance_meters: 0.5  # VAR standard

camera_calibration:
  enabled: true
  auto_calibrate: true
  reprojection_threshold: 5.0

llm:
  enabled: true
  provider: "gemini"
  model: "gemini-pro"
  use_for_explanations: true   # Generate human-readable text
  use_for_detection: false     # NEVER enable - deprecated
```

## Why Remove LLM Vision?

| Problem | Why It Failed |
|---------|---------------|
| **Geometric accuracy** | LLMs can't calculate 3D positions from 2D images |
| **Perspective** | No mathematical basis for correcting camera angles |
| **Consistency** | Same input could produce different outputs |
| **Confidence scores** | Arbitrary (how confident the LLM is in its guess) |
| **Speed** | Slow API calls vs fast local computation |

**The hard truth:** LLMs are pattern matchers, not geometric calculators. Offside detection requires mathematical transformations (homography) that LLMs cannot perform accurately.

## Calibration Quality Indicator

The system displays calibration quality:

- 🟢 **GOOD** - Reprojection error < 2m (high accuracy)
- 🟡 **POOR** - Reprojection error 2-5m (acceptable)
- 🔴 **FALLBACK** - Auto-calibration failed, using defaults
- ⚪ **FAILED** - Could not calibrate

For best results, use manual calibration with known pitch corner positions.

## Requirements

**Backend:** Python 3.9+, 4GB+ RAM, OpenCV
**Frontend:** Node.js 18+

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend API | FastAPI |
| Detection | YOLOv8 (ultralytics) |
| Team Separation | OpenCV + scikit-learn |
| Camera Calibration | OpenCV (homography) |
| Offside Analysis | NumPy geometric calculations |
| LLM | Google Gemini (explanations only) |
| Frontend | React + Vite + Tailwind |

## Version History

- **v2.0** - Replaced LLM Vision with geometric perspective correction
- **v1.0** - Initial release with YOLO detection and LLM explanations

## License

MIT