# ⚽ AtleticoIntelligence

### AI-Powered Soccer Offside Review System

---

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB.svg?style=flat-square&logo=react&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Computer%20Vision-red.svg?style=flat-square&logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)

**Deterministic offside decisions using geometric perspective correction — not LLM guessing.**

[Features](#features) • [Architecture](#architecture) • [Quick Start](#quick-start) • [API](#api) • [Tech Stack](#tech-stack)

</div>

---

## 🎯 Overview

AtleticoIntelligence is a **human-in-the-loop** system for match officials that provides mathematically accurate offside decisions using computer vision and geometric calculations.

> **v2.0 Note:** Replaced LLM-based detection with pure geometric algorithms. Same input always produces the same output — no guessing, no hallucinations.

### Why Geometric Over LLM?

| Aspect | LLM Vision | Geometric Calculation |
|--------|------------|----------------------|
| **Accuracy** | Pattern matching guesses | Mathematical precision |
| **Perspective** | Cannot correct camera angles | Homography transformation |
| **Consistency** | Variable outputs | Deterministic results |
| **Speed** | API latency | Real-time local inference |
| **Offside Margin** | Arbitrary confidence | Exact meters |

---

## ✨ Features

- 🔍 **YOLOv8 Detection** — Real-time player, ball, goalkeeper, and referee detection
- 🎨 **K-means Team Separation** — Jersey color clustering in LAB color space
- 📐 **Camera Calibration** — Homography matrix for pixel-to-meter transformation
- 📏 **Geometric Offside Analysis** — Perspective-corrected positions in real-world coordinates
- 🎯 **VAR Standard Tolerance** — 50cm offside margin tolerance
- ⚽ **Goal-Line Technology** — Ball position vs goal line verification
- 📊 **SVG Pitch Diagrams** — Top-down visualization with offside lines
- 🖼️ **Annotated Frames** — OpenCV overlays with team colors and markers

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              BROWSER                                    │
│                           (React Frontend)                              │
│                    localhost:3000  │  Vite Proxy                       │
└─────────────────────────────────────┼───────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           FASTAPI BACKEND                               │
│                           localhost:8000                                │
│  ┌─────────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │  /detect-   │  │  /analyze-   │  │  /check-    │  │  /generate-  │  │
│  │  teams      │  │  offside     │  │  goal       │  │  visual      │  │
│  └─────────────┘  └──────────────┘  └─────────────┘  └──────────────┘  │
│         │                │                │                │           │
│         └────────────────┴────────────────┴────────────────┘           │
│                              │
│         ┌────────────────────┴────────────────────┐                    │
│         ▼                                         ▼                    │
│  ┌──────────────────┐                    ┌──────────────────┐            │
│  │   YOLOv8         │                    │   Geometric     │            │
│  │   Detector       │                    │   Analyzer      │            │
│  │   (Players/Ball) │                    │   (Homography)  │            │
│  └──────────────────┘                    └──────────────────┘            │
│         │                                         │                      │
│         ▼                                         ▼                      │
│  ┌──────────────────┐                    ┌──────────────────┐            │
│  │  Team Separation │                    │  Offside Engine  │            │
│  │  (K-means LAB)   │                    │  (Metric Space)  │            │
│  └──────────────────┘                    └──────────────────┘            │
│                                                               │
└───────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
                           ┌──────────────────┐
                           │   File Output    │
                           │  /output/*       │
                           └──────────────────┘
```

### Detection Pipeline

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│  Image  │───▶│  YOLOv8 │───▶│ K-means │───▶│ Camera  │───▶│ Geometric│
│ Input   │    │ Detect  │    │  Teams  │    │Calibrate│    │ Analyze  │
└─────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
                                                              │
                                         ┌────────────────────┴────────────┐
                                         ▼                                 ▼
                                  ┌─────────────┐                  ┌─────────────┐
                                  │   OFFSIDE   │                  │   ONSIDE    │
                                  │  /ONSIDE    │                  │  Decision   │
                                  └─────────────┘                  └─────────────┘
                                         │
                                         ▼
                              ┌─────────────────────┐
                              │  SVG + Annotations  │
                              └─────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- **Python** 3.9+
- **Node.js** 18+
- **4GB+ RAM**

### 1. Start Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Start server
python -m src.main
```

> Backend runs at `http://localhost:8000`

### 2. Start Frontend

```bash
cd frontend

# Install dependencies
npm install

# Start dev server
npm run dev
```

> Frontend runs at `http://localhost:3000`

### 3. Use the System

1. Open `http://localhost:3000`
2. Upload a football match image or video
3. Select the **attacking team** and **goal direction**
4. Click **Analyze**
5. View the offside verdict with annotated frame and pitch diagram

---

## 📡 API Reference

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/detect-teams` | POST | Detect teams from image |
| `/api/v1/analyze-offside` | POST | Full offside analysis |
| `/api/v1/check-goal` | POST | Goal-line technology check |
| `/api/v1/generate-visual` | POST | Generate SVG pitch diagram |
| `/health` | GET | Health check |

### Analyze Offside

```bash
curl -X POST "http://localhost:8000/api/v1/analyze-offside" \
  -F "image_file=@frame.jpg" \
  -F "goal_direction=right" \
  -F "attacking_team=team1"
```

**Response:**

```json
{
  "decision": "OFFSIDE",
  "confidence": 0.92,
  "attacker_foot": {"x": 650.0, "y": 200.0},
  "defender_foot": {"x": 540.0, "y": 200.0},
  "offside_margin_pixels": 110.0,
  "offside_margin_meters": 0.85,
  "annotated_image_url": "/annotated/annotated_abc123.jpg",
  "svg_url": "/pitch/pitch_abc123.svg",
  "attacking_team": "team1",
  "team1_info": {
    "color_name": "red",
    "color_rgb": [255, 0, 0],
    "player_count": 6,
    "goalkeeper": {"x": 120, "y": 200}
  },
  "team2_info": {
    "color_name": "blue",
    "color_rgb": [0, 0, 255],
    "player_count": 5,
    "goalkeeper": {"x": 80, "y": 200}
  }
}
```

---

## 📁 Project Structure

```
AtleticoIntelligence/
├── backend/
│   ├── src/
│   │   ├── main.py                      # FastAPI entry point
│   │   ├── api/
│   │   │   └── endpoints.py             # All API endpoints
│   │   ├── detection/
│   │   │   └── yolo_detector.py         # YOLOv8 player/ball detection
│   │   ├── logic/
│   │   │   ├── team_separation.py       # K-means jersey color clustering
│   │   │   ├── offside_analyzer.py      # Geometric offside calculations
│   │   │   └── goal_line.py             # Goal-line technology
│   │   ├── visualization/
│   │   │   ├── annotator.py             # OpenCV frame annotation
│   │   │   └── svg_generator.py         # SVG pitch diagram generation
│   │   └── utils/
│   │       ├── config.py                # YAML config loader
│   │       └── colors.py                # Jersey color extraction
│   ├── config.yaml                      # Configuration file
│   └── requirements.txt                 # Python dependencies
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx                      # Router setup
│   │   ├── components/
│   │   │   ├── VideoPlayer.jsx          # Upload + frame capture
│   │   │   ├── VerdictDisplay.jsx       # Result overlay
│   │   │   ├── GoalCheckDisplay.jsx     # Goal check results
│   │   │   └── SVGViewer.jsx            # SVG pitch viewer
│   │   ├── pages/
│   │   │   ├── MatchConsole.jsx         # Main offside review page
│   │   │   ├── GoalCheckConsole.jsx      # Goal check page
│   │   │   └── IncidentDetail.jsx        # Incident detail page
│   │   └── services/
│   │       └── api.js                   # API client
│   ├── package.json
│   └── vite.config.js                   # Vite + API proxy config
│
└── README.md
```

---

## ⚙️ Configuration

`backend/config.yaml`:

```yaml
detection:
  player_model_path: "uisikdag/yolo-v8-football-players-detection"
  confidence_threshold: 0.25
  ball_confidence_threshold: 0.01

pitch:
  width: 105.0        # meters
  height: 68.0        # meters
  goal_width: 7.32    # meters

offside:
  tolerance_meters: 0.5   # VAR standard

camera_calibration:
  enabled: true
  auto_calibrate: true
  reprojection_threshold: 5.0

visualization:
  attacker_color: [255, 0, 0]    # Red (BGR)
  defender_color: [0, 0, 255]    # Blue (BGR)
  ball_color: [0, 255, 0]        # Green (BGR)
```

---

## 🧠 Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Backend** | FastAPI | REST API server |
| **Detection** | YOLOv8 (Ultralytics) | Player/ball detection |
| **Vision** | OpenCV | Image processing |
| **ML** | scikit-learn | K-means clustering |
| **Math** | NumPy | Geometric calculations |
| **Frontend** | React 18 + Vite | UI framework |
| **Styling** | Tailwind CSS | Styling |
| **Routing** | React Router DOM | SPA navigation |
| **HTTP** | Axios | API client |

---

## 📜 Version History

| Version | Changes |
|---------|---------|
| **v2.0** | Replaced LLM Vision with geometric perspective correction using homography transforms |
| **v1.0** | Initial release with YOLO detection and Gemini explanations |

---

<div align="center">

**Built for match officials who need deterministic, mathematically accurate offside decisions.**

</div>
