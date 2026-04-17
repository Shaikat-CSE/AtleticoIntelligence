import sys
from pathlib import Path
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from src.api import router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

def create_app() -> FastAPI:
    app = FastAPI(
        title="AtleticoIntelligence - Match Review System",
        description="AI-powered soccer incident review system with offside and goal-line checks",
        version="3.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "annotated").mkdir(exist_ok=True)

    app.mount("/output", StaticFiles(directory=str(output_dir)), name="output")
    app.mount("/annotated", StaticFiles(directory=str(output_dir / "annotated")), name="annotated")
    app.mount("/pitch", StaticFiles(directory=str(output_dir)), name="pitch")

    app.include_router(router, prefix="/api/v1", tags=["offside"])

    @app.get("/")
    def root():
        return {
            "name": "AtleticoIntelligence",
            "version": "3.0.0",
            "description": "AI-powered offside review system",
            "user_flow": [
                "Step 1: POST /api/v1/detect-teams - Upload image to detect team colors",
                "Step 2: User selects attacking team + direction",
                "Step 3: POST /api/v1/analyze-offside - Full offside analysis",
                "Step 4: POST /api/v1/check-goal - Goal-line technology style goal check"
            ],
            "endpoints": {
                "detect_teams": "/api/v1/detect-teams",
                "analyze_offside": "/api/v1/analyze-offside",
                "check_goal": "/api/v1/check-goal",
                "generate_visual": "/api/v1/generate-visual"
            },
            "features": [
                "YOLOv8 player and ball detection",
                "K-means team separation by jersey color",
                "Goalkeeper detection and handling",
                "Offside line placement based on attacker/defender positions",
                "Goal-line estimation and goal/no-goal decisions from freeze frames"
            ]
        }

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "version": "3.0.0"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
