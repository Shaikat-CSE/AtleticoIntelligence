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
        title="AtleticoIntelligence - Offside Review System",
        description="AI-powered soccer incident review system with geometric perspective correction",
        version="2.0.0"
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # TODO: Restrict to specific origins in production
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
            "version": "2.0.0",
            "description": "AI-powered offside review system with geometric perspective correction",
            "endpoints": {
                "analyze_frame": "/api/v1/analyze-frame",
                "analyze_with_calibration": "/api/v1/analyze-with-calibration",
                "generate_visual": "/api/v1/generate-visual"
            },
            "features": [
                "YOLOv8 player and ball detection",
                "K-means team separation by jersey color",
                "Automatic camera calibration with homography",
                "Perspective-corrected offside calculation",
                "Real-world pitch coordinates (meters)",
                "LLM-generated explanations (post-detection only)"
            ],
            "note": "LLM is used ONLY for explanations, NOT for detection. Detection uses geometric calculations."
        }

    @app.get("/health")
    def health_check():
        return {"status": "healthy", "version": "2.0.0"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)