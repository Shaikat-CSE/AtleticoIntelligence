from .team_separation import TeamSeparator, separate_teams
from .offside_analyzer import (
    GeometricOffsideAnalyzer,
    OffsideAnalysisResult,
    analyze_offside,
    PlayerPosition
)
from .camera_calibration import (
    CameraCalibrator,
    CalibrationResult,
    PerspectiveCorrector,
    PitchDimensions,
    create_calibrator
)

# Keep old name for backward compatibility
OffsideAnalyzer = GeometricOffsideAnalyzer

__all__ = [
    "TeamSeparator",
    "separate_teams",
    "GeometricOffsideAnalyzer",
    "OffsideAnalyzer",
    "OffsideAnalysisResult",
    "analyze_offside",
    "PlayerPosition",
    "CameraCalibrator",
    "CalibrationResult",
    "PerspectiveCorrector",
    "PitchDimensions",
    "create_calibrator",
]