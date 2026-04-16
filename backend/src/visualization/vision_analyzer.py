"""
DEPRECATED: This module previously used LLM Vision for offside detection.

This approach has been REMOVED because:
1. LLMs cannot accurately perform geometric calculations
2. Perspective correction requires mathematical transformations, not pattern matching
3. Detection should use proper computer vision techniques

The new pipeline uses:
- YOLO for object detection
- Camera calibration and homography for perspective correction
- Geometric calculations in real-world pitch coordinates

LLMs are now used ONLY for generating human-readable explanations,
NOT for making detection decisions.

For detection, use:
- src.logic.camera_calibration for perspective correction
- src.logic.offside_analyzer for geometric offside calculation

This file is kept for backward compatibility but the detection functions
are no longer used in the main pipeline.
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def analyze_with_vision(image_path: str, provider: str = None, model: str = None) -> Optional[Dict[str, Any]]:
    """
    DEPRECATED: Do not use for offside detection.
    
    This function previously used LLM vision for detection, which is
    fundamentally incorrect for geometric problems.
    
    Use src.logic.offside_analyzer.analyze_offside() instead.
    """
    logger.warning(
        "analyze_with_vision is DEPRECATED and returns None. "
        "Use geometric offside analyzer instead."
    )
    return None


def analyze_hybrid(image_path: str, yolo_detections: Dict[str, Any], provider: str = None, model: str = None) -> Optional[Dict[str, Any]]:
    """
    DEPRECATED: Do not use for offside detection.
    
    This function previously combined YOLO with LLM vision, which
    compounds errors rather than improving accuracy.
    
    Use src.logic.offside_analyzer.analyze_offside() instead.
    """
    logger.warning(
        "analyze_hybrid is DEPRECATED and returns None. "
        "Use geometric offside analyzer instead."
    )
    return None


# Keep imports for any code that might reference them
# but they should not be used
try:
    from .llm_integration import LLMIntegration, generate_llm_explanation
except ImportError:
    pass