import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Any


@dataclass
class DetectionConfig:
    model_path: str
    confidence_threshold: float
    iou_threshold: float
    classes: Dict[str, int]


@dataclass
class PitchConfig:
    width: float
    height: float
    goal_width: float
    goal_depth: float


@dataclass
class TeamSeparationConfig:
    method: str
    n_clusters: int


@dataclass
class OffsideConfig:
    tolerance_pixels: float
    confidence_weights: Dict[str, float]


@dataclass
class VisualizationConfig:
    output_dir: str
    bbox_thickness: int
    attacker_color: List[int]
    defender_color: List[int]
    ball_color: List[int]
    offside_line_color: List[int]


@dataclass
class LLMConfig:
    enabled: bool
    provider: str
    model: str
    api_key_env: str


@dataclass
class AppConfig:
    detection: DetectionConfig
    pitch: PitchConfig
    team_separation: TeamSeparationConfig
    offside: OffsideConfig
    visualization: VisualizationConfig
    llm: LLMConfig


def load_config(config_path: str = "config.yaml") -> AppConfig:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    return AppConfig(
        detection=DetectionConfig(**data["detection"]),
        pitch=PitchConfig(**data["pitch"]),
        team_separation=TeamSeparationConfig(**data["team_separation"]),
        offside=OffsideConfig(**data["offside"]),
        visualization=VisualizationConfig(**data["visualization"]),
        llm=LLMConfig(**data["llm"]),
    )


def get_default_config() -> AppConfig:
    return load_config(Path(__file__).parent.parent / "config.yaml")
