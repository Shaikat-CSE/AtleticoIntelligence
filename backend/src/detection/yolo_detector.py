from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str

    @property
    def foot_position(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, self.y2)

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass
class DetectionResult:
    players: List[BoundingBox]
    ball: Optional[BoundingBox]
    image_shape: Tuple[int, int]


class YOLODetector:
    PERSON_CLASS = 0
    BALL_CLASS = 32

    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.3, ball_confidence_threshold: float = 0.05):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.ball_confidence_threshold = ball_confidence_threshold
        self.model = None
        print(f"[YOLODetector] Initialized with confidence_threshold={confidence_threshold}, ball_confidence_threshold={ball_confidence_threshold}")

    def load_model(self):
        from ultralytics import YOLO
        self.model = YOLO(self.model_path)
        print(f"[YOLODetector] Model loaded: {self.model_path}")

    def detect(self, image: np.ndarray) -> DetectionResult:
        if self.model is None:
            self.load_model()

        h, w = image.shape[:2]
        results = self.model(image, conf=self.confidence_threshold, verbose=False)

        players = []
        ball = None

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                print(f"[YOLODetector] Detected: class={cls_id}, conf={conf:.2f}, box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

                if cls_id == self.PERSON_CLASS and conf >= self.confidence_threshold:
                    players.append(BoundingBox(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name="player"
                    ))
                elif cls_id == self.BALL_CLASS and conf >= self.ball_confidence_threshold:
                    ball = BoundingBox(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name="ball"
                    )

        players = self._filter_overlapping_players(players)

        print(f"[YOLODetector] Final: {len(players)} players, ball={ball is not None}")

        return DetectionResult(
            players=players,
            ball=ball,
            image_shape=(h, w)
        )

    def _filter_overlapping_players(self, players: List[BoundingBox], iou_threshold: float = 0.5) -> List[BoundingBox]:
        if len(players) <= 1:
            return players

        keep = []
        players_sorted = sorted(players, key=lambda p: p.confidence, reverse=True)

        for player in players_sorted:
            is_duplicate = False
            for kept in keep:
                if self._compute_iou(player, kept) > iou_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                keep.append(player)

        return keep

    def _compute_iou(self, box1: BoundingBox, box2: BoundingBox) -> float:
        x1 = max(box1.x1, box2.x1)
        y1 = max(box1.y1, box2.y1)
        x2 = min(box1.x2, box2.x2)
        y2 = min(box1.y2, box2.y2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0


def create_detector(model_path: str = "yolov8n.pt", confidence_threshold: float = 0.5, ball_confidence_threshold: float = 0.05) -> YOLODetector:
    detector = YOLODetector(model_path, confidence_threshold, ball_confidence_threshold)
    detector.load_model()
    return detector
