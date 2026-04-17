from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
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

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class DetectionResult:
    players: List[BoundingBox]
    goalkeepers: List[BoundingBox]
    ball: Optional[BoundingBox]
    image_shape: Tuple[int, int]


class YOLODetector:
    FOOTBALL_CLASSES = {
        0: "player",
        1: "ball", 
        2: "goalkeeper",
        3: "referee"
    }
    
    def __init__(
        self, 
        player_model_path: str = "uisikdag/yolo-v8-football-players-detection", 
        confidence_threshold: float = 0.25, 
        ball_confidence_threshold: float = 0.01
    ):
        self.player_model_path = player_model_path
        self.confidence_threshold = confidence_threshold
        self.ball_confidence_threshold = ball_confidence_threshold
        self.player_model = None
        self.ball_model = None
        print(f"[YOLODetector] Initialized with player_model={player_model_path}")

    def load_model(self):
        from ultralytics import YOLO
        
        player_path = self._download_model(self.player_model_path)
        self.player_model = YOLO(player_path)
        print(f"[YOLODetector] Player model loaded: {player_path}")
        
        ball_path = self._download_model("yolov8n.pt")
        self.ball_model = YOLO(ball_path)
        print(f"[YOLODetector] Ball model loaded: {ball_path}")

    def _download_model(self, model_id: str) -> str:
        if "/" not in model_id:
            return model_id
        
        cache_dir = Path("models")
        cache_dir.mkdir(exist_ok=True)
        safe_name = model_id.replace("/", "_")
        local_file = cache_dir / f"{safe_name}.pt"
        
        if local_file.exists():
            print(f"[YOLODetector] Using cached model: {local_file}")
            return str(local_file)
        
        print(f"[YOLODetector] Downloading model from HuggingFace: {model_id}")
        try:
            from huggingface_hub import hf_hub_download
            downloaded_path = hf_hub_download(
                repo_id=model_id,
                filename="best.pt",
                cache_dir=str(cache_dir)
            )
            import shutil
            shutil.copy(downloaded_path, local_file)
            print(f"[YOLODetector] Model saved to: {local_file}")
            return str(local_file)
        except Exception as e:
            print(f"[YOLODetector] Download failed: {e}, using as-is")
            return model_id

    def detect(self, image: np.ndarray) -> DetectionResult:
        if self.player_model is None:
            self.load_model()

        h, w = image.shape[:2]
        players = []
        goalkeepers = []
        ball_candidates = []

        player_results = self.player_model(image, conf=self.confidence_threshold, verbose=False)
        
        for result in player_results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                class_name = self.FOOTBALL_CLASSES.get(cls_id, f"class_{cls_id}")
                
                if cls_id == 1:
                    ball_candidates.append(BoundingBox(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name="ball"
                    ))
                elif cls_id == 3:
                    pass
                else:
                    players.append(BoundingBox(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=class_name
                    ))

        print(f"[YOLODetector] Class counts: players={len(players)}, goalkeepers={len(goalkeepers)}, ball_candidates={len(ball_candidates)}")

        if self.ball_model is not None:
            ball_results = self.ball_model(image, conf=self.ball_confidence_threshold, verbose=False)
            for result in ball_results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id == 32:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        ball_candidates.append(BoundingBox(
                            x1=float(x1), y1=float(y1),
                            x2=float(x2), y2=float(y2),
                            confidence=conf,
                            class_id=cls_id,
                            class_name="ball"
                        ))

        ball = self._filter_ball(ball_candidates, players, image)
        players = self._filter_overlapping_players(players)

        print(f"[YOLODetector] Final: {len(players)} players, {len(goalkeepers)} goalkeepers, ball={ball is not None}")

        return DetectionResult(
            players=players,
            goalkeepers=goalkeepers,
            ball=ball,
            image_shape=(h, w)
        )

    def _filter_ball(self, candidates: List[BoundingBox], players: List[BoundingBox], image: np.ndarray) -> Optional[BoundingBox]:
        if not candidates:
            return None
        
        avg_player_area = np.mean([p.area for p in players]) if players else 10000
        player_avg_height = np.mean([p.height for p in players]) if players else 100
        
        filtered = []
        
        for ball in candidates:
            ball_area = ball.area
            ball_center_x, ball_center_y = ball.center
            ball_width = ball.x2 - ball.x1
            ball_height = ball.y2 - ball.y1
            
            if ball_area < 50 or ball_area > 5000:
                print(f"[YOLODetector] Ball filtered by size: area={ball_area:.0f}")
                continue
            
            if ball_height > player_avg_height * 0.3:
                print(f"[YOLODetector] Ball filtered: ball height ({ball_height:.0f}) > 30% of player height ({player_avg_height:.0f})")
                continue
            
            is_inside_player = False
            for player in players:
                if self._point_in_bbox(ball_center_x, ball_center_y, player):
                    is_inside_player = True
                    print(f"[YOLODetector] Ball filtered: inside player")
                    break
            
            if is_inside_player:
                continue
            
            is_far_from_all = True
            min_dist = float('inf')
            for player in players:
                dist = self._distance(ball_center_x, ball_center_y, player.center)
                min_dist = min(min_dist, dist)
            
            if min_dist > ball_width * 5:
                print(f"[YOLODetector] Ball filtered: too far from all players (min_dist={min_dist:.0f})")
                continue
            
            filtered.append(ball)
        
        if not filtered:
            print(f"[YOLODetector] No valid ball candidates")
            return None
        
        best_ball = max(filtered, key=lambda b: b.confidence)
        print(f"[YOLODetector] Ball selected: area={best_ball.area:.0f}, conf={best_ball.confidence:.2f}")
        return best_ball

    def _point_in_bbox(self, x: float, y: float, bbox: BoundingBox) -> bool:
        return bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2

    def _distance(self, x1: float, y1: float, x2y2: Tuple[float, float]) -> float:
        x2, y2 = x2y2
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

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


def create_detector(
    player_model_path: str = "uisikdag/yolo-v8-football-players-detection", 
    confidence_threshold: float = 0.25, 
    ball_confidence_threshold: float = 0.01
) -> YOLODetector:
    detector = YOLODetector(player_model_path, confidence_threshold, ball_confidence_threshold)
    detector.load_model()
    return detector
