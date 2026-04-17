from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import cv2

from ..utils import extract_jersey_color_bgr


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
    source: str = ""

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
    referees: List[BoundingBox]
    ball: Optional[BoundingBox]
    ball_candidates: List[BoundingBox]
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
        referees = []
        football_ball_candidates = []
        generic_ball_candidates = []

        player_results = self.player_model(image, conf=self.confidence_threshold, verbose=False)
        
        all_class_ids = set()
        raw_detections = []
        
        for result in player_results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                all_class_ids.add(cls_id)
                class_name = self.FOOTBALL_CLASSES.get(cls_id, f"class_{cls_id}")
                raw_detections.append((cls_id, class_name, conf))
                
                if cls_id == 1:
                    football_ball_candidates.append(BoundingBox(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name="ball",
                        source="football"
                    ))
                elif cls_id == 2:
                    goalkeepers.append(BoundingBox(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name="goalkeeper",
                        source="football-goalkeeper"
                    ))
                elif cls_id == 3:
                    referees.append(BoundingBox(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name="referee",
                        source="football-referee"
                    ))
                else:
                    players.append(BoundingBox(
                        x1=float(x1), y1=float(y1),
                        x2=float(x2), y2=float(y2),
                        confidence=conf,
                        class_id=cls_id,
                        class_name=class_name,
                        source="football-player"
                    ))

        print(f"[YOLODetector] All unique class IDs seen: {all_class_ids}")
        print(f"[YOLODetector] Raw detections: {raw_detections}")
        print(
            f"[YOLODetector] Class counts BEFORE filter: players={len(players)}, "
            f"goalkeepers={len(goalkeepers)}, football_ball_candidates={len(football_ball_candidates)}"
        )

        if self.ball_model is not None:
            generic_ball_candidates = self._detect_generic_ball_candidates(image)

        players, goalkeepers = self._rebalance_goalkeeper_overload(players, goalkeepers, w)

        if self._should_run_player_recovery(players, goalkeepers, referees):
            recovery_confidence = max(0.08, min(self.confidence_threshold * 0.5, self.confidence_threshold - 0.05))
            recovered_players = self._recover_low_conf_players(image, recovery_confidence)
            if recovered_players:
                print(f"[YOLODetector] Low-confidence recovery added {len(recovered_players)} player candidates")
                players.extend(recovered_players)

        if len(players) < 4 and self.ball_model is not None:
            generic_person_players = self._recover_generic_person_players(image, confidence_threshold=0.18)
            if generic_person_players:
                print(f"[YOLODetector] Generic person fallback added {len(generic_person_players)} player candidates")
                players.extend(generic_person_players)

        players = self._filter_players(
            players,
            image,
            goalkeepers,
            referees,
            football_ball_candidates + generic_ball_candidates
        )
        players = self._filter_overlapping_players(players, iou_threshold=0.45)
        goalkeepers = self._filter_goalkeepers(
            goalkeepers,
            players,
            football_ball_candidates + generic_ball_candidates
        )

        players, goalkeepers = self._promote_singleton_color_goalkeeper(
            players,
            goalkeepers,
            referees,
            image,
            football_ball_candidates + generic_ball_candidates
        )

        ball_context = players + goalkeepers
        valid_ball_candidates = self._get_valid_ball_candidates(football_ball_candidates, ball_context, image)
        valid_ball_candidates.extend(self._get_valid_ball_candidates(generic_ball_candidates, ball_context, image))
        valid_ball_candidates = self._deduplicate_ball_candidates(valid_ball_candidates)
        ball = self._choose_best_ball_candidate(valid_ball_candidates, ball_context)

        print(
            f"[YOLODetector] Final: {len(players)} players, {len(goalkeepers)} goalkeepers, "
            f"{len(referees)} referees, ball={ball is not None}"
        )

        return DetectionResult(
            players=players,
            goalkeepers=goalkeepers,
            referees=referees,
            ball=ball,
            ball_candidates=valid_ball_candidates,
            image_shape=(h, w)
        )

    def _rebalance_goalkeeper_overload(
        self,
        players: List[BoundingBox],
        goalkeepers: List[BoundingBox],
        image_width: int
    ) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        if len(goalkeepers) <= 2:
            return players, goalkeepers

        suspicious_overload = (
            len(goalkeepers) > 4
            or len(players) <= 2
            or len(goalkeepers) >= len(players) + 2
        )
        if not suspicious_overload:
            return players, goalkeepers

        sorted_goalkeepers = sorted(
            goalkeepers,
            key=lambda player: (player.foot_position[0], -player.confidence)
        )
        leftmost = sorted_goalkeepers[0]
        rightmost = sorted_goalkeepers[-1]
        horizontal_spread = rightmost.foot_position[0] - leftmost.foot_position[0]

        kept_goalkeepers = [leftmost]
        if rightmost is not leftmost and horizontal_spread > image_width * 0.3:
            kept_goalkeepers.append(rightmost)

        relabeled_players = []
        for goalkeeper in goalkeepers:
            if any(goalkeeper is kept for kept in kept_goalkeepers):
                continue

            relabeled_players.append(BoundingBox(
                x1=goalkeeper.x1,
                y1=goalkeeper.y1,
                x2=goalkeeper.x2,
                y2=goalkeeper.y2,
                confidence=goalkeeper.confidence,
                class_id=0,
                class_name="player",
                source="goalkeeper-relabel"
            ))

        print(
            f"[YOLODetector] Goalkeeper overload rebalance: kept={len(kept_goalkeepers)}, "
            f"relabeled_to_players={len(relabeled_players)}, spread={horizontal_spread:.0f}"
        )
        return players + relabeled_players, kept_goalkeepers

    def _should_run_player_recovery(
        self,
        players: List[BoundingBox],
        goalkeepers: List[BoundingBox],
        referees: List[BoundingBox]
    ) -> bool:
        participant_count = len(players) + len(goalkeepers) + len(referees)
        return len(players) < 4 or participant_count < 6

    def _recover_low_conf_players(
        self,
        image: np.ndarray,
        confidence_threshold: float
    ) -> List[BoundingBox]:
        if confidence_threshold >= self.confidence_threshold:
            return []

        recovered_players = []
        results = self.player_model(image, conf=confidence_threshold, verbose=False)
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                recovered_players.append(BoundingBox(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name="player",
                    source="football-lowconf"
                ))
        return recovered_players

    def _recover_generic_person_players(
        self,
        image: np.ndarray,
        confidence_threshold: float
    ) -> List[BoundingBox]:
        recovered_players = []
        results = self.ball_model(image, conf=confidence_threshold, classes=[0], verbose=False)
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                recovered_players.append(BoundingBox(
                    x1=float(x1), y1=float(y1),
                    x2=float(x2), y2=float(y2),
                    confidence=conf,
                    class_id=0,
                    class_name="player",
                    source="generic-person"
                ))
        return recovered_players

    def _detect_generic_ball_candidates(self, image: np.ndarray) -> List[BoundingBox]:
        generic_ball_candidates = self._run_generic_ball_pass(
            image=image,
            confidence_threshold=self.ball_confidence_threshold,
            source="generic",
            upscale_factor=1.0
        )

        if len(generic_ball_candidates) < 2:
            rescue_confidence = max(0.005, self.ball_confidence_threshold * 0.5)
            rescue_candidates = self._run_generic_ball_pass(
                image=image,
                confidence_threshold=rescue_confidence,
                source="generic-upscaled",
                upscale_factor=1.5
            )
            if rescue_candidates:
                print(f"[YOLODetector] Upscaled ball pass added {len(rescue_candidates)} candidates")
                generic_ball_candidates.extend(rescue_candidates)

        return generic_ball_candidates

    def _run_generic_ball_pass(
        self,
        image: np.ndarray,
        confidence_threshold: float,
        source: str,
        upscale_factor: float
    ) -> List[BoundingBox]:
        inference_image = image
        scale_x = 1.0
        scale_y = 1.0

        if abs(upscale_factor - 1.0) > 1e-6:
            resized_width = max(1, int(round(image.shape[1] * upscale_factor)))
            resized_height = max(1, int(round(image.shape[0] * upscale_factor)))
            inference_image = cv2.resize(
                image,
                (resized_width, resized_height),
                interpolation=cv2.INTER_CUBIC
            )
            scale_x = image.shape[1] / float(resized_width)
            scale_y = image.shape[0] / float(resized_height)

        inference_size = 1280 if max(inference_image.shape[:2]) >= 960 else 960
        results = self.ball_model(
            inference_image,
            conf=confidence_threshold,
            classes=[32],
            imgsz=inference_size,
            verbose=False
        )

        candidates = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id != 32:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                candidates.append(BoundingBox(
                    x1=float(x1) * scale_x,
                    y1=float(y1) * scale_y,
                    x2=float(x2) * scale_x,
                    y2=float(y2) * scale_y,
                    confidence=conf,
                    class_id=cls_id,
                    class_name="ball",
                    source=source
                ))

        return candidates

    def _filter_players(
        self,
        candidates: List[BoundingBox],
        image: np.ndarray,
        goalkeepers: List[BoundingBox],
        referees: List[BoundingBox],
        ball_candidates: List[BoundingBox]
    ) -> List[BoundingBox]:
        if not candidates:
            return []

        image_height, image_width = image.shape[:2]
        heights = np.array([player.height for player in candidates if player.height > 0], dtype=float)
        widths = np.array([player.width for player in candidates if player.width > 0], dtype=float)
        areas = np.array([player.area for player in candidates if player.area > 0], dtype=float)

        reference_height = float(np.percentile(heights, 35)) if heights.size else max(24.0, image_height * 0.04)
        reference_width = float(np.percentile(widths, 35)) if widths.size else max(10.0, image_width * 0.01)
        reference_area = float(np.percentile(areas, 35)) if areas.size else max(160.0, image_height * image_width * 0.00012)

        min_height = max(18.0, image_height * 0.025, reference_height * 0.45)
        min_area = max(120.0, reference_area * 0.15)

        filtered = []
        for player in candidates:
            aspect_ratio = player.height / max(player.width, 1.0)

            if player.height < min_height:
                continue

            if player.area < min_area:
                continue

            if player.width > image_width * 0.65 and player.height > image_height * 0.7:
                continue

            if player.y2 < image_height * 0.12 and player.height < reference_height:
                continue

            if aspect_ratio < 0.45:
                continue

            if aspect_ratio < 0.75 and player.area < reference_area * 0.8:
                continue

            if aspect_ratio > 6.5 and player.width < reference_width * 0.5:
                continue

            if self._overlaps_similar_detection(player, goalkeepers, iou_threshold=0.55):
                continue

            if self._overlaps_similar_detection(player, referees, iou_threshold=0.55):
                continue

            if self._looks_like_ball_promoted_to_player(player, ball_candidates, reference_height):
                continue

            filtered.append(player)

        if not filtered:
            print("[YOLODetector] Player validation removed all candidates")
            return []

        if len(filtered) >= 4:
            support_heights = np.array([player.height for player in filtered if player.height > 0], dtype=float)
            support_areas = np.array([player.area for player in filtered if player.area > 0], dtype=float)
            low_height_limit = float(np.percentile(support_heights, 15) * 0.5) if support_heights.size else min_height
            low_area_limit = float(np.percentile(support_areas, 15) * 0.3) if support_areas.size else min_area

            context_filtered = []
            for player in filtered:
                is_recovered_candidate = player.source in {"football-lowconf", "generic-person"}
                if is_recovered_candidate and player.height < low_height_limit and player.area < low_area_limit:
                    continue
                context_filtered.append(player)
            filtered = context_filtered

        print(f"[YOLODetector] Player candidates after validation: {len(filtered)}/{len(candidates)}")
        return filtered

    def _filter_goalkeepers(
        self,
        candidates: List[BoundingBox],
        players: List[BoundingBox],
        ball_candidates: List[BoundingBox]
    ) -> List[BoundingBox]:
        if not candidates:
            return []

        if not players:
            return self._filter_overlapping_players(candidates, iou_threshold=0.6)

        player_heights = np.array([player.height for player in players if player.height > 0], dtype=float)
        player_areas = np.array([player.area for player in players if player.area > 0], dtype=float)

        reference_height = float(np.percentile(player_heights, 25)) if player_heights.size else 60.0
        reference_area = float(np.percentile(player_areas, 25)) if player_areas.size else 1800.0

        min_height = max(26.0, reference_height * 0.55)
        min_area = max(220.0, reference_area * 0.16)

        filtered = []
        for goalkeeper in candidates:
            aspect_ratio = goalkeeper.height / max(goalkeeper.width, 1.0)

            if goalkeeper.height < min_height:
                print(
                    f"[YOLODetector] Goalkeeper filtered: too short "
                    f"(height={goalkeeper.height:.0f}, min={min_height:.0f})"
                )
                continue

            if goalkeeper.area < min_area * 0.6:
                print(
                    f"[YOLODetector] Goalkeeper filtered: too small "
                    f"(area={goalkeeper.area:.0f}, min={min_area * 0.6:.0f})"
                )
                continue

            if goalkeeper.area < min_area and aspect_ratio < 1.35:
                print(
                    f"[YOLODetector] Goalkeeper filtered: ball-like shape "
                    f"(area={goalkeeper.area:.0f}, aspect={aspect_ratio:.2f})"
                )
                continue

            if self._overlaps_ball_candidate(goalkeeper, ball_candidates):
                print("[YOLODetector] Goalkeeper filtered: overlaps likely ball candidate")
                continue

            filtered.append(goalkeeper)

        if len(filtered) != len(candidates):
            print(f"[YOLODetector] Goalkeeper candidates after validation: {len(filtered)}/{len(candidates)}")

        return self._filter_overlapping_players(filtered, iou_threshold=0.6)

    def _promote_singleton_color_goalkeeper(
        self,
        players: List[BoundingBox],
        goalkeepers: List[BoundingBox],
        referees: List[BoundingBox],
        image: np.ndarray,
        ball_candidates: List[BoundingBox]
    ) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        if goalkeepers or referees or len(players) < 3:
            return players, goalkeepers

        candidate = self._find_singleton_color_player(players, image)
        if candidate is None:
            return players, goalkeepers

        if not self._is_player_like_singleton(candidate, players):
            print(
                f"[YOLODetector] Third-color singleton not promoted: geometry is not player-like "
                f"(height={candidate.height:.0f}, width={candidate.width:.0f})"
            )
            return players, goalkeepers

        if self._overlaps_ball_candidate(candidate, ball_candidates):
            print("[YOLODetector] Third-color singleton not promoted: overlaps likely ball candidate")
            return players, goalkeepers

        promoted_players = [player for player in players if player is not candidate]
        candidate.class_id = 2
        candidate.class_name = "goalkeeper"
        candidate.source = "third-color-yolo-box"
        promoted_goalkeepers = [candidate]
        print(
            f"[YOLODetector] Third-color YOLO player promoted to goalkeeper: "
            f"foot_pos={candidate.foot_position}, remaining_players={len(promoted_players)}"
        )
        return promoted_players, promoted_goalkeepers

    def _find_singleton_color_player(
        self,
        players: List[BoundingBox],
        image: np.ndarray
    ) -> Optional[BoundingBox]:
        try:
            from sklearn.cluster import KMeans
        except Exception as exc:
            print(f"[YOLODetector] Third-color goalkeeper promotion skipped: KMeans unavailable ({exc})")
            return None

        player_colors = [extract_jersey_color_bgr(player, image) for player in players]
        valid_players = list(players)

        if len(valid_players) < 5:
            return None

        bgr_colors = np.array(player_colors, dtype=np.uint8).reshape(-1, 1, 3)
        lab_colors = cv2.cvtColor(bgr_colors, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)

        kmeans3 = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels3 = kmeans3.fit_predict(lab_colors)
        cluster_counts = [int(np.sum(labels3 == i)) for i in range(3)]
        singleton_clusters = [i for i, count in enumerate(cluster_counts) if count == 1]

        if len(singleton_clusters) != 1:
            return None

        singleton_cluster = singleton_clusters[0]
        singleton_index = int(np.where(labels3 == singleton_cluster)[0][0])
        remaining_labels = [i for i in range(3) if i != singleton_cluster]
        if any(cluster_counts[label] < 2 for label in remaining_labels):
            print(
                f"[YOLODetector] Third-color singleton not promoted: other clusters are not both team-sized "
                f"(counts={cluster_counts})"
            )
            return None

        singleton_color = lab_colors[singleton_index]
        remaining_centers = [kmeans3.cluster_centers_[label] for label in remaining_labels]
        nearest_team_distance = min(float(np.linalg.norm(singleton_color - center)) for center in remaining_centers)

        if nearest_team_distance < 20.0:
            print(
                f"[YOLODetector] Third-color singleton not promoted: color distance too small "
                f"(distance={nearest_team_distance:.1f}, counts={cluster_counts})"
            )
            return None

        candidate = valid_players[singleton_index]
        print(
            f"[YOLODetector] Third-color singleton found from YOLO player boxes: "
            f"counts={cluster_counts}, color_distance={nearest_team_distance:.1f}"
        )
        return candidate

    def _is_player_like_singleton(
        self,
        candidate: BoundingBox,
        players: List[BoundingBox]
    ) -> bool:
        other_players = [player for player in players if player is not candidate]
        other_heights = np.array([player.height for player in other_players if player.height > 0], dtype=float)
        other_areas = np.array([player.area for player in other_players if player.area > 0], dtype=float)

        reference_height = float(np.median(other_heights)) if other_heights.size else 60.0
        reference_area = float(np.median(other_areas)) if other_areas.size else 1800.0
        aspect_ratio = candidate.height / max(candidate.width, 1.0)

        if candidate.height < max(16.0, reference_height * 0.22):
            return False

        if candidate.area < max(90.0, reference_area * 0.035):
            return False

        if aspect_ratio < 0.85 and candidate.area < reference_area * 0.12:
            return False

        return True

    def _overlaps_ball_candidate(
        self,
        goalkeeper: BoundingBox,
        ball_candidates: List[BoundingBox]
    ) -> bool:
        for ball in ball_candidates:
            center_distance = self._distance(goalkeeper.center[0], goalkeeper.center[1], ball.center)
            if center_distance > max(14.0, ball.width * 2.5):
                continue

            if self._compute_iou(goalkeeper, ball) > 0.15:
                return True

            similar_scale = (
                goalkeeper.height <= max(ball.height * 3.5, 28.0)
                and goalkeeper.width <= max(ball.width * 3.5, 28.0)
            )
            if similar_scale:
                return True

        return False

    def _overlaps_similar_detection(
        self,
        candidate: BoundingBox,
        existing_detections: List[BoundingBox],
        iou_threshold: float = 0.55
    ) -> bool:
        for other in existing_detections:
            if self._compute_iou(candidate, other) > iou_threshold:
                return True
        return False

    def _looks_like_ball_promoted_to_player(
        self,
        player: BoundingBox,
        ball_candidates: List[BoundingBox],
        reference_height: float
    ) -> bool:
        if not ball_candidates:
            return False

        aspect_ratio = player.height / max(player.width, 1.0)
        if aspect_ratio >= 1.1 and player.height >= reference_height * 0.35:
            return False

        for ball in ball_candidates:
            if self._compute_iou(player, ball) > 0.12:
                return True

            center_distance = self._distance(player.center[0], player.center[1], ball.center)
            if center_distance <= max(ball.width * 2.5, 12.0):
                similar_scale = (
                    player.height <= max(ball.height * 4.0, 28.0)
                    and player.width <= max(ball.width * 4.0, 28.0)
                )
                if similar_scale:
                    return True

        return False

    def _get_valid_ball_candidates(
        self,
        candidates: List[BoundingBox],
        participants: List[BoundingBox],
        image: np.ndarray
    ) -> List[BoundingBox]:
        if not candidates:
            return []

        image_height, image_width = image.shape[:2]
        image_area = float(image_height * image_width)
        participant_areas = np.array([p.area for p in participants if p.area > 0], dtype=float)
        participant_heights = np.array([p.height for p in participants if p.height > 0], dtype=float)

        avg_participant_area = (
            float(np.mean(participant_areas))
            if participant_areas.size else max(1600.0, image_area * 0.002)
        )
        participant_avg_height = (
            float(np.mean(participant_heights))
            if participant_heights.size else max(48.0, image_height * 0.08)
        )

        min_ball_area = max(10.0, min(28.0, image_area * 0.00002))
        max_ball_area = max(8000.0, avg_participant_area * 0.22, image_area * 0.01)
        max_ball_height = max(26.0, participant_avg_height * 0.78, image_height * 0.12)
        filtered = []

        for ball in candidates:
            ball_area = ball.area
            ball_width = max(ball.width, 1.0)
            ball_height = max(ball.height, 1.0)
            square_ratio = min(ball_width, ball_height) / max(ball_width, ball_height)
            source = ball.source or ""
            trusted_football = source == "football" and ball.confidence >= 0.2
            trusted_generic = source.startswith("generic") and ball.confidence >= 0.08
            trusted_ball = trusted_football or trusted_generic

            if ball_area < min_ball_area:
                if trusted_ball and square_ratio >= 0.45 and ball_area >= min_ball_area * 0.45:
                    pass
                else:
                    print(f"[YOLODetector] Ball filtered by size: area={ball_area:.0f}")
                    continue

            if ball_area > max_ball_area:
                if trusted_football and square_ratio >= 0.55 and ball_height <= max_ball_height * 1.15:
                    pass
                else:
                    print(f"[YOLODetector] Ball filtered by size: area={ball_area:.0f}")
                    continue

            if ball_height > max_ball_height:
                if trusted_football and ball_height <= max_ball_height * 1.2 and square_ratio >= 0.55:
                    pass
                else:
                    print(
                        f"[YOLODetector] Ball filtered: height too large "
                        f"(height={ball_height:.0f}, max={max_ball_height:.0f})"
                    )
                    continue

            if square_ratio < 0.4 and ball.confidence < 0.2:
                print(f"[YOLODetector] Ball filtered by size: area={ball_area:.0f}")
                continue

            is_inside_upper_body = False
            for participant in participants:
                if self._ball_center_inside_upper_body(ball, participant):
                    is_inside_upper_body = True
                    print(f"[YOLODetector] Ball filtered: deep inside participant body")
                    break

            if is_inside_upper_body:
                continue

            min_center_dist, min_foot_dist = self._ball_proximity(ball, participants)

            if participants:
                max_foot_dist = max(ball_width * 14.0, participant_avg_height * 1.05, 38.0)
                relaxed_foot_dist = max(ball_width * 20.0, participant_avg_height * 1.4, 62.0)
                max_center_dist = max(ball_width * 18.0, participant_avg_height * 1.55, 68.0)

                if min_foot_dist > max_foot_dist:
                    if not (trusted_ball and min_foot_dist <= relaxed_foot_dist and min_center_dist <= max_center_dist):
                        print(
                            f"[YOLODetector] Ball filtered: too far from all participant feet "
                            f"(min_foot_dist={min_foot_dist:.0f}, max={max_foot_dist:.0f})"
                        )
                        continue

                if min_center_dist > max_center_dist and min_foot_dist > relaxed_foot_dist:
                    print(
                        f"[YOLODetector] Ball filtered: too far from all participants "
                        f"(min_center_dist={min_center_dist:.0f}, max={max_center_dist:.0f})"
                    )
                    continue

            if self._looks_like_field_mark(ball, image, min_foot_dist, participant_avg_height):
                if not trusted_football:
                    print("[YOLODetector] Ball filtered: candidate looks like a field mark")
                    continue

            filtered.append(ball)

        if not filtered:
            fallback_candidates = []
            for ball in candidates:
                ball_width = max(ball.width, 1.0)
                ball_height = max(ball.height, 1.0)
                ball_area = ball.area
                square_ratio = min(ball_width, ball_height) / max(ball_width, ball_height)

                if ball_area < min_ball_area * 0.45 or ball_area > max_ball_area * 1.35:
                    continue

                if ball_height > max_ball_height * 1.25:
                    continue

                if square_ratio < 0.4 and ball.confidence < 0.2:
                    continue

                if any(self._ball_center_inside_upper_body(ball, participant) for participant in participants):
                    continue

                fallback_candidates.append(ball)

            if fallback_candidates:
                print(
                    f"[YOLODetector] Falling back to {len(fallback_candidates)} loosely validated ball candidates"
                )
                return fallback_candidates

            print(f"[YOLODetector] No valid ball candidates")
            return []

        print(f"[YOLODetector] Valid ball candidates: {len(filtered)}")
        return filtered

    def _choose_best_ball_candidate(
        self,
        candidates: List[BoundingBox],
        players: List[BoundingBox]
    ) -> Optional[BoundingBox]:
        if not candidates:
            return None

        best_ball = max(candidates, key=lambda candidate: self._ball_candidate_score(candidate, players))
        print(
            f"[YOLODetector] Selected best ball candidate across sources: "
            f"source={best_ball.source or 'unknown'}, conf={best_ball.confidence:.2f}"
        )
        return best_ball

    def _ball_candidate_score(self, ball: BoundingBox, players: List[BoundingBox]) -> float:
        score = ball.confidence

        source = ball.source or ""
        if source == "football":
            score += 0.04
        elif source.startswith("generic"):
            score += 0.02

        if ball.width > 0 and ball.height > 0:
            square_ratio = min(ball.width, ball.height) / max(ball.width, ball.height)
            score += square_ratio * 0.08

        if players:
            player_avg_height = np.mean([p.height for p in players]) if players else 100
            min_center_dist, min_foot_dist = self._ball_proximity(ball, players)
            if player_avg_height > 0:
                close_contact_range = max(ball.width * 7.0, player_avg_height * 0.3, 18.0)
                if min_foot_dist <= close_contact_range:
                    score += 0.05

                score -= min(min_foot_dist / player_avg_height, 2.0) * 0.05
                score -= min(min_center_dist / player_avg_height, 2.0) * 0.03

                if ball.height > player_avg_height * 0.85:
                    score -= 0.08

        if ball.area < 18 and ball.confidence < 0.08:
            score -= 0.06

        return score

    def _deduplicate_ball_candidates(
        self,
        candidates: List[BoundingBox],
        iou_threshold: float = 0.4,
        center_threshold: float = 12.0
    ) -> List[BoundingBox]:
        if len(candidates) <= 1:
            return candidates

        deduped = []
        for candidate in sorted(candidates, key=lambda ball: ball.confidence, reverse=True):
            is_duplicate = False
            for kept in deduped:
                if self._compute_iou(candidate, kept) > iou_threshold:
                    is_duplicate = True
                    break
                if self._distance(candidate.center[0], candidate.center[1], kept.center) < center_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduped.append(candidate)

        return deduped

    def _looks_like_field_mark(
        self,
        ball: BoundingBox,
        image: np.ndarray,
        min_foot_dist: float,
        player_avg_height: float
    ) -> bool:
        x1, y1 = max(0, int(ball.x1)), max(0, int(ball.y1))
        x2, y2 = min(image.shape[1], int(ball.x2)), min(image.shape[0], int(ball.y2))

        if x2 <= x1 or y2 <= y1:
            return False

        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_intensity = float(np.mean(gray))
        intensity_std = float(np.std(gray))

        # Painted field marks are usually bright, flat, and not in immediate foot-contact range.
        if mean_intensity > 175 and intensity_std < 16:
            contact_range = player_avg_height * 0.2 if player_avg_height > 0 else 20
            if min_foot_dist > max(contact_range, ball.width * 3):
                return True

        return False

    def _ball_center_inside_upper_body(
        self,
        ball: BoundingBox,
        participant: BoundingBox,
        lower_body_ratio: float = 0.72
    ) -> bool:
        ball_center_x, ball_center_y = ball.center
        if not self._point_in_bbox(ball_center_x, ball_center_y, participant):
            return False

        lower_body_start = participant.y1 + participant.height * lower_body_ratio
        return ball_center_y < lower_body_start

    def _ball_proximity(
        self,
        ball: BoundingBox,
        participants: List[BoundingBox]
    ) -> Tuple[float, float]:
        if not participants:
            return float("inf"), float("inf")

        ball_center_x, ball_center_y = ball.center
        ball_foot_x, ball_foot_y = ball.foot_position

        min_center_dist = min(
            self._distance(ball_center_x, ball_center_y, participant.center)
            for participant in participants
        )
        min_foot_dist = min(
            self._distance(ball_foot_x, ball_foot_y, participant.foot_position)
            for participant in participants
        )
        return min_center_dist, min_foot_dist

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
