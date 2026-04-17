from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

from ..detection import BoundingBox
from ..utils import extract_jersey_color_profile, extract_team_color_profile


@dataclass
class TeamInfo:
    players: List[BoundingBox]
    color_bgr: Tuple[int, int, int]
    color_name: str
    goalkeeper: Optional[BoundingBox] = None
    color_confidence: float = 0.0
    color_warning: Optional[str] = None


class TeamSeparator:
    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters

    def separate_teams(
        self,
        players: List[BoundingBox],
        image: np.ndarray
    ) -> Tuple[List[BoundingBox], List[BoundingBox], Optional[BoundingBox]]:
        if len(players) < 2:
            return [], players, None

        team1_info, team2_info, goalkeeper = self.get_team_info(players, image)

        if team1_info and team2_info:
            team1_x = np.median([p.foot_position[0] for p in team1_info.players])
            team2_x = np.median([p.foot_position[0] for p in team2_info.players])

            if team1_x > team2_x:
                return team2_info.players, team1_info.players, goalkeeper

        return team1_info.players if team1_info else [], team2_info.players if team2_info else [], goalkeeper

    def get_team_info(
        self,
        players: List[BoundingBox],
        image: np.ndarray
    ) -> Tuple[Optional[TeamInfo], Optional[TeamInfo], Optional[BoundingBox]]:
        if len(players) < 2:
            return None, None, None

        player_profiles = self._extract_jersey_colors(players, image)
        if len(player_profiles) < 2:
            return None, None, None

        ordered_players = [player for player, _ in player_profiles]
        ordered_profiles = [profile for _, profile in player_profiles]
        lab_colors = np.array(
            [self._bgr_to_lab(profile.color_bgr) for profile in ordered_profiles],
            dtype=np.float32
        )

        goalkeeper_player = None

        if len(ordered_players) >= 5:
            kmeans3 = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels3 = kmeans3.fit_predict(lab_colors)
            cluster_counts = [int(np.sum(labels3 == i)) for i in range(3)]

            smallest_cluster_idx = int(np.argmin(cluster_counts))
            smallest_cluster_count = cluster_counts[smallest_cluster_idx]
            remaining_cluster_indices = [i for i in range(3) if i != smallest_cluster_idx]
            singleton_is_distinct = False

            if smallest_cluster_count == 1 and all(cluster_counts[idx] >= 2 for idx in remaining_cluster_indices):
                singleton_center = kmeans3.cluster_centers_[smallest_cluster_idx]
                remaining_centers = [kmeans3.cluster_centers_[idx] for idx in remaining_cluster_indices]
                nearest_team_distance = min(
                    float(np.linalg.norm(singleton_center - center))
                    for center in remaining_centers
                )
                singleton_is_distinct = nearest_team_distance >= 22.0

                if singleton_is_distinct:
                    print(
                        f"[TeamSeparator] Distinct third-color singleton found: "
                        f"counts={cluster_counts}, distance={nearest_team_distance:.1f}"
                    )

            if singleton_is_distinct:
                goalkeeper_index = int(np.where(labels3 == smallest_cluster_idx)[0][0])
                goalkeeper_player = ordered_players[goalkeeper_index]
                if not goalkeeper_player.source:
                    goalkeeper_player.source = "third-color-singleton"
                print(f"[TeamSeparator] Goalkeeper identified at foot_pos={goalkeeper_player.foot_position}")

                remaining_indices = [idx for idx, label in enumerate(labels3) if label != smallest_cluster_idx]
                remaining_players = [ordered_players[idx] for idx in remaining_indices]
                remaining_labs = lab_colors[remaining_indices]

                team1_info, team2_info = self._cluster_two_teams(remaining_players, remaining_labs, image)
                if team1_info and team2_info:
                    return team1_info, team2_info, goalkeeper_player

        team1_info, team2_info = self._cluster_two_teams(ordered_players, lab_colors, image)
        return team1_info, team2_info, goalkeeper_player

    def _cluster_two_teams(
        self,
        players: List[BoundingBox],
        lab_colors: np.ndarray,
        image: np.ndarray
    ) -> Tuple[Optional[TeamInfo], Optional[TeamInfo]]:
        if len(players) < 2:
            return None, None

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(lab_colors)

        team1_indices = [idx for idx, label in enumerate(labels) if label == 0]
        team2_indices = [idx for idx, label in enumerate(labels) if label == 1]

        if not team1_indices or not team2_indices:
            return None, None

        team1_players = [players[idx] for idx in team1_indices]
        team2_players = [players[idx] for idx in team2_indices]

        team1_goalkeeper = self._find_goalkeeper_by_position(team1_players, team2_players)
        team2_goalkeeper = self._find_goalkeeper_by_position(team2_players, team1_players)

        team1_info = self._build_team_info(team1_players, image, team1_goalkeeper)
        team2_info = self._build_team_info(team2_players, image, team2_goalkeeper)
        return team1_info, team2_info

    def _build_team_info(
        self,
        players: List[BoundingBox],
        image: np.ndarray,
        goalkeeper: Optional[BoundingBox]
    ) -> Optional[TeamInfo]:
        if not players:
            return None

        color_profile = extract_team_color_profile(players, image)
        return TeamInfo(
            players=players,
            color_bgr=color_profile.color_bgr,
            color_name=color_profile.color_name,
            goalkeeper=goalkeeper,
            color_confidence=color_profile.confidence,
            color_warning=color_profile.warning
        )

    def _extract_jersey_colors(self, players: List[BoundingBox], image: np.ndarray):
        player_colors = []

        for player in players:
            player_colors.append((player, extract_jersey_color_profile(player, image)))

        return player_colors

    def _find_goalkeeper_by_position(
        self,
        team_players: List[BoundingBox],
        other_team_players: List[BoundingBox]
    ) -> Optional[BoundingBox]:
        if len(team_players) <= 2:
            return None

        sorted_by_x = sorted(team_players, key=lambda p: p.foot_position[0])
        if sorted_by_x:
            return sorted_by_x[0]
        return None

    def _bgr_to_lab(self, bgr: Tuple[int, int, int]) -> np.ndarray:
        swatch = np.uint8([[list(bgr)]])
        return cv2.cvtColor(swatch, cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)


def separate_teams(
    players: List[BoundingBox],
    image: np.ndarray,
    n_clusters: int = 2
) -> Tuple[List[BoundingBox], List[BoundingBox], Optional[BoundingBox]]:
    separator = TeamSeparator(n_clusters)
    return separator.separate_teams(players, image)


def get_team_info(
    players: List[BoundingBox],
    image: np.ndarray
) -> Tuple[Optional[TeamInfo], Optional[TeamInfo], Optional[BoundingBox]]:
    separator = TeamSeparator()
    return separator.get_team_info(players, image)
