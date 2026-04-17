from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2
from sklearn.cluster import KMeans

from ..detection import BoundingBox
from ..utils import get_color_name_from_bgr, extract_jersey_color_bgr


@dataclass
class TeamInfo:
    players: List[BoundingBox]
    color_bgr: Tuple[int, int, int]
    color_name: str
    goalkeeper: Optional[BoundingBox] = None


class TeamSeparator:
    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters

    def separate_teams(self, players: List[BoundingBox], image: np.ndarray) -> Tuple[List[BoundingBox], List[BoundingBox], Optional[BoundingBox]]:
        if len(players) < 2:
            return [], players, None

        team1_info, team2_info, goalkeeper = self.get_team_info(players, image)
        
        if team1_info and team2_info:
            team1_x = np.median([p.foot_position[0] for p in team1_info.players])
            team2_x = np.median([p.foot_position[0] for p in team2_info.players])
            
            if team1_x > team2_x:
                return team2_info.players, team1_info.players, goalkeeper

        return team1_info.players if team1_info else [], team2_info.players if team2_info else [], goalkeeper

    def get_team_info(self, players: List[BoundingBox], image: np.ndarray) -> Tuple[TeamInfo, TeamInfo, Optional[BoundingBox]]:
        if len(players) < 2:
            return None, None, None

        player_colors = self._extract_jersey_colors(players, image)
        
        if len(player_colors) < 2:
            return None, None, None

        bgr_colors = np.array([p[1] for p in player_colors])
        
        goalkeeper_player = None
        
        if len(players) >= 3:
            kmeans3 = KMeans(n_clusters=3, random_state=42, n_init=10)
            labels3 = kmeans3.fit_predict(bgr_colors)
            cluster_counts = [np.sum(labels3 == i) for i in range(3)]
            
            smallest_cluster_idx = np.argmin(cluster_counts)
            smallest_cluster_count = cluster_counts[smallest_cluster_idx]
            
            if smallest_cluster_count == 1:
                print(
                    f"[TeamSeparator] Detected 3-cluster: counts={cluster_counts}, "
                    f"smallest cluster {smallest_cluster_idx} has 1 player - treating as distinct goalkeeper"
                )
                
                goalkeeper_indices = [i for i, label in enumerate(labels3) if label == smallest_cluster_idx]
                if goalkeeper_indices:
                    goalkeeper_player = players[goalkeeper_indices[0]]
                    if not goalkeeper_player.source:
                        goalkeeper_player.source = "third-color-singleton"
                    print(f"[TeamSeparator] Goalkeeper identified at foot_pos={goalkeeper_player.foot_position}")
                
                remaining_indices = [i for i, label in enumerate(labels3) if label != smallest_cluster_idx]
                remaining_colors = bgr_colors[remaining_indices]
                remaining_players = [players[i] for i in remaining_indices]
                
                if len(remaining_players) >= 2:
                    kmeans2 = KMeans(n_clusters=2, random_state=42, n_init=10)
                    labels2 = kmeans2.fit_predict(remaining_colors)
                    
                    team1_indices = [i for i, label in enumerate(labels2) if label == 0]
                    team2_indices = [i for i, label in enumerate(labels2) if label == 1]
                    
                    team1_players = [remaining_players[i] for i in team1_indices]
                    team2_players = [remaining_players[i] for i in team2_indices]
                    
                    team1_colors = [remaining_colors[i] for i in team1_indices]
                    team2_colors = [remaining_colors[i] for i in team2_indices]
                    
                    team1_bgr_avg = self._avg_bgr(team1_colors)
                    team2_bgr_avg = self._avg_bgr(team2_colors)
                    
                    if not team1_players or not team2_players:
                        return None, None, goalkeeper_player
                    
                    team1_info = TeamInfo(
                        players=team1_players,
                        color_bgr=team1_bgr_avg,
                        color_name=self._get_color_name_from_bgr(team1_bgr_avg),
                        goalkeeper=None
                    )
                    team2_info = TeamInfo(
                        players=team2_players,
                        color_bgr=team2_bgr_avg,
                        color_name=self._get_color_name_from_bgr(team2_bgr_avg),
                        goalkeeper=None
                    )
                    return team1_info, team2_info, goalkeeper_player
        
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(bgr_colors)

        team1_indices = [i for i, label in enumerate(labels) if label == 0]
        team2_indices = [i for i, label in enumerate(labels) if label == 1]
        
        team1_players = [players[i] for i in team1_indices]
        team2_players = [players[i] for i in team2_indices]
        
        team1_colors = [bgr_colors[i] for i in team1_indices]
        team2_colors = [bgr_colors[i] for i in team2_indices]

        team1_bgr_avg = self._avg_bgr(team1_colors)
        team2_bgr_avg = self._avg_bgr(team2_colors)

        if not team1_players or not team2_players:
            return None, None, None

        team1_goalkeeper = self._find_goalkeeper_by_position(team1_players, team2_players)
        team2_goalkeeper = self._find_goalkeeper_by_position(team2_players, team1_players)

        team1_info = TeamInfo(
            players=team1_players,
            color_bgr=team1_bgr_avg,
            color_name=self._get_color_name_from_bgr(team1_bgr_avg),
            goalkeeper=team1_goalkeeper
        )
        team2_info = TeamInfo(
            players=team2_players,
            color_bgr=team2_bgr_avg,
            color_name=self._get_color_name_from_bgr(team2_bgr_avg),
            goalkeeper=team2_goalkeeper
        )

        return team1_info, team2_info, None

    def _avg_bgr(self, color_list):
        if not color_list:
            return (0, 0, 0)
        return tuple(np.mean(color_list, axis=0).astype(int))

    def _extract_jersey_colors(self, players: List[BoundingBox], image: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        player_colors = []
        
        for player in players:
            player_colors.append((player, extract_jersey_color_bgr(player, image)))

        return player_colors

    def _find_goalkeeper_by_position(self, team_players: List[BoundingBox], other_team_players: List[BoundingBox]) -> Optional[BoundingBox]:
        if len(team_players) < 2:
            return None
        
        if len(team_players) == 2:
            return None
            
        sorted_by_x = sorted(team_players, key=lambda p: p.foot_position[0])
        
        if len(sorted_by_x) > 0:
            return sorted_by_x[0]
        return None

    def _get_color_name_from_bgr(self, bgr: Tuple[int, int, int]) -> str:
        return get_color_name_from_bgr(bgr)


def separate_teams(players: List[BoundingBox], image: np.ndarray, n_clusters: int = 2) -> Tuple[List[BoundingBox], List[BoundingBox], Optional[BoundingBox]]:
    separator = TeamSeparator(n_clusters)
    return separator.separate_teams(players, image)


def get_team_info(players: List[BoundingBox], image: np.ndarray) -> Tuple[TeamInfo, TeamInfo, Optional[BoundingBox]]:
    separator = TeamSeparator()
    return separator.get_team_info(players, image)
