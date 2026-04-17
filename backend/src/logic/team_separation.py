from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import cv2
from sklearn.cluster import KMeans

from ..detection import BoundingBox


@dataclass
class TeamInfo:
    players: List[BoundingBox]
    color_bgr: Tuple[int, int, int]
    color_name: str
    goalkeeper: Optional[BoundingBox] = None


class TeamSeparator:
    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters

    def separate_teams(self, players: List[BoundingBox], image: np.ndarray) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        if len(players) < 2:
            return [], players

        team1_info, team2_info = self.get_team_info(players, image)
        
        if team1_info and team2_info:
            team1_x = np.median([p.foot_position[0] for p in team1_info.players])
            team2_x = np.median([p.foot_position[0] for p in team2_info.players])
            
            if team1_x > team2_x:
                return team2_info.players, team1_info.players

        return team1_info.players if team1_info else [], team2_info.players if team2_info else []

    def get_team_info(self, players: List[BoundingBox], image: np.ndarray) -> Tuple[TeamInfo, TeamInfo]:
        if len(players) < 2:
            return None, None

        player_colors = self._extract_jersey_colors(players, image)
        
        if len(player_colors) < 2:
            return None, None

        bgr_colors = np.array([p[1] for p in player_colors])
        
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        labels = kmeans.fit_predict(bgr_colors)

        team1_indices = [i for i, label in enumerate(labels) if label == 0]
        team2_indices = [i for i, label in enumerate(labels) if label == 1]
        
        team1_players = [players[i] for i in team1_indices]
        team2_players = [players[i] for i in team2_indices]
        
        team1_colors = [bgr_colors[i] for i in team1_indices]
        team2_colors = [bgr_colors[i] for i in team2_indices]

        def avg_bgr(color_list):
            if not color_list:
                return (0, 0, 0)
            return tuple(np.mean(color_list, axis=0).astype(int))

        team1_bgr_avg = avg_bgr(team1_colors)
        team2_bgr_avg = avg_bgr(team2_colors)

        if not team1_players or not team2_players:
            return None, None

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

        return team1_info, team2_info

    def _extract_jersey_colors(self, players: List[BoundingBox], image: np.ndarray) -> List[Tuple[int, np.ndarray]]:
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        player_colors = []
        
        for player in players:
            x1, y1, x2, y2 = int(player.x1), int(player.y1), int(player.x2), int(player.y2)
            x1, x2 = max(0, x1), min(image.shape[1], x2)
            y1, y2 = max(0, y1), min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                player_colors.append((player, np.array([128, 128, 128])))
                continue
            
            roi_height = y2 - y1
            
            jersey_top = int(y1 + roi_height * 0.2)
            jersey_bottom = int(y1 + roi_height * 0.5)
            jersey_left = int(x1 + (x2 - x1) * 0.2)
            jersey_right = int(x2 - (x2 - x1) * 0.2)
            
            jersey_roi = hsv_image[jersey_top:jersey_bottom, jersey_left:jersey_right]
            
            if jersey_roi.size == 0:
                player_colors.append((player, np.array([128, 128, 128])))
                continue
            
            jersey_roi_bgr = cv2.cvtColor(jersey_roi, cv2.COLOR_HSV2BGR)
            avg_bgr = np.mean(jersey_roi_bgr.reshape(-1, 3), axis=0)
            player_colors.append((player, avg_bgr))

        return player_colors

    def _find_goalkeeper_by_position(self, team_players: List[BoundingBox], other_team_players: List[BoundingBox]) -> Optional[BoundingBox]:
        if len(team_players) < 2:
            return None
        return None

    def _get_color_name_from_bgr(self, bgr: Tuple[int, int, int]) -> str:
        b, g, r = bgr
        
        r_norm = r / 255.0
        g_norm = g / 255.0
        b_norm = b / 255.0
        
        max_val = max(r_norm, g_norm, b_norm)
        min_val = min(r_norm, g_norm, b_norm)
        
        brightness = max_val
        
        if max_val - min_val < 0.15:
            if brightness > 0.85:
                return "White"
            elif brightness > 0.5:
                return "Gray"
            else:
                return "Black"
        
        if r_norm > 0.6 and g_norm > 0.4 and b_norm < 0.4:
            if g_norm > 0.5:
                return "Yellow"
            return "Orange"
        
        if r_norm > 0.7 and g_norm < 0.3 and b_norm < 0.3:
            return "Red"
        
        if r_norm < 0.3 and g_norm > 0.5 and b_norm < 0.3:
            return "Green"
        
        if r_norm < 0.3 and g_norm > 0.4 and b_norm > 0.4:
            return "Cyan"
        
        if r_norm < 0.4 and g_norm < 0.5 and b_norm > 0.5:
            return "Blue"
        
        if r_norm > 0.5 and g_norm < 0.3 and b_norm > 0.4:
            return "Purple"
        
        if r_norm > 0.7 and g_norm > 0.5 and b_norm > 0.5:
            return "Pink"
        
        if r_norm > 0.9 and g_norm > 0.9 and b_norm > 0.9:
            return "White"
        
        return f"RGB({r},{g},{b})"


def separate_teams(players: List[BoundingBox], image: np.ndarray, n_clusters: int = 2) -> Tuple[List[BoundingBox], List[BoundingBox]]:
    separator = TeamSeparator(n_clusters)
    return separator.separate_teams(players, image)


def get_team_info(players: List[BoundingBox], image: np.ndarray) -> Tuple[TeamInfo, TeamInfo]:
    separator = TeamSeparator()
    return separator.get_team_info(players, image)
