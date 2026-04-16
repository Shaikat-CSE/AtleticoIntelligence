from typing import List, Tuple
import numpy as np
import cv2
from sklearn.cluster import KMeans
from collections import Counter

from ..detection import BoundingBox


class TeamSeparator:
    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters

    def separate_teams(self, players: List[BoundingBox], image: np.ndarray) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        if len(players) < 2:
            return [], players

        colors = self._extract_player_colors(players, image)
        hsv_colors = self._extract_hsv_colors(players, image)

        if len(colors) < self.n_clusters:
            return [], players

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(colors)

        team1 = [p for p, label in zip(players, labels) if label == 0]
        team2 = [p for p, label in zip(players, labels) if label == 1]

        team1_hsv = [hsv for hsv, label in zip(hsv_colors, labels) if label == 0]
        team2_hsv = [hsv for hsv, label in zip(hsv_colors, labels) if label == 1]

        if team1_hsv and team2_hsv:
            team1_sat = np.mean([h[1] for h in team1_hsv])
            team2_sat = np.mean([h[1] for h in team2_hsv])
            if team1_sat < team2_sat:
                return team2, team1

        return team1, team2

    def _extract_hsv_colors(self, players: List[BoundingBox], image: np.ndarray) -> List[Tuple[float, float, float]]:
        """Extract HSV colors focused on jersey region with saturation filtering."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        colors = []
        
        for player in players:
            x1, y1, x2, y2 = int(player.x1), int(player.y1), int(player.x2), int(player.y2)
            x1, x2 = max(0, x1), min(image.shape[1], x2)
            y1, y2 = max(0, y1), min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                colors.append((0, 0, 128))
                continue
            
            roi_height = y2 - y1
            roi_width = x2 - x1
            
            jersey_top = int(y1 + roi_height * 0.15)
            jersey_bottom = int(y1 + roi_height * 0.55)
            jersey_left = int(x1 + roi_width * 0.15)
            jersey_right = int(x2 - roi_width * 0.15)
            
            jersey_roi = hsv_image[jersey_top:jersey_bottom, jersey_left:jersey_right]
            
            if jersey_roi.size == 0:
                colors.append((0, 0, 128))
                continue
            
            h, s, v = cv2.split(jersey_roi)
            
            sat_mask = s > 40
            val_mask = v > 30
            
            mask = sat_mask & val_mask
            masked_s = s[mask]
            masked_h = h[mask]
            masked_v = v[mask]
            
            if len(masked_s) > 10:
                median_s = np.median(masked_s)
                median_h = np.median(masked_h)
                median_v = np.median(masked_v)
                colors.append((float(median_h), float(median_s), float(median_v)))
            else:
                colors.append((0, 0, 128))
        
        return colors if colors else [(0, 0, 128)] * len(players)

    def _extract_player_colors(self, players: List[BoundingBox], image: np.ndarray) -> List[np.ndarray]:
        """Extract dominant BGR colors from jersey region."""
        colors = []
        
        for player in players:
            x1, y1, x2, y2 = int(player.x1), int(player.y1), int(player.x2), int(player.y2)
            x1, x2 = max(0, x1), min(image.shape[1], x2)
            y1, y2 = max(0, y1), min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                colors.append(np.array([128, 128, 128]))
                continue
            
            roi_height = y2 - y1
            
            jersey_top = int(y1 + roi_height * 0.15)
            jersey_bottom = int(y1 + roi_height * 0.55)
            jersey_left = int(x1 + (x2 - x1) * 0.15)
            jersey_right = int(x2 - (x2 - x1) * 0.15)
            
            jersey_roi = image[jersey_top:jersey_bottom, jersey_left:jersey_right]
            
            if jersey_roi.size == 0:
                colors.append(np.array([128, 128, 128]))
                continue
            
            hsv_roi = cv2.cvtColor(jersey_roi, cv2.COLOR_BGR2HSV)
            s, v = cv2.split(hsv_roi)[1], cv2.split(hsv_roi)[2]
            
            mask = (s > 40) & (v > 30)
            
            if np.sum(mask) > 10:
                masked_roi = jersey_roi[mask]
                avg_color = np.mean(masked_roi.reshape(-1, 3), axis=0)
            else:
                avg_color = np.mean(jersey_roi.reshape(-1, 3), axis=0)
            
            colors.append(avg_color)
        
        return colors if colors else [np.array([128, 128, 128])] * len(players)


def separate_teams(players: List[BoundingBox], image: np.ndarray, n_clusters: int = 2) -> Tuple[List[BoundingBox], List[BoundingBox]]:
    separator = TeamSeparator(n_clusters)
    return separator.separate_teams(players, image)
