from typing import List, Tuple
import numpy as np
from sklearn.cluster import KMeans

from ..detection import BoundingBox


class TeamSeparator:
    def __init__(self, n_clusters: int = 2):
        self.n_clusters = n_clusters

    def separate_teams(self, players: List[BoundingBox], image: np.ndarray) -> Tuple[List[BoundingBox], List[BoundingBox]]:
        if len(players) < 2:
            return [], players

        colors = self._extract_player_colors(players, image)

        if len(colors) < self.n_clusters:
            return [], players

        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(colors)

        team1 = [p for p, label in zip(players, labels) if label == 0]
        team2 = [p for p, label in zip(players, labels) if label == 1]

        return team1, team2

    def _extract_player_colors(self, players: List[BoundingBox], image: np.ndarray) -> List[np.ndarray]:
        colors = []
        for player in players:
            x1, y1, x2, y2 = int(player.x1), int(player.y1), int(player.x2), int(player.y2)
            x1, x2 = max(0, x1), min(image.shape[1], x2)
            y1, y2 = max(0, y1), min(image.shape[0], y2)

            if x2 > x1 and y2 > y1:
                roi = image[y1:y2, x1:x2]
                avg_color = np.mean(roi.reshape(-1, 3), axis=0)
                colors.append(avg_color)

        return colors if colors else [np.array([128, 128, 128])] * len(players)


def separate_teams(players: List[BoundingBox], image: np.ndarray, n_clusters: int = 2) -> Tuple[List[BoundingBox], List[BoundingBox]]:
    separator = TeamSeparator(n_clusters)
    return separator.separate_teams(players, image)
