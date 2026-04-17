from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import logging

from ..detection import BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class OffsideAnalysisResult:
    decision: str
    attacker: BoundingBox
    second_last_defender: BoundingBox
    confidence: float
    offside_margin_pixels: float
    goalkeeper: Optional[BoundingBox] = None
    attacking_team: str = "unknown"
    defending_team: str = "unknown"


class OffsideAnalyzer:
    def __init__(self, tolerance_pixels: float = 50.0):
        self.tolerance_pixels = tolerance_pixels

    def analyze(
        self,
        team1: List[BoundingBox],
        team2: List[BoundingBox],
        ball_position: Optional[Tuple[float, float]],
        attacking_team_input: Optional[str],
        team1_info,
        team2_info,
        goal_direction: str = "right"
    ) -> OffsideAnalysisResult:
        if not team1 or not team2:
            logger.warning("Insufficient players for offside analysis")
            return self._create_unknown_result()

        if len(team1) < 1 or len(team2) < 1:
            logger.warning("Need at least 1 player per team")
            return self._create_unknown_result()

        attacking_team, defending_team = self._determine_teams(attacking_team_input)
        
        if attacking_team == "team1":
            attacking_list = list(team1)
            defending_list = list(team2)
            team1_info_ref = team1_info
            team2_info_ref = team2_info
        else:
            attacking_list = list(team2)
            defending_list = list(team1)
            team1_info_ref = team2_info
            team2_info_ref = team1_info

        logger.info(f"Attacking team: {attacking_team}, Defending team: {defending_team}")

        goalkeeper = self._find_goalkeeper(defending_list, goal_direction)
        field_defenders = [d for d in defending_list if d != goalkeeper]
        
        if not field_defenders and goalkeeper:
            second_last_defender = goalkeeper
            logger.info("Using goalkeeper as second-last defender (no field defenders)")
        elif not field_defenders:
            second_last_defender = None
        else:
            second_last_defender = self._find_second_last_defender(field_defenders, goal_direction)

        logger.info(f"Goalkeeper: {goalkeeper.foot_position if goalkeeper else 'None'}")
        logger.info(f"Second-last defender: {second_last_defender.foot_position if second_last_defender else 'None'}")

        attacker = self._find_attacker(attacking_list, ball_position, goalkeeper)
        
        if second_last_defender is None:
            logger.warning("No second-last defender found")
            return self._create_unknown_result(attacker, goalkeeper, attacking_team, defending_team)

        attacker_x = attacker.foot_position[0]
        defender_x = second_last_defender.foot_position[0]

        if goal_direction == "right":
            is_offside = attacker_x > defender_x
        else:
            is_offside = attacker_x < defender_x

        margin = abs(attacker_x - defender_x)

        logger.info(f"Offside check: attacker_x={attacker_x:.1f}, defender_x={defender_x:.1f}, is_offside={is_offside}, margin={margin:.1f}")

        decision = "OFFSIDE" if is_offside else "ONSIDE"
        confidence = (attacker.confidence + second_last_defender.confidence) / 2 if second_last_defender else 0.5

        return OffsideAnalysisResult(
            decision=decision,
            attacker=attacker,
            second_last_defender=second_last_defender,
            confidence=confidence,
            offside_margin_pixels=margin,
            goalkeeper=goalkeeper,
            attacking_team=attacking_team,
            defending_team=defending_team
        )

    def _determine_teams(self, attacking_team_input: Optional[str]) -> Tuple[str, str]:
        if attacking_team_input == "team1":
            return "team1", "team2"
        elif attacking_team_input == "team2":
            return "team2", "team1"
        else:
            return "unknown", "unknown"

    def _find_goalkeeper(self, defenders: List[BoundingBox], goal_direction: str) -> Optional[BoundingBox]:
        if not defenders:
            return None
        if goal_direction == "right":
            return min(defenders, key=lambda p: p.foot_position[0])
        else:
            return max(defenders, key=lambda p: p.foot_position[0])

    def _find_second_last_defender(self, defenders: List[BoundingBox], goal_direction: str) -> Optional[BoundingBox]:
        if not defenders:
            return None
        if goal_direction == "right":
            return max(defenders, key=lambda p: p.foot_position[0])
        else:
            return min(defenders, key=lambda p: p.foot_position[0])

    def _find_attacker(
        self,
        attacking_list: List[BoundingBox],
        ball_position: Optional[Tuple[float, float]],
        goalkeeper: Optional[BoundingBox]
    ) -> BoundingBox:
        if ball_position is None:
            logger.warning("No ball position - using first attacker")
            return attacking_list[0]

        temp_attacker = min(
            attacking_list,
            key=lambda p: np.linalg.norm(np.array(p.foot_position) - np.array(ball_position))
        )
        
        is_behind_gk = False
        if goalkeeper is not None:
            gk_x = goalkeeper.foot_position[0]
            if self._is_behind_goalkeeper(temp_attacker, gk_x, len(attacking_list) > 1):
                logger.warning("Attacker behind goalkeeper - using fallback to first field player")
                is_behind_gk = True

        if temp_attacker == goalkeeper or is_behind_gk:
            field_attackers = [p for p in attacking_list if p != goalkeeper]
            if field_attackers:
                return field_attackers[0]
            if attacking_list:
                return attacking_list[0]
            return temp_attacker

        logger.info(f"Attacker (ball-touching): foot=({temp_attacker.foot_position[0]:.1f}, {temp_attacker.foot_position[1]:.1f})")
        return temp_attacker

    def _is_behind_goalkeeper(self, player: BoundingBox, gk_x: float, has_multiple: bool) -> bool:
        if not has_multiple:
            return False
        player_x = player.foot_position[0]
        return abs(player_x - gk_x) > 20

    def _create_unknown_result(
        self,
        attacker: Optional[BoundingBox] = None,
        goalkeeper: Optional[BoundingBox] = None,
        attacking_team: str = "unknown",
        defending_team: str = "unknown"
    ) -> OffsideAnalysisResult:
        return OffsideAnalysisResult(
            decision="UNKNOWN",
            attacker=attacker or BoundingBox(0, 0, 0, 0, 0, 0, "unknown"),
            second_last_defender=goalkeeper or BoundingBox(0, 0, 0, 0, 0, 0, "unknown"),
            confidence=0.0,
            offside_margin_pixels=0.0,
            goalkeeper=goalkeeper,
            attacking_team=attacking_team,
            defending_team=defending_team
        )


def analyze_offside(
    team1: List[BoundingBox],
    team2: List[BoundingBox],
    ball_position: Optional[Tuple[float, float]],
    attacking_team_input: Optional[str],
    team1_info,
    team2_info,
    goal_direction: str = "right"
) -> OffsideAnalysisResult:
    analyzer = OffsideAnalyzer()
    return analyzer.analyze(
        team1, team2, ball_position, attacking_team_input,
        team1_info, team2_info, goal_direction
    )
