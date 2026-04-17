from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np


class SVGPitchGenerator:
    def __init__(self, pitch_width: float = 105, pitch_height: float = 68):
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.scale = 10

    def generate_topdown_pitch_svg(
        self,
        attacker_pos: Tuple[float, float],
        defender_pos: Tuple[float, float],
        goalkeeper_pos: Optional[Tuple[float, float]] = None,
        ball_pos: Optional[Tuple[float, float]] = None,
        offside_line_x: Optional[float] = None,
        offside_line_top: Optional[Tuple[float, float]] = None,
        offside_line_bottom: Optional[Tuple[float, float]] = None,
        decision: str = "UNKNOWN",
        team1_positions: List[Tuple[float, float]] = None,
        team2_positions: List[Tuple[float, float]] = None,
        image_width: int = 1280,
        image_height: int = 720,
        positions_are_pitch_coords: bool = False,
        attacking_team: str = "team1",
        goal_direction: str = "right"
    ) -> str:
        w = self.pitch_width * self.scale
        h = self.pitch_height * self.scale

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
            '<style>',
            '.pitch { fill: #2e7d32; }',
            '.line { stroke: white; stroke-width: 2; fill: none; }',
            '.attacker { fill: #ff3b3b; stroke: white; stroke-width: 2; }',
            '.defender { fill: #3b8bff; stroke: white; stroke-width: 2; }',
            '.goalkeeper { fill: #ffaa00; stroke: white; stroke-width: 2; }',
            '.ball { fill: white; stroke: black; stroke-width: 1; }',
            '.offside-line { stroke: #ffd700; stroke-width: 3; stroke-dasharray: 10,5; }',
            '.label { font-family: Arial, sans-serif; font-size: 14px; fill: white; text-anchor: middle; }',
            '.decision { font-family: Arial, sans-serif; font-size: 24px; font-weight: bold; }',
            '.decision-offside { fill: #ff3b3b; }',
            '.decision-onside { fill: #39ff14; }',
            '</style>',
            '<rect width="100%" height="100%" class="pitch"/>',
            '<rect x="10" y="10" width="1030" height="660" class="line"/>',
            '<rect x="10" y="290" width="165" height="100" class="line"/>',
            '<rect x="865" y="290" width="165" height="100" class="line"/>',
            '<line x1="520" y1="10" x2="520" y2="670" class="line"/>',
            '<circle cx="520" cy="340" r="60" class="line"/>',
            '<circle cx="520" cy="340" r="5" fill="white"/>',
        ]

        # Add attack direction indicator at BOTTOM of pitch based on goal_direction
        # Show arrow pointing in the attack direction
        if goal_direction == "right":
            svg.append('<path d="M 50 620 L 950 620 M 920 590 L 950 620 L 920 650" stroke="#00ff00" stroke-width="5" fill="none" stroke-linecap="round"/>')
            svg.append('<text x="500" y="580" class="label" style="fill: #00ff00; font-size: 16px; font-weight: bold;">ATTACK DIRECTION ==&gt;</text>')
            svg.append('<text x="500" y="665" class="label" style="fill: #ffff00; font-size: 12px;">Goal being attacked: LEFT (x=0)</text>')
        else:
            svg.append('<path d="M 950 620 L 50 620 M 80 590 L 50 620 L 80 650" stroke="#00ff00" stroke-width="5" fill="none" stroke-linecap="round"/>')
            svg.append('<text x="500" y="580" class="label" style="fill: #00ff00; font-size: 16px; font-weight: bold;">&lt;== ATTACK DIRECTION</text>')
            svg.append('<text x="500" y="665" class="label" style="fill: #ffff00; font-size: 12px;">Goal being attacked: RIGHT (x=105)</text>')

        def to_svg_coords(x, y):
            """Convert position - already in SVG coordinates, just return as-is."""
            return x, y

        def draw_player(cx, cy, team_class, is_attacker=False, has_ball=False):
            """Draw player circle with optional ball indicator."""
            radius = 15 if is_attacker else 12
            opacity = 1.0 if is_attacker else 0.7
            extra_stroke = 'stroke-width: 4;' if is_attacker else 'stroke-width: 2;'
            extra = ''
            if has_ball:
                extra = f'<circle cx="{cx}" cy="{cy}" r="{radius + 8}" fill="none" stroke="#ffd700" stroke-width="3" stroke-dasharray="5,3"/>'
                extra += f'<text x="{cx}" y="{cy - radius - 12}" class="label" style="font-size: 12px; fill: #ffd700;">⚽ BALL</text>'
            return f'{extra}<circle cx="{cx}" cy="{cy}" r="{radius}" class="{team_class}" style="{extra_stroke}" opacity="{opacity}"/>'

        team1_class = "attacker" if attacking_team == "team1" else "defender"
        team2_class = "attacker" if attacking_team == "team2" else "defender"

        if team1_positions:
            for i, (px, py) in enumerate(team1_positions):
                sx, sy = to_svg_coords(px, py)
                svg.append(f'<circle cx="{sx}" cy="{sy}" r="12" class="{team1_class}" opacity="0.7"/>')

        if team2_positions:
            for i, (px, py) in enumerate(team2_positions):
                sx, sy = to_svg_coords(px, py)
                svg.append(f'<circle cx="{sx}" cy="{sy}" r="12" class="{team2_class}" opacity="0.7"/>')

        defender_color = "#3b8bff"
        
        # Only draw attacker if valid position (not 0,0)
        if attacker_pos[0] > 0 and attacker_pos[1] > 0:
            ax, ay = to_svg_coords(attacker_pos[0], attacker_pos[1])
            has_ball = ball_pos is not None and np.linalg.norm(np.array(attacker_pos) - np.array(ball_pos)) < 3
            svg.append(draw_player(ax, ay, "attacker", is_attacker=True, has_ball=has_ball))
            svg.append(f'<text x="{ax}" y="{ay - 30}" class="label" style="font-weight: bold; fill: #ff3b3b;">ATTACKER</text>')

        # Only draw defender if valid position (not 0,0)
        if defender_pos[0] > 0 and defender_pos[1] > 0:
            dx, dy = to_svg_coords(defender_pos[0], defender_pos[1])
            svg.append(f'<circle cx="{dx}" cy="{dy}" r="18" fill="{defender_color}" stroke="white" stroke-width="3"/>')
            svg.append(f'<text x="{dx}" y="{dy + 40}" class="label" style="font-weight: bold;">DEFENDER</text>')

        # Draw goalkeeper (orange circle)
        if goalkeeper_pos and goalkeeper_pos[0] > 0 and goalkeeper_pos[1] > 0:
            gx, gy = to_svg_coords(goalkeeper_pos[0], goalkeeper_pos[1])
            svg.append(f'<circle cx="{gx}" cy="{gy}" r="18" fill="#ffaa00" stroke="white" stroke-width="3"/>')
            svg.append(f'<text x="{gx}" y="{gy + 40}" class="label" style="font-weight: bold; fill: #ffaa00;">GOALKEEPER</text>')

        if ball_pos:
            bx, by = to_svg_coords(ball_pos[0], ball_pos[1])
            svg.append(f'<circle cx="{bx}" cy="{by}" r="6" class="ball"/>')

        # Draw offside line based on decision:
        # - OFFSIDE: line at attacker's position (attacker is ahead)
        # - ONSIDE: line at defender's position (defender is ahead)
        line_x = None
        if decision == "OFFSIDE":
            line_x = attacker_pos[0] if attacker_pos[0] > 0 else defender_pos[0]
        else:  # ONSIDE
            line_x = defender_pos[0] if defender_pos[0] > 0 else attacker_pos[0]
        
        if line_x is not None and line_x > 0:
            lx, _ = to_svg_coords(line_x, 0)
            svg.append(f'<line x1="{lx}" y1="10" x2="{lx}" y2="670" class="offside-line"/>')
            line_label = "OFFSIDE LINE" if decision == "OFFSIDE" else "ONSIDE LINE"
            svg.append(f'<text x="{lx}" y="80" class="label" style="font-weight: bold; fill: #ffd700;">{line_label}</text>')

        decision_class = "decision-offside" if decision == "OFFSIDE" else "decision-onside"
        svg.append(f'<text x="{w//2}" y="30" class="decision {decision_class}" text-anchor="middle">{decision}</text>')

        svg.append('</svg>')
        return '\n'.join(svg)

    def save_svg(self, svg_content: str, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(svg_content)


def generate_offside_svg(
    attacker_pos: Tuple[float, float],
    defender_pos: Tuple[float, float],
    goalkeeper_pos: Optional[Tuple[float, float]] = None,
    ball_pos: Optional[Tuple[float, float]] = None,
    offside_line_x: Optional[float] = None,
    offside_line_top: Optional[Tuple[float, float]] = None,
    offside_line_bottom: Optional[Tuple[float, float]] = None,
    decision: str = "UNKNOWN",
    team1_positions: List[Tuple[float, float]] = None,
    team2_positions: List[Tuple[float, float]] = None,
    image_width: int = 1280,
    image_height: int = 720,
    output_path: str = "output/pitch.svg",
    attacking_team: str = "team1",
    goal_direction: str = "right"
) -> str:
    generator = SVGPitchGenerator()
    svg_content = generator.generate_topdown_pitch_svg(
        attacker_pos, defender_pos, goalkeeper_pos, ball_pos, offside_line_x, offside_line_top, offside_line_bottom,
        decision, team1_positions, team2_positions, image_width, image_height,
        attacking_team=attacking_team, goal_direction=goal_direction
    )
    generator.save_svg(svg_content, output_path)
    return svg_content
