from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path


class SVGPitchGenerator:
    def __init__(self, pitch_width: float = 105, pitch_height: float = 68):
        self.pitch_width = pitch_width
        self.pitch_height = pitch_height
        self.scale = 10

    def generate_topdown_pitch_svg(
        self,
        attacker_pos: Tuple[float, float],
        defender_pos: Tuple[float, float],
        ball_pos: Optional[Tuple[float, float]] = None,
        offside_line_x: Optional[float] = None,
        offside_line_top: Optional[Tuple[float, float]] = None,
        offside_line_bottom: Optional[Tuple[float, float]] = None,
        decision: str = "UNKNOWN",
        team1_positions: List[Tuple[float, float]] = None,
        team2_positions: List[Tuple[float, float]] = None,
        image_width: int = 1280,
        image_height: int = 720,
        normalize: bool = True
    ) -> str:
        w = self.pitch_width * self.scale
        h = self.pitch_height * self.scale

        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">',
            '<style>',
            '.pitch {{ fill: #2e7d32; }}',
            '.line {{ stroke: white; stroke-width: 2; fill: none; }}',
            '.attacker {{ fill: #ff3b3b; stroke: white; stroke-width: 2; }}',
            '.defender {{ fill: #3b8bff; stroke: white; stroke-width: 2; }}',
            '.ball {{ fill: white; stroke: black; stroke-width: 1; }}',
            '.offside-line {{ stroke: #ffd700; stroke-width: 3; stroke-dasharray: 10,5; }}',
            '.label {{ font-family: Arial, sans-serif; font-size: 14px; fill: white; text-anchor: middle; }}',
            '.decision {{ font-family: Arial, sans-serif; font-size: 24px; font-weight: bold; }}',
            '.decision-offside {{ fill: #ff3b3b; }}',
            '.decision-onside {{ fill: #39ff14; }}',
            '</style>',
            '<rect width="100%" height="100%" class="pitch"/>',
            '<rect x="10" y="10" width="1030" height="660" class="line"/>',
            '<rect x="10" y="290" width="165" height="100" class="line"/>',
            '<rect x="865" y="290" width="165" height="100" class="line"/>',
            '<line x1="520" y1="10" x2="520" y2="670" class="line"/>',
            '<circle cx="520" cy="340" r="60" class="line"/>',
            '<circle cx="520" cy="340" r="5" fill="white"/>',
        ]

        def normalize_position(x, y):
            if normalize and image_width > 0 and image_height > 0:
                norm_x = (x / image_width) * self.pitch_width
                norm_y = (y / image_height) * self.pitch_height
                return norm_x, norm_y
            return x, y

        def scale_pos(x, y):
            return x * self.scale, y * self.scale

        if team1_positions:
            for i, (px, py) in enumerate(team1_positions):
                nx, ny = normalize_position(px, py)
                sx, sy = scale_pos(nx, ny)
                svg.append(f'<circle cx="{sx}" cy="{sy}" r="12" class="attacker" opacity="0.7"/>')

        if team2_positions:
            for i, (px, py) in enumerate(team2_positions):
                nx, ny = normalize_position(px, py)
                sx, sy = scale_pos(nx, ny)
                svg.append(f'<circle cx="{sx}" cy="{sy}" r="12" class="defender" opacity="0.7"/>')

        ax, ay = normalize_position(attacker_pos[0], attacker_pos[1])
        sx, sy = scale_pos(ax, ay)
        svg.append(f'<circle cx="{sx}" cy="{sy}" r="15" class="attacker"/>')
        svg.append(f'<text x="{sx}" y="{sy - 20}" class="label">ATTACKER</text>')

        dx, dy = normalize_position(defender_pos[0], defender_pos[1])
        sx, sy = scale_pos(dx, dy)
        svg.append(f'<circle cx="{sx}" cy="{sy}" r="15" class="defender"/>')
        svg.append(f'<text x="{sx}" y="{sy - 20}" class="label">2ND LAST DEF</text>')

        if ball_pos:
            bx, by = normalize_position(ball_pos[0], ball_pos[1])
            sx, sy = scale_pos(bx, by)
            svg.append(f'<circle cx="{sx}" cy="{sy}" r="6" class="ball"/>')

        if offside_line_top and offside_line_bottom:
            top_x, top_y = offside_line_top
            bottom_x, bottom_y = offside_line_bottom
            nx_top, ny_top = normalize_position(top_x, top_y)
            nx_bottom, ny_bottom = normalize_position(bottom_x, bottom_y)
            sx_top, sy_top = scale_pos(nx_top, ny_top)
            sx_bottom, sy_bottom = scale_pos(nx_bottom, ny_bottom)
            svg.append(f'<line x1="{sx_top}" y1="{sy_top}" x2="{sx_bottom}" y2="{sy_bottom}" class="offside-line"/>')
        elif offside_line_x is not None:
            nx, _ = normalize_position(offside_line_x, 0)
            lx, _ = scale_pos(nx, 0)
            svg.append(f'<line x1="{lx}" y1="10" x2="{lx}" y2="670" class="offside-line"/>')

        decision_class = "decision-offside" if decision == "OFFSIDE" else "decision-onside"
        svg.append(f'<text x="{w//2}" y="30" class="decision {decision_class}" text-anchor="middle">{decision}</text>')

        svg.append('</svg>')
        return '\n'.join(svg)

    def save_svg(self, svg_content: str, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(svg_content)


def generate_offside_svg(
    attacker_pos: Tuple[float, float],
    defender_pos: Tuple[float, float],
    ball_pos: Optional[Tuple[float, float]] = None,
    offside_line_x: Optional[float] = None,
    offside_line_top: Optional[Tuple[float, float]] = None,
    offside_line_bottom: Optional[Tuple[float, float]] = None,
    decision: str = "UNKNOWN",
    team1_positions: List[Tuple[float, float]] = None,
    team2_positions: List[Tuple[float, float]] = None,
    image_width: int = 1280,
    image_height: int = 720,
    output_path: str = "output/pitch.svg"
) -> str:
    generator = SVGPitchGenerator()
    svg_content = generator.generate_topdown_pitch_svg(
        attacker_pos, defender_pos, ball_pos, offside_line_x, offside_line_top, offside_line_bottom,
        decision, team1_positions, team2_positions, image_width, image_height
    )
    generator.save_svg(svg_content, output_path)
    return svg_content
