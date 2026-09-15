#!/usr/bin/env python3
"""Generate the clean 2026 competition board textures from layouts.yaml."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = (
    PROJECT_ROOT
    / "src"
    / "real_so101_vla_rl"
    / "assets"
    / "mujoco"
    / "competition_2026"
)
DEFAULT_FONT_CANDIDATES = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-root",
        type=Path,
        default=ASSET_ROOT,
        help="Directory containing layouts.yaml (defaults to the repository asset path).",
    )
    parser.add_argument(
        "--font",
        type=Path,
        help="Path to a Noto Sans CJK SC compatible TrueType/OpenType font.",
    )
    parser.add_argument(
        "--font-index",
        type=int,
        help="Face index for a .ttc font (the system Noto Sans CJK SC face is 2).",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_font(explicit_font: Path | None) -> Path:
    candidates = (explicit_font,) if explicit_font is not None else DEFAULT_FONT_CANDIDATES
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    searched = ", ".join(str(path) for path in candidates if path is not None)
    raise FileNotFoundError(
        "Noto Sans CJK SC was not found. Install fonts-noto-cjk or pass --font. "
        f"Searched: {searched}"
    )


class BoardRenderer:
    def __init__(
        self,
        config: dict[str, Any],
        font_path: Path,
        font_index: int = 0,
    ) -> None:
        board = config["board"]
        style = config["style"]
        self.config = config
        self.x_min, self.x_max = board["bounds_m"]["x"]
        self.y_min, self.y_max = board["bounds_m"]["y"]
        self.pixels_per_meter = int(board["pixels_per_meter"])
        self.image_size = tuple(int(value) for value in board["texture_size_px"])
        self.background = tuple(style["background_rgb"])
        self.zone_fill = tuple(style["zone_fill_rgb"])
        self.outline = tuple(style["outline_rgb"])
        self.inner_fill = tuple(style["inner_fill_rgb"])
        self.outline_width = self.to_pixels(style["outline_width_m"])
        self.placeholder_colors = {
            name: tuple(color) for name, color in style["placeholder_colors_rgb"].items()
        }
        self.fonts = {
            name: ImageFont.truetype(str(font_path), int(size), index=font_index)
            for name, size in style["font_sizes_px"].items()
        }

    def to_pixels(self, length_m: float) -> int:
        return max(1, round(float(length_m) * self.pixels_per_meter))

    def point(self, position_m: list[float] | tuple[float, float]) -> tuple[int, int]:
        x_m, y_m = position_m
        return (
            round((x_m - self.x_min) * self.pixels_per_meter),
            round((self.y_max - y_m) * self.pixels_per_meter),
        )

    def box(
        self,
        center_m: list[float],
        size_m: list[float],
    ) -> tuple[int, int, int, int]:
        center_x, center_y = self.point(center_m)
        half_width = self.to_pixels(size_m[0]) // 2
        half_height = self.to_pixels(size_m[1]) // 2
        return (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        )

    def centered_text(
        self,
        draw: ImageDraw.ImageDraw,
        position_m: list[float],
        text: str,
        font_name: str,
        fill: tuple[int, int, int],
    ) -> None:
        draw.text(
            self.point(position_m),
            text,
            font=self.fonts[font_name],
            fill=fill,
            anchor="mm",
            stroke_width=0,
        )

    def draw_common(
        self,
        draw: ImageDraw.ImageDraw,
        scene: dict[str, Any],
    ) -> None:
        source = scene["source_region"]
        draw.rounded_rectangle(
            self.box(source["center_m"], source["size_m"]),
            radius=self.to_pixels(source["corner_radius_m"]),
            fill=self.zone_fill,
            outline=self.outline,
            width=self.outline_width,
        )
        self.centered_text(
            draw,
            source["label_center_m"],
            source["label"],
            "source",
            self.outline,
        )

        placeholder_size = self.config["common"]["placeholder_size_m"]
        for name, placeholder in scene["placeholders"].items():
            draw.rounded_rectangle(
                self.box(placeholder["center_m"], placeholder_size),
                radius=self.to_pixels(0.008),
                fill=self.placeholder_colors[name],
            )
            self.centered_text(
                draw,
                placeholder["center_m"],
                placeholder["label"],
                "placeholder",
                (255, 255, 255),
            )

        base = self.config["common"]["robot_base"]
        diameter = float(base["diameter_m"])
        draw.ellipse(
            self.box(base["center_m"], [diameter, diameter]),
            fill=self.zone_fill,
            outline=self.outline,
            width=self.outline_width,
        )
        self.centered_text(
            draw,
            base["center_m"],
            base["label"],
            "base",
            self.outline,
        )

    def render_basic(self, scene: dict[str, Any]) -> Image.Image:
        image = Image.new("RGB", self.image_size, self.background)
        draw = ImageDraw.Draw(image)
        self.draw_common(draw, scene)
        target = scene["target"]
        draw.rounded_rectangle(
            self.box(target["center_m"], target["size_m"]),
            radius=self.to_pixels(target["corner_radius_m"]),
            fill=self.zone_fill,
            outline=self.outline,
            width=self.outline_width,
        )
        self.centered_text(
            draw,
            target["center_m"],
            target["label"],
            "target",
            self.outline,
        )
        return image

    def render_sequence(self, scene: dict[str, Any]) -> Image.Image:
        image = Image.new("RGB", self.image_size, self.background)
        draw = ImageDraw.Draw(image)
        self.draw_common(draw, scene)
        for target in scene["targets"].values():
            outer_diameter = float(target["outer_diameter_m"])
            inner_diameter = float(target["inner_diameter_m"])
            draw.ellipse(
                self.box(target["center_m"], [outer_diameter, outer_diameter]),
                fill=self.zone_fill,
                outline=self.outline,
                width=self.outline_width,
            )
            draw.ellipse(
                self.box(target["center_m"], [inner_diameter, inner_diameter]),
                fill=self.inner_fill,
                outline=self.outline,
                width=self.outline_width,
            )
            self.centered_text(
                draw,
                target["center_m"],
                target["label"],
                "target",
                self.outline,
            )
        return image


def validate_config(config: dict[str, Any], asset_root: Path) -> None:
    source = config["source"]
    source_path = asset_root / source["file"]
    actual_hash = file_sha256(source_path)
    if actual_hash != source["sha256"]:
        raise ValueError(
            f"Source PDF hash mismatch: expected {source['sha256']}, got {actual_hash}"
        )

    board = config["board"]
    width_m, height_m = board["size_m"]
    width_px, height_px = board["texture_size_px"]
    pixels_per_meter = board["pixels_per_meter"]
    if [round(width_m * pixels_per_meter), round(height_m * pixels_per_meter)] != [
        width_px,
        height_px,
    ]:
        raise ValueError("Texture dimensions do not match board size and resolution")


def main() -> None:
    args = parse_args()
    asset_root = args.asset_root.resolve()
    with (asset_root / "layouts.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    validate_config(config, asset_root)

    font_path = resolve_font(args.font)
    font_index = args.font_index
    if font_index is None:
        font_index = 2 if font_path.name == "NotoSansCJK-Regular.ttc" else 0
    renderer = BoardRenderer(config, font_path, font_index)
    renderers = {
        "basic_t0": renderer.render_basic,
        "sequence_p1_p2_p3": renderer.render_sequence,
    }
    for scene_name, render in renderers.items():
        scene = config["scenes"][scene_name]
        output_path = asset_root / scene["texture"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        render(scene).save(output_path, format="PNG", optimize=False, compress_level=9)
        print(f"generated {output_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
