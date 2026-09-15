from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = (
    PROJECT_ROOT
    / "src"
    / "real_so101_vla_rl"
    / "assets"
    / "mujoco"
    / "competition_2026"
)
MUJOCO_ASSET_ROOT = ASSET_ROOT.parent
SOURCE_ASSET_ROOT = PROJECT_ROOT / "so101-nexus" / "src" / "so101_nexus" / "assets"
SCENE_NAMES = ("basic_t0", "sequence_p1_p2_p3")
JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
BATTERY_SLOT_COLORS = {
    "A": "red",
    "B": "yellow",
    "C": "blue",
    "D": "green",
}
BATTERY_BODY_NAMES = tuple(
    f"battery_{color}" for color in BATTERY_SLOT_COLORS.values()
)
BATTERY_JOINT_NAMES = tuple(f"{name}_joint" for name in BATTERY_BODY_NAMES)


@pytest.fixture(scope="module")
def layout() -> dict:
    with (ASSET_ROOT / "layouts.yaml").open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xml_path(scene_name: str) -> Path:
    return ASSET_ROOT / f"scene_{scene_name}.xml"


def _world_to_pixel(layout: dict, point_m: list[float]) -> tuple[int, int]:
    x_min, _ = layout["board"]["bounds_m"]["x"]
    _, y_max = layout["board"]["bounds_m"]["y"]
    pixels_per_meter = layout["board"]["pixels_per_meter"]
    x_m, y_m = point_m
    return (
        round((x_m - x_min) * pixels_per_meter),
        round((y_max - y_m) * pixels_per_meter),
    )


def test_source_pdf_is_archived_unchanged(layout: dict) -> None:
    source = layout["source"]
    assert _sha256(ASSET_ROOT / source["file"]) == source["sha256"]


def test_vendored_so101_assets_match_manifest_and_source() -> None:
    manifest_path = MUJOCO_ASSET_ROOT / "SO101_NEXUS_SHA256SUMS"
    checksums = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        digest, relative_path = line.split(maxsplit=1)
        checksums[relative_path] = digest

    expected_counts = {"SO101": 33, "SO101_menagerie": 26}
    for directory, expected_count in expected_counts.items():
        vendored_files = {
            path.relative_to(MUJOCO_ASSET_ROOT).as_posix()
            for path in (MUJOCO_ASSET_ROOT / directory).rglob("*")
            if path.is_file()
        }
        manifest_files = {path for path in checksums if path.startswith(f"{directory}/")}
        assert len(vendored_files) == expected_count
        assert vendored_files == manifest_files

    for relative_path, expected_digest in checksums.items():
        vendored_path = MUJOCO_ASSET_ROOT / relative_path
        assert _sha256(vendored_path) == expected_digest
        source_path = SOURCE_ASSET_ROOT / relative_path
        if source_path.is_file():
            assert _sha256(source_path) == expected_digest

    source_license = PROJECT_ROOT / "so101-nexus" / "LICENSE.md"
    vendored_license = MUJOCO_ASSET_ROOT / "SO101_NEXUS_LICENSE.md"
    assert "Apache License" in vendored_license.read_text(encoding="utf-8")
    if source_license.is_file():
        assert _sha256(vendored_license) == _sha256(source_license)


def test_competition_robot_adapter_has_only_declared_model_changes() -> None:
    source = (MUJOCO_ASSET_ROOT / "SO101_menagerie" / "so101.xml").read_text(
        encoding="utf-8"
    )
    expected = source.replace(
        "  <!--\n    Local modifications",
        """  <!--
    Competition 2026 adaptation of ../SO101_menagerie/so101.xml.
    Changes made by this project:
    - meshdir points to the immutable vendored Menagerie mesh directory;
    - the root base is rotated +90 degrees around world z so model +x aligns
      with the competition board's +y task direction.

    Local modifications""",
        1,
    )
    expected = expected.replace(
        'meshdir="assets"',
        'meshdir="../SO101_menagerie/assets"',
        1,
    )
    expected = expected.replace(
        '<body name="base" pos="0 0 0" quat="1 0 0 0" childclass="so101">',
        '<body name="base" pos="0 0 0" quat="0.70710678 0 0 0.70710678" '
        'childclass="so101">',
        1,
    )
    adapter = (ASSET_ROOT / "so101_competition.xml").read_text(encoding="utf-8")
    assert adapter == expected


def test_locked_metric_layout(layout: dict) -> None:
    board = layout["board"]
    assert board["size_m"] == [0.7, 0.55]
    assert board["texture_size_px"] == [2800, 2200]
    assert board["pixels_per_meter"] == 4000
    assert board["bounds_m"]["y"] == [-0.1, 0.45]

    common = layout["common"]
    assert common["robot_base"]["center_m"] == [0.0, 0.0]
    assert common["robot_base"]["diameter_m"] == 0.2
    assert common["placeholder_size_m"] == [0.07, 0.08]

    batteries = common["aaa_batteries"]
    assert batteries["model_file"] == "batteries_aaa.xml"
    assert batteries["diameter_m"] == 0.0105
    assert batteries["total_length_m"] == 0.0445
    assert batteries["mass_kg"] == 0.0115
    assert batteries["initial_center_z_m"] == 0.00525
    assert batteries["initial_axis"] == "world_+y"
    assert batteries["slot_to_color"] == BATTERY_SLOT_COLORS

    basic = layout["scenes"]["basic_t0"]
    assert basic["battery_frame_offset_m"] == [0.0, 0.0, 0.0]
    assert basic["source_region"]["center_m"] == [-0.1751, 0.325]
    assert basic["source_region"]["size_m"] == [0.3, 0.25]
    assert basic["target"]["center_m"] == [0.115, 0.3]
    assert basic["target"]["size_m"] == [0.2, 0.1]

    sequence = layout["scenes"]["sequence_p1_p2_p3"]
    assert sequence["battery_frame_offset_m"] == [-0.037866, 0.0, 0.0]
    assert sequence["source_region"]["center_m"] == [-0.213, 0.325]
    assert sequence["source_region"]["size_m"] == [0.3, 0.25]
    assert sequence["source_region"]["center_m"][0] == pytest.approx(
        basic["source_region"]["center_m"][0] - 0.0379
    )
    expected_centers = {
        "P1": [0.0416, 0.375],
        "P2": [0.2165, 0.375],
        "P3": [0.2165, 0.1993],
    }
    for name, center in expected_centers.items():
        target = sequence["targets"][name]
        assert target["center_m"] == center
        assert target["outer_diameter_m"] == 0.15
        assert target["inner_diameter_m"] == 0.08

    for slot, canonical_center in batteries["canonical_centers_m"].items():
        assert canonical_center == basic["placeholders"][slot]["center_m"]
        assert sequence["placeholders"][slot]["center_m"] == pytest.approx(
            [
                canonical_center[0] + sequence["battery_frame_offset_m"][0],
                canonical_center[1] + sequence["battery_frame_offset_m"][1],
            ]
        )


@pytest.mark.parametrize("scene_name", SCENE_NAMES)
def test_texture_dimensions_orientation_and_key_regions(
    layout: dict,
    scene_name: str,
) -> None:
    image_module = pytest.importorskip("PIL.Image")
    scene = layout["scenes"][scene_name]
    image = image_module.open(ASSET_ROOT / scene["texture"])
    assert image.mode == "RGB"
    assert image.size == tuple(layout["board"]["texture_size_px"])

    colors = layout["style"]["placeholder_colors_rgb"]
    # Sample away from centered text. A/C sharing x but differing y also checks
    # that +y maps towards the top of the image rather than being mirrored.
    for name, placeholder in scene["placeholders"].items():
        sample_m = [
            placeholder["center_m"][0] - 0.020,
            placeholder["center_m"][1] + 0.025,
        ]
        assert image.getpixel(_world_to_pixel(layout, sample_m)) == tuple(colors[name])

    assert image.getpixel(_world_to_pixel(layout, [0.29, 0.43])) == (255, 255, 255)


@pytest.mark.parametrize("scene_name", SCENE_NAMES)
def test_mjcf_has_separate_collision_and_texture_layers(
    layout: dict,
    scene_name: str,
) -> None:
    root = ET.parse(_xml_path(scene_name)).getroot()
    include = root.find("./include")
    texture = root.find("./asset/texture[@name='board_texture']")
    collision = root.find("./worldbody/geom[@name='board_collision']")
    visual = root.find("./worldbody/geom[@name='board_visual']")
    camera = root.find("./worldbody/camera[@name='overview']")
    battery_frame = root.find("./worldbody/frame")

    assert include is not None
    assert include.attrib["file"] == "so101_competition.xml"
    assert texture is not None
    assert texture.attrib["file"] == f"textures/{scene_name}.png"
    assert collision is not None
    assert collision.attrib["type"] == "box"
    assert collision.attrib["size"] == "0.350 0.275 0.001"
    assert collision.attrib["pos"] == "-0.042853 0.175 -0.001"
    assert visual is not None
    assert visual.attrib["type"] == "plane"
    assert visual.attrib["contype"] == "0"
    assert visual.attrib["conaffinity"] == "0"
    assert camera is not None
    assert camera.attrib["xyaxes"] == "1 0 0 0 1 0"
    assert battery_frame is not None
    assert [float(value) for value in battery_frame.attrib["pos"].split()] == (
        layout["scenes"][scene_name]["battery_frame_offset_m"]
    )
    battery_include = battery_frame.find("./include")
    assert battery_include is not None
    assert battery_include.attrib["file"] == "batteries_aaa.xml"


def test_shared_battery_mjcf_matches_layout(layout: dict) -> None:
    root = ET.parse(ASSET_ROOT / "batteries_aaa.xml").getroot()
    batteries = layout["common"]["aaa_batteries"]
    colors = layout["style"]["placeholder_colors_rgb"]

    assert root.tag == "mujocoinclude"
    bodies = root.findall("./body")
    assert tuple(body.attrib["name"] for body in bodies) == BATTERY_BODY_NAMES

    for slot, color in BATTERY_SLOT_COLORS.items():
        body_name = f"battery_{color}"
        body = root.find(f"./body[@name='{body_name}']")
        assert body is not None
        expected_center = batteries["canonical_centers_m"][slot]
        assert [float(value) for value in body.attrib["pos"].split()] == pytest.approx(
            [*expected_center, batteries["initial_center_z_m"]]
        )
        assert body.attrib["quat"] == "0.70710678 -0.70710678 0 0"

        freejoint = body.find(f"./freejoint[@name='{body_name}_joint']")
        collision = body.find(f"./geom[@name='{body_name}_collision']")
        barrel = body.find(f"./geom[@name='{body_name}_barrel']")
        assert freejoint is not None
        assert collision is not None
        assert collision.attrib["type"] == "cylinder"
        assert [float(value) for value in collision.attrib["size"].split()] == [
            batteries["diameter_m"] / 2,
            batteries["total_length_m"] / 2,
        ]
        assert float(collision.attrib["mass"]) == batteries["mass_kg"]
        assert collision.attrib["condim"] == "6"
        assert collision.attrib["friction"] == "0.8 0.005 0.0001"
        assert barrel is not None
        assert barrel.attrib["contype"] == "0"
        assert barrel.attrib["conaffinity"] == "0"
        assert [float(value) for value in barrel.attrib["rgba"].split()[:3]] == (
            pytest.approx([component / 255 for component in colors[slot]], abs=1e-6)
        )


@pytest.mark.parametrize("scene_name", SCENE_NAMES)
def test_mujoco_loads_full_robot_and_battery_scene(
    layout: dict,
    scene_name: str,
) -> None:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(_xml_path(scene_name)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert (model.nq, model.nv, model.nu) == (34, 30, 6)
    assert model.opt.timestep == pytest.approx(0.002)
    assert tuple(data.qpos[:6]) == pytest.approx((0.0,) * 6)
    assert tuple(data.ctrl) == pytest.approx((0.0,) * 6)
    assert tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, index)
                 for index in range(model.njnt)) == JOINT_NAMES + BATTERY_JOINT_NAMES
    assert tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
                 for index in range(model.nu)) == JOINT_NAMES
    assert {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, index)
        for index in range(model.ncam)
    } == {"overview", "wrist_cam"}

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripperframe")
    assert tuple(data.xpos[base_id]) == pytest.approx((0.0, 0.0, 0.0))
    assert tuple(data.xmat[base_id]) == pytest.approx(
        (0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0),
        abs=1e-7,
    )
    assert data.site_xpos[tcp_id][1] > 0.0
    assert data.site_xpos[tcp_id][0] == pytest.approx(0.000979, abs=1e-5)
    assert data.site_xpos[tcp_id][1] == pytest.approx(0.391362, abs=1e-5)

    collision_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "board_collision",
    )
    visual_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "board_visual")
    assert tuple(model.geom_size[collision_id]) == pytest.approx((0.35, 0.275, 0.001))
    assert tuple(model.geom_pos[collision_id]) == pytest.approx((-0.042853, 0.175, -0.001))
    assert model.geom_contype[visual_id] == 0
    assert model.geom_conaffinity[visual_id] == 0

    battery_config = layout["common"]["aaa_batteries"]
    scene = layout["scenes"][scene_name]
    frame_offset = scene["battery_frame_offset_m"]
    for slot, color in BATTERY_SLOT_COLORS.items():
        body_name = f"battery_{color}"
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"{body_name}_joint",
        )
        collision_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{body_name}_collision",
        )
        barrel_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            f"{body_name}_barrel",
        )
        expected_xy = scene["placeholders"][slot]["center_m"]

        assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
        assert model.body_mass[body_id] == pytest.approx(battery_config["mass_kg"])
        assert tuple(data.xpos[body_id]) == pytest.approx(
            (*expected_xy, battery_config["initial_center_z_m"])
        )
        assert model.geom_type[collision_id] == mujoco.mjtGeom.mjGEOM_CYLINDER
        assert tuple(model.geom_size[collision_id][:2]) == pytest.approx(
            (
                battery_config["diameter_m"] / 2,
                battery_config["total_length_m"] / 2,
            )
        )
        assert tuple(data.geom_xmat[collision_id][[2, 5, 8]]) == pytest.approx(
            (0.0, 1.0, 0.0),
            abs=1e-7,
        )
        assert tuple(model.geom_rgba[barrel_id][:3]) == pytest.approx(
            tuple(
                component / 255
                for component in layout["style"]["placeholder_colors_rgb"][slot]
            ),
            abs=1e-6,
        )
        canonical_xy = battery_config["canonical_centers_m"][slot]
        assert expected_xy == pytest.approx(
            [
                canonical_xy[0] + frame_offset[0],
                canonical_xy[1] + frame_offset[1],
            ]
        )

    for _ in range(500):
        mujoco.mj_step(model, data)
    assert all(math.isfinite(value) for value in data.qpos)
    assert all(math.isfinite(value) for value in data.qvel)
    assert max(abs(value) for value in data.qpos[:6]) < 0.01
    for slot, color in BATTERY_SLOT_COLORS.items():
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"battery_{color}",
        )
        expected_xy = scene["placeholders"][slot]["center_m"]
        assert tuple(data.xpos[body_id][:2]) == pytest.approx(expected_xy, abs=1e-4)
        assert data.xpos[body_id][2] == pytest.approx(
            battery_config["diameter_m"] / 2,
            abs=1e-4,
        )
