"""CPU-only frame/export audit and visual-only playback asset preparation."""
import copy
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
PHYSICAL = OUT.parents[2]
VIDEOS = PHYSICAL / "training/videos/walking_gui_20261004"
manifest = json.loads((VIDEOS / "manifest.json").read_text())
recorded_asset = Path(manifest["source_run"]) / "source/physical_v29/training/assets/robot_training.urdf"
cad_asset = PHYSICAL / "assets/humanoid_physical_v29.urdf"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(element):
    return (element.tag, dict(element.attrib), [signature(c) for c in element])


def describe(root):
    return {
        "links": len(root.findall("link")),
        "joints": len(root.findall("joint")),
        "visual_boxes": len(root.findall("./link/visual/geometry/box")),
        "visual_meshes": len(root.findall("./link/visual/geometry/mesh")),
        "collision_boxes": len(root.findall("./link/collision/geometry/box")),
        "mass_kg": sum(float(x.get("value")) for x in root.findall("./link/inertial/mass")),
    }


training = ET.parse(recorded_asset).getroot()
cad = ET.parse(cad_asset).getroot()
source_before = {str(p): sha(p) for p in [recorded_asset, cad_asset]}
tl = {x.get("name"): x for x in training.findall("link")}
cl = {x.get("name"): x for x in cad.findall("link")}
tj = {x.get("name"): x for x in training.findall("joint")}
cj = {x.get("name"): x for x in cad.findall("joint")}
assert tl.keys() == cl.keys() and tj.keys() == cj.keys()
for name in tj:
    for tag in ["parent", "child", "origin", "axis"]:
        assert signature(tj[name].find(tag)) == signature(cj[name].find(tag)), (name, tag)

hybrid = copy.deepcopy(training)
mesh_paths = {}
for link in hybrid.findall("link"):
    for visual in link.findall("visual"):
        link.remove(visual)
    for visual in cl[link.get("name")].findall("visual"):
        restored = copy.deepcopy(visual)
        for mesh in restored.findall("geometry/mesh"):
            path = (cad_asset.parent / mesh.get("filename")).resolve()
            assert path.is_file(), path
            mesh_paths[str(path)] = sha(path)
            mesh.set("filename", path.as_posix())
        link.append(restored)
asset_dir = OUT / "playback_asset"
asset_dir.mkdir(parents=True, exist_ok=True)
playback = asset_dir / "robot_training_full_visuals.urdf"
ET.indent(hybrid)
ET.ElementTree(hybrid).write(playback, encoding="utf-8", xml_declaration=True)
reloaded = ET.parse(playback).getroot()
for original, new in zip(training.findall("joint"), reloaded.findall("joint")):
    assert signature(original) == signature(new)
for original, new in zip(training.findall("link"), reloaded.findall("link")):
    assert signature(original.find("inertial")) == signature(new.find("inertial"))
    assert [signature(c) for c in original.findall("collision")] == [signature(c) for c in new.findall("collision")]
assert describe(reloaded)["visual_meshes"] == 140
assert source_before == {str(p): sha(Path(p)) for p in source_before}

joint_names = json.loads((Path(manifest["checkpoint"]).parent / "run_config.json").read_text())["joints"]
assert set(joint_names) == set(tj)
clips = []
contact_sheets = []
for clip in manifest["clips"]:
    video = Path(clip["video"])
    rows = json.loads((VIDEOS / (video.stem + "_telemetry.json")).read_text())
    assert len(rows) == 500
    cap = cv2.VideoCapture(str(video))
    frames = []
    count = 0
    selected = [0, 124, 249, 374, 499]
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if count in selected:
            frames.append(frame)
            cv2.imwrite(str(OUT / f"{video.stem}_decoded_{count:03d}.jpg"), frame)
        count += 1
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert count == 500 and abs(count / fps - 10) < 1e-6
    # Close-up source pixels only: the robot is centered by the existing camera.
    tiles = []
    for index, frame in zip(selected, frames):
        crop = frame[170:720, 380:880]
        tile = cv2.resize(crop, (300, 330))
        cv2.putText(tile, f"{clip['command_m_s']:.2f} m/s | {(index + 1) / fps:.2f}s", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    sheet = np.concatenate(tiles, axis=1)
    contact_sheets.append(sheet)
    cv2.imwrite(str(OUT / f"{video.stem}_closeups.jpg"), sheet)
    q = np.asarray([r["q"] for r in rows])
    roots = np.asarray([r["root"] for r in rows])
    angles = {}
    for name in ["head_pitch", "head_yaw", "waist_pitch", "waist_roll", "left_knee_pitch", "right_knee_pitch"]:
        v = np.degrees(q[:, joint_names.index(name)])
        angles[name] = {"min_deg": float(v.min()), "max_deg": float(v.max()), "mean_deg": float(v.mean())}
    clips.append({
        "video": str(video), "sha256": sha(video), "command_m_s": clip["command_m_s"],
        "decoded_frames": count, "fps": fps, "duration_s": count / fps,
        "inspected_frame_indices": selected,
        "root_height_range_m": [float(roots[:, 2].min()), float(roots[:, 2].max())],
        "terminated_frames": sum(r["terminated"] for r in rows),
        "angles": angles,
        "note": "Joint angles are offsets in the measured native frames, not a human-motion score.",
    })
cv2.imwrite(str(OUT / "all_speeds_closeups.jpg"), np.concatenate(contact_sheets, axis=0))
result = {
    "recorded_asset": str(recorded_asset), "source_hashes_unchanged": source_before,
    "recorded_geometry": describe(training), "cad_geometry": describe(cad),
    "source_visuals_available": len(mesh_paths), "mesh_sha256": mesh_paths,
    "same_link_names_joint_frames_axes": True,
    "playback_asset": str(playback), "playback_sha256": sha(playback),
    "playback_geometry": describe(reloaded),
    "joints_collisions_inertias_preserved_exactly": True,
    "playback_scope": "Prepared visual-only v29 asset; not converted/rendered, not installed in training, no v32 geometry migration.",
    "clips": clips,
    "finding": "Recorder used provisional box visuals. The CAD body was absent from the loaded visual asset, not lost during motion or removed by October4 execution optimization.",
}
(OUT / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: result[k] for k in ["recorded_geometry", "cad_geometry", "same_link_names_joint_frames_axes", "playback_asset", "joints_collisions_inertias_preserved_exactly", "clips"]}, indent=2))
