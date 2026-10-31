"""Render synchronized locomotion video/telemetry dashboards as 15-second GIFs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT, FPS = 1280, 720, 10
BG, PANEL, PANEL_2 = "#0b0f14", "#141b24", "#1b2531"
TEXT, MUTED, GRID, ACCENT = "#edf4fb", "#9cadbf", "#344252", "#62c4ff"
COLORS = [
    "#5bc0eb", "#3388dd", "#9bc53d", "#64b547", "#fde74c", "#f5a623",
    "#fa7921", "#e94f37", "#d36ac2", "#9b5de5", "#7d8cc4", "#b8c0ff",
    "#00d4aa", "#ff6b9d", "#63e6be", "#ffd43b", "#748ffc", "#ff922b",
    "#51cf66", "#cc5de8", "#74c0fc", "#ffa94d",
]


def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    if mono:
        path = Path("C:/Windows/Fonts/consola.ttf")
    elif bold:
        path = Path("C:/Windows/Fonts/seguisb.ttf")
    else:
        path = Path("C:/Windows/Fonts/segoeui.ttf")
    return ImageFont.truetype(str(path), size)


F_TITLE, F_BODY, F_SMALL, F_MONO = font(22, True), font(12), font(10), font(10, mono=True)


def load(folder: Path) -> tuple[dict, list[dict[str, float]]]:
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    with (folder / manifest["csv"]).open(newline="", encoding="utf-8") as stream:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
    expected = int(round(float(manifest["duration_s"]) / float(manifest.get("control_dt_s", manifest.get("dt_s")))))
    if len(rows) != expected:
        raise RuntimeError(f"{folder.name}: telemetry has {len(rows)} rows, expected {expected}")
    if abs(rows[-1]["time_s"] - float(manifest["duration_s"])) > 1.0e-6:
        raise RuntimeError(f"{folder.name}: telemetry does not end at requested simulation time")
    return manifest, rows


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rectangle(box, fill=PANEL, outline=GRID, width=1)


def fit_video(frame: np.ndarray, box: tuple[int, int, int, int]) -> Image.Image:
    x0, y0, x1, y1 = box
    target_w, target_h = x1 - x0, y1 - y0
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    source = Image.fromarray(rgb)
    ratio = min(target_w / source.width, target_h / source.height)
    resized = source.resize((round(source.width * ratio), round(source.height * ratio)), Image.Resampling.LANCZOS)
    result = Image.new("RGB", (target_w, target_h), "#000000")
    result.paste(resized, ((target_w - resized.width) // 2, (target_h - resized.height) // 2))
    return result


def plot(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str,
         time_values: np.ndarray, values: np.ndarray, cursor: int, colors: list[str]) -> None:
    x0, y0, x1, y1 = box
    panel(draw, box)
    draw.text((x0 + 10, y0 + 6), title, fill=TEXT, font=F_BODY)
    left, right, top, bottom = x0 + 42, x1 - 10, y0 + 28, y1 - 22
    peak = max(1.0, float(np.nanmax(np.abs(values))) * 1.05)
    for index, fraction in enumerate((0.0, 0.5, 1.0)):
        y = round(top + fraction * (bottom - top))
        draw.line((left, y, right, y), fill=GRID, width=1)
        label = peak * (1.0 - 2.0 * fraction)
        draw.text((x0 + 3, y - 6), f"{label:.0f}", fill=MUTED, font=F_SMALL)
    duration = float(time_values[-1])
    stride = max(1, len(time_values) // 240)
    for joint in range(values.shape[1]):
        points = []
        for sample in range(0, len(time_values), stride):
            x = left + time_values[sample] / duration * (right - left)
            y = top + (peak - values[sample, joint]) / (2.0 * peak) * (bottom - top)
            points.append((round(x), round(y)))
        if len(points) > 1:
            draw.line(points, fill=colors[joint], width=1)
    cursor_x = round(left + time_values[cursor] / duration * (right - left))
    draw.line((cursor_x, top, cursor_x, bottom), fill="#ffffff", width=2)
    draw.text((left, bottom + 5), "0 s", fill=MUTED, font=F_SMALL)
    draw.text((right - 28, bottom + 5), f"{duration:.0f} s", fill=MUTED, font=F_SMALL)


def render_capture(folder: Path) -> Path:
    manifest, rows = load(folder)
    video_path = folder / manifest["video"]
    capture = cv2.VideoCapture(str(video_path))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    source_frames = round(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    expected_frames = int(manifest["control_steps"])
    if source_frames < expected_frames:
        raise RuntimeError(f"{folder.name}: video has {source_frames} frames, expected {expected_frames}")
    if abs(source_fps - float(manifest["video_fps"])) > 0.1:
        raise RuntimeError(f"{folder.name}: video FPS {source_fps} does not match simulation FPS")

    joints = manifest["joints"]
    slugs = [joint["slug"] for joint in joints]
    times = np.asarray([row["time_s"] for row in rows], dtype=np.float32)
    torque = np.asarray([[row[f"{name}_applied_torque_nm"] for name in slugs] for row in rows])
    velocity = np.asarray([[row[f"{name}_velocity_rad_s"] for name in slugs] for row in rows])
    mean_speed = float(np.mean([row["base_vx_m_s"] for row in rows if row["time_s"] >= 2.0]))
    colors = [COLORS[index % len(COLORS)] for index in range(len(joints))]

    dashboard_mp4 = folder / "dashboard.mp4"
    writer = cv2.VideoWriter(str(dashboard_mp4), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create dashboard MP4")
    step_stride = round(source_fps / FPS)
    if abs(source_fps / FPS - step_stride) > 1.0e-6:
        raise RuntimeError("Source FPS must be an integer multiple of dashboard FPS")

    source_index = -1
    next_output = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        source_index += 1
        if source_index != next_output or source_index >= expected_frames:
            continue
        row = rows[source_index]
        canvas = Image.new("RGB", (WIDTH, HEIGHT), BG)
        draw = ImageDraw.Draw(canvas)
        draw.text((16, 12), f"Five-foot humanoid — {manifest['label']}", fill=TEXT, font=F_TITLE)
        draw.text((16, 42),
                  "Synchronized physics-time playback · applied joint torque and measured joint velocity",
                  fill=MUTED, font=F_BODY)
        video_box = (16, 72, 736, 477)
        panel(draw, (10, 66, 742, 483))
        canvas.paste(fit_video(frame, video_box), (video_box[0], video_box[1]))

        panel(draw, (752, 66, 1270, 483))
        draw.rectangle((762, 76, 1260, 125), fill=PANEL_2)
        draw.text((772, 82), f"SIM {row['time_s']:05.2f} / {manifest['duration_s']:.2f} s", fill=ACCENT, font=F_BODY)
        draw.text((970, 82), f"command/actual {row['command_vx_m_s']:.2f}/{row['base_vx_m_s']:.2f} m/s", fill=TEXT, font=F_BODY)
        draw.text((772, 103), f"height {row['base_height_m']:.3f} m", fill=MUTED, font=F_SMALL)
        draw.text((970, 103), f"mean actual {mean_speed:.2f} m/s · falls {manifest.get('falls', 0)}", fill=MUTED, font=F_SMALL)
        draw.text((764, 132), "Joint", fill=MUTED, font=F_SMALL)
        draw.text((1010, 132), "τ / limit [N·m]", fill=MUTED, font=F_SMALL)
        draw.text((1147, 132), "ω / limit [rad/s]", fill=MUTED, font=F_SMALL)
        for index, joint in enumerate(joints):
            y = 147 + index * 15
            draw.ellipse((764, y + 3, 770, y + 9), fill=colors[index])
            name = joint["name"].replace("left_", "L ").replace("right_", "R ").replace("_", " ")
            draw.text((775, y), name[:25], fill=TEXT, font=F_MONO)
            effort = float(joint["effort_limit_nm"])
            speed_limit = float(joint.get("velocity_limit_rad_s", 0.0))
            draw.text((1008, y), f"{torque[source_index,index]:+6.1f}/{effort:5.1f}", fill=TEXT, font=F_MONO)
            draw.text((1150, y), f"{velocity[source_index,index]:+5.1f}/{speed_limit:4.1f}", fill=TEXT, font=F_MONO)

        plot(draw, (10, 495, 635, 710), "Applied joint torque [N·m]", times, torque, source_index, colors)
        plot(draw, (645, 495, 1270, 710), "Measured joint velocity [rad/s]", times, velocity, source_index, colors)
        writer.write(cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR))
        next_output += step_stride
    capture.release()
    writer.release()
    expected_output = round(float(manifest["duration_s"]) * FPS)
    if next_output // step_stride != expected_output:
        raise RuntimeError(f"{folder.name}: dashboard produced {next_output // step_stride} frames, expected {expected_output}")

    gif = folder / "dashboard.gif"
    command = [
        "ffmpeg", "-y", "-i", str(dashboard_mp4), "-filter_complex",
        "[0:v]split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];"
        "[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
        "-loop", "0", str(gif),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-3000:])
    manifest["dashboard_mp4"] = dashboard_mp4.name
    manifest["dashboard_gif"] = gif.name
    manifest["dashboard_fps"] = FPS
    manifest["dashboard_frames"] = expected_output
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return gif


def build_gallery(root: Path, outputs: list[Path]) -> Path:
    cards = []
    for gif in outputs:
        manifest = json.loads((gif.parent / "manifest.json").read_text(encoding="utf-8"))
        relative = gif.relative_to(root).as_posix()
        csv_link = (gif.parent / manifest["csv"]).relative_to(root).as_posix()
        mp4_link = (gif.parent / manifest["dashboard_mp4"]).relative_to(root).as_posix()
        cards.append(
            f'<section><h2>{manifest["label"]}</h2><img src="{relative}" alt="{manifest["label"]} dashboard">'
            f'<p><a href="{mp4_link}">dashboard MP4</a> · <a href="{csv_link}">telemetry CSV</a></p></section>'
        )
    html = """<!doctype html><meta charset=utf-8><title>Working locomotion reviews</title>
<style>body{margin:0;background:#0b0f14;color:#edf4fb;font:15px Segoe UI;padding:24px}main{max-width:1320px;margin:auto}section{background:#141b24;border:1px solid #344252;padding:14px;margin:18px 0}img{width:100%;height:auto}a{color:#62c4ff}h1,h2{margin-top:0}</style>
<main><h1>Five-foot humanoid — working locomotion policies</h1>""" + "".join(cards) + "</main>"
    index = root / "index.html"
    index.write_text(html, encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("folders", nargs="+")
    args = parser.parse_args()
    root = args.root.resolve()
    outputs = [render_capture(root / folder) for folder in args.folders]
    print(build_gallery(root, outputs))
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
