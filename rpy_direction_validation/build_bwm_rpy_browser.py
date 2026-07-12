#!/usr/bin/env python3
"""Build a static browser for rpy_grid_rpy_rotation BWM outputs."""

from __future__ import annotations

import argparse
import html
import json
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "robotwin_bwm"
    / "rpy_grid_rpy_rotation_position_only_clean_uncropped"
)
DEFAULT_SCENE_INFO = (
    PROJECT_ROOT
    / "third_party"
    / "robotwin"
    / "data"
    / "rpy_grid_rpy_rotation"
    / "rpy_grid_rpy_rotation_position_only_clean"
    / "scene_info.json"
)
ANGLE_ORDER = {30: 0, -30: 1, 90: 2, -90: 3, 180: 4, -180: 5}
AXIS_ORDER = {"roll": 0, "pitch": 1, "yaw": 2}
START_LABELS = {
    "move_to_point": "从初始位置移动到点位再旋转",
    "direct_at_point": "已在点位直接旋转",
}
AXIS_LABELS = {"roll": "Roll", "pitch": "Pitch", "yaw": "Yaw"}
ROBOTWIN_PLAYBACK_RATE = 24 / 30
WM_PLAYBACK_RATE = 1.0


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def rel(path: str | Path, base: Path) -> str:
    return Path(path).resolve().relative_to(base.resolve()).as_posix()


def load_records(output_dir: Path, scene_info_path: Path) -> list[dict]:
    manifest_path = output_dir / "manifest.jsonl"
    scene_info = json.loads(scene_info_path.read_text(encoding="utf-8"))
    records = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            manifest = json.loads(line)
            episode_id = int(manifest["episode_index"])
            scene_record = scene_info[f"episode_{episode_id}"]
            info = scene_record.get("info", scene_record)
            record = {
                "episode_id": episode_id,
                "start_mode": info["start_mode"],
                "point_id": int(info["left_point_id"]),
                "grid_index": info["left_target_grid_index"],
                "axis": info["axis"],
                "angle_deg": int(info["angle_deg"]),
                "num_frames": int(manifest["num_frames"]),
                "start_frame": int(manifest["start_frame"]),
                "robotwin_video": rel(manifest["converted_video"], output_dir),
                "wm_video": rel(manifest["bwm_output"], output_dir),
            }
            records.append(record)
    return sorted(
        records,
        key=lambda r: (
            r["start_mode"] != "move_to_point",
            r["point_id"],
            AXIS_ORDER[r["axis"]],
            ANGLE_ORDER.get(r["angle_deg"], 99),
            r["episode_id"],
        ),
    )


def option_tags(values: list[tuple[str, str]]) -> str:
    return "".join(f'<option value="{esc(value)}">{esc(label)}</option>' for value, label in values)


def metric(value: object, label: str) -> str:
    return f'<div class="metric"><b>{esc(value)}</b><span>{esc(label)}</span></div>'


def episode_html(record: dict) -> str:
    episode = record["episode_id"]
    angle = record["angle_deg"]
    direction = "+" if angle > 0 else "-"
    grid = ",".join(str(v) for v in record["grid_index"])
    label = (
        f"point {record['point_id']} / {AXIS_LABELS[record['axis']]} / "
        f"{angle} deg / ep{episode}"
    )
    return f"""
    <details class="episode" data-episode="{episode}" data-angle="{angle}">
      <summary>
        <div class="summary-title">
          <span class="ep-num">Episode {episode}</span>
          <span class="direction">angle {angle} deg</span>
          <span class="badge">direction {direction}</span>
          <span class="badge">grid [{esc(grid)}]</span>
          <span class="badge">{record['num_frames']} frames</span>
        </div>
        <span class="note">点击展开播放</span>
      </summary>
      <div class="video-wrap">
        <section class="clip" data-clip="uncropped">
          <div class="clip-head">
            <h3>完整首帧到旋转结束</h3>
            <span>position-only preparation; start_frame={record['start_frame']}; {esc(label)}</span>
          </div>
          <div class="controls">
            <button class="primary sync-play">同步播放</button>
            <button class="sync-pause">同步暂停</button>
            <button class="sync-reset">回到开头</button>
          </div>
          <div class="video-grid">
            <div class="panel">
              <div class="panel-title sim"><span>RoboTwin 仿真器</span><span>{esc(label)}</span></div>
              <video class="sim-video" preload="metadata" controls data-playback-rate="{ROBOTWIN_PLAYBACK_RATE:.6f}" src="{esc(record['robotwin_video'])}"></video>
            </div>
            <div class="panel">
              <div class="panel-title wm"><span>WM 生成结果</span><span>{esc(label)}</span></div>
              <video class="wm-video" preload="metadata" controls data-playback-rate="{WM_PLAYBACK_RATE:.6f}" src="{esc(record['wm_video'])}"></video>
            </div>
          </div>
        </section>
      </div>
    </details>"""


def run_card_html(key: tuple[str, int, str], records: list[dict]) -> str:
    start_mode, point_id, axis = key
    grid = records[0]["grid_index"]
    angles = ", ".join(str(r["angle_deg"]) for r in records)
    title = f"{START_LABELS[start_mode]} / point {point_id} / {AXIS_LABELS[axis]}"
    code = f"{start_mode}_point_{point_id:02d}_{axis}"
    episodes = "\n".join(episode_html(record) for record in records)
    return f"""
    <article class="run-card" data-start="{esc(start_mode)}" data-point="{point_id}" data-axis="{esc(axis)}">
      <div class="run-head">
        <div>
          <h2>{esc(title)}</h2>
          <p class="note"><code>{esc(code)}</code></p>
        </div>
        <div class="badges">
          <span class="badge">{len(records)} episodes</span>
          <span class="badge">grid [{esc(','.join(str(v) for v in grid))}]</span>
          <span class="badge">angles: {esc(angles)}</span>
        </div>
      </div>
      <div class="episodes">{episodes}
      </div>
    </article>"""


def build_html(records: list[dict], output_dir: Path) -> str:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    for record in records:
        grouped[(record["start_mode"], record["point_id"], record["axis"])].append(record)

    start_counts = Counter(r["start_mode"] for r in records)
    axis_counts = Counter(r["axis"] for r in records)
    angles = sorted({r["angle_deg"] for r in records}, key=lambda a: ANGLE_ORDER.get(a, 99))
    points = sorted({r["point_id"] for r in records})
    cards = "\n".join(
        run_card_html(key, grouped[key])
        for key in sorted(grouped, key=lambda k: (k[0] != "move_to_point", k[1], AXIS_ORDER[k[2]]))
    )

    point_options = [("all", "全部")] + [(str(point), f"point {point}") for point in points]
    angle_options = [("all", "全部")] + [(str(angle), f"{angle} deg") for angle in angles]
    axis_options = [("all", "全部"), ("roll", "Roll"), ("pitch", "Pitch"), ("yaw", "Yaw")]
    start_options = [
        ("all", "全部"),
        ("move_to_point", START_LABELS["move_to_point"]),
        ("direct_at_point", START_LABELS["direct_at_point"]),
    ]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rpy_grid_rpy_rotation position-only uncropped 浏览器</title>
<style>
:root {{ --fg:#17202a; --muted:#667085; --line:#d8dee8; --bg:#f5f7fb; --card:#fff; --accent:#2f6fed; --sim:#34495e; --wm:#8e44ad; --full:#b45309; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--fg); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; line-height:1.5; }}
main {{ max-width:1360px; margin:0 auto; padding:24px; }}
h1 {{ margin:0 0 6px; font-size:28px; letter-spacing:0; }}
h2 {{ margin:0; font-size:18px; letter-spacing:0; }}
h3 {{ margin:0; font-size:15px; letter-spacing:0; }}
p {{ margin:8px 0; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; }}
.header {{ margin-bottom:16px; }}
.toolbar {{ position:sticky; top:0; z-index:10; background:rgba(245,247,251,.96); border:1px solid var(--line); border-radius:8px; padding:12px; margin:14px 0; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
.toolbar label {{ display:flex; align-items:center; gap:6px; color:var(--muted); font-size:13px; }}
select, button {{ border:1px solid var(--line); background:#fff; color:var(--fg); border-radius:6px; padding:7px 10px; font-size:13px; }}
button {{ cursor:pointer; }}
button.primary {{ background:var(--accent); color:#fff; border-color:var(--accent); }}
.summary {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px; margin:14px 0; }}
.metric {{ background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px; }}
.metric b {{ display:block; font-size:22px; }}
.metric span {{ color:var(--muted); font-size:13px; }}
.run-card {{ background:#fff; border:1px solid var(--line); border-radius:8px; margin:14px 0; overflow:hidden; }}
.run-head {{ padding:14px 16px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; gap:12px; }}
.badges {{ display:flex; gap:8px; flex-wrap:wrap; }}
.badge {{ display:inline-flex; align-items:center; border:1px solid var(--line); border-radius:999px; padding:3px 9px; color:var(--muted); font-size:12px; background:#fafbfc; }}
details {{ border-top:1px solid var(--line); }}
details:first-child {{ border-top:0; }}
summary {{ cursor:pointer; list-style:none; padding:12px 16px; display:flex; align-items:center; justify-content:space-between; gap:10px; }}
summary::-webkit-details-marker {{ display:none; }}
.summary-title {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
.ep-num {{ width:96px; font-weight:700; }}
.direction {{ color:var(--muted); }}
.video-wrap {{ padding:0 16px 16px; display:grid; gap:14px; }}
.clip {{ border:1px solid var(--line); border-radius:8px; background:#fbfcfe; padding:12px; }}
.clip-head {{ display:flex; justify-content:space-between; gap:12px; align-items:flex-start; margin-bottom:8px; }}
.clip-head span {{ color:var(--muted); font-size:13px; max-width:780px; text-align:right; }}
.clip[data-clip="uncropped"] h3 {{ color:var(--full); }}
.video-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; align-items:start; }}
.panel {{ border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#111; }}
.panel-title {{ display:flex; justify-content:space-between; align-items:center; gap:8px; padding:8px 10px; color:#fff; font-size:13px; }}
.panel-title.sim {{ background:var(--sim); }}
.panel-title.wm {{ background:var(--wm); }}
video {{ display:block; width:100%; background:#111; aspect-ratio:4/3; }}
.controls {{ display:flex; gap:8px; flex-wrap:wrap; margin:8px 0 10px; }}
.note {{ color:var(--muted); font-size:13px; }}
.hidden {{ display:none !important; }}
@media (max-width:900px) {{ main {{ padding:14px; }} .summary {{ grid-template-columns:1fr 1fr; }} .video-grid {{ grid-template-columns:1fr; }} .run-head, .clip-head {{ align-items:flex-start; flex-direction:column; }} .clip-head span {{ text-align:left; }} }}
</style>
</head>
<body>
<main>
  <section class="header">
    <h1>rpy_grid_rpy_rotation position-only uncropped 浏览器</h1>
    <p class="note">所有视频都直接引用 <code>{esc(output_dir.name)}</code> 目录下的现有 MP4。RoboTwin 在网页播放时按 0.8x 对齐 WM 的 24 fps，不改动原视频文件。</p>
  </section>
  <section class="summary">
    {metric(len(grouped), "实验 run")}
    {metric(len(records), "episode")}
    {metric(start_counts.get("move_to_point", 0), "move_to_point")}
    {metric(start_counts.get("direct_at_point", 0), "direct_at_point")}
    {metric(len(points), "point id")}
    {metric(axis_counts.get("roll", 0), "roll")}
    {metric(axis_counts.get("pitch", 0), "pitch")}
    {metric(axis_counts.get("yaw", 0), "yaw")}
  </section>
  <section class="toolbar">
    <label>起点 <select id="startFilter">{option_tags(start_options)}</select></label>
    <label>点位 <select id="pointFilter">{option_tags(point_options)}</select></label>
    <label>轴 <select id="axisFilter">{option_tags(axis_options)}</select></label>
    <label>角度 <select id="angleFilter">{option_tags(angle_options)}</select></label>
    <button id="expandAll">展开全部</button>
    <button id="collapseAll">收起全部</button>
    <button id="pauseAll">暂停全部</button>
  </section>
  <section id="runs">{cards}
  </section>
</main>
<script>
function pauseAll() {{ document.querySelectorAll('video').forEach(v => v.pause()); }}
function clipVideos(section) {{ return Array.from(section.querySelectorAll('video')); }}
function applyPlaybackRates() {{
  document.querySelectorAll('video[data-playback-rate]').forEach(v => {{
    const rate = Number(v.dataset.playbackRate);
    if (Number.isFinite(rate) && rate > 0) {{
      v.defaultPlaybackRate = rate;
      v.playbackRate = rate;
    }}
  }});
}}
function playVideos(section) {{ clipVideos(section).forEach(v => v.play()); }}
function pauseVideos(section) {{ clipVideos(section).forEach(v => v.pause()); }}
function resetVideos(section) {{ clipVideos(section).forEach(v => {{ v.pause(); v.currentTime = 0; }}); }}
function applyFilters() {{
  const start = document.getElementById('startFilter').value;
  const point = document.getElementById('pointFilter').value;
  const axis = document.getElementById('axisFilter').value;
  const angle = document.getElementById('angleFilter').value;
  document.querySelectorAll('.run-card').forEach(card => {{
    const showStart = start === 'all' || card.dataset.start === start;
    const showPoint = point === 'all' || card.dataset.point === point;
    const showAxis = axis === 'all' || card.dataset.axis === axis;
    let anyEpisode = false;
    card.querySelectorAll('.episode').forEach(ep => {{
      const showEpisode = angle === 'all' || ep.dataset.angle === angle;
      ep.classList.toggle('hidden', !showEpisode);
      anyEpisode = anyEpisode || showEpisode;
    }});
    card.classList.toggle('hidden', !(showStart && showPoint && showAxis && anyEpisode));
  }});
}}
document.querySelectorAll('.clip').forEach(section => {{
  section.querySelector('.sync-play').addEventListener('click', () => playVideos(section));
  section.querySelector('.sync-pause').addEventListener('click', () => pauseVideos(section));
  section.querySelector('.sync-reset').addEventListener('click', () => resetVideos(section));
}});
['startFilter', 'pointFilter', 'axisFilter', 'angleFilter'].forEach(id => document.getElementById(id).addEventListener('change', applyFilters));
document.getElementById('expandAll').addEventListener('click', () => document.querySelectorAll('.run-card:not(.hidden) details:not(.hidden)').forEach(d => d.open = true));
document.getElementById('collapseAll').addEventListener('click', () => {{ pauseAll(); document.querySelectorAll('details').forEach(d => d.open = false); }});
document.getElementById('pauseAll').addEventListener('click', pauseAll);
applyPlaybackRates();
applyFilters();
</script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scene-info", type=Path, default=DEFAULT_SCENE_INFO)
    parser.add_argument("--html-name", default="rpy_grid_rpy_rotation_position_only_clean_uncropped_browser.html")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = load_records(args.output_dir, args.scene_info)
    html_text = build_html(records, args.output_dir)
    target = args.output_dir / args.html_name
    target.write_text(html_text, encoding="utf-8")
    print(target)
    print(f"records={len(records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
