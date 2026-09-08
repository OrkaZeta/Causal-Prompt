"""Build frame strips and an annotated causal/current visual-evaluation page."""

import argparse
import html
import json
import re
import subprocess
from pathlib import Path

from tqdm import tqdm

from causal_prompt.prompt.schedule import schedule_record

PROJECT_DIR = Path(__file__).resolve().parents[2]
COLORS = ("e0", "e1", "e2", "e3", "e4")
SEEDED_STEM = re.compile(r"^(?P<sample_id>.+)_seed(?P<seed>\d+)$")


def _dataset_id(path: Path) -> str:
    if "single_obj" in path.stem:
        return "single_obj"
    if "activitynet_causal" in path.stem:
        return "activitynet_causal"
    return path.stem


def _experiment_metadata(exp_id: str) -> tuple[str, int | None]:
    if exp_id.startswith("singleobj-"):
        exp_data = "single_obj"
    elif exp_id.startswith("actnet-"):
        exp_data = "activitynet_causal"
    else:
        exp_data = exp_id.split("-", 1)[0]
    match = re.search(r"(?:^|-)seed(\d+)(?:-|$)", exp_id)
    return exp_data, int(match.group(1)) if match else None


def _videos_by_seed(video_root: Path) -> dict[int, dict[str, Path]]:
    videos: dict[int, dict[str, Path]] = {}
    for path in sorted(video_root.glob("*.mp4")):
        match = SEEDED_STEM.fullmatch(path.stem)
        # Treat legacy unsuffixed files as seed 0 until they are migrated.
        seed = int(match.group("seed")) if match else 0
        sample_id = match.group("sample_id") if match else path.stem
        videos.setdefault(seed, {})[sample_id] = path
    return videos


def discover_runs() -> list[dict]:
    """Discover every dataset/model/mode/seed/step as a distinct generation run."""
    runs = []
    baseline_root = PROJECT_DIR / "outputs" / "zeroshot_baseline"
    for video_root in sorted(baseline_root.glob("*/*/*/vid")):
        gen_data, model, gen_mode = video_root.parts[-4:-1]
        for gen_seed, videos in _videos_by_seed(video_root).items():
            runs.append({
                "model": model,
                "prompt": gen_mode,
                "exp": "zeroshot_baseline",
                "step": None,
                "exp_id": "zeroshot_baseline",
                "exp_mode": None,
                "exp_seed": None,
                "exp_data": None,
                "exp_step": None,
                "gen_mode": gen_mode,
                "gen_seed": gen_seed,
                "gen_data": gen_data,
                "videos": videos,
                "strips": video_root.parent / "strips",
                "suffix": f"_seed{gen_seed}",
            })

    cp_root = PROJECT_DIR / "outputs" / "exp1_cp_dmd"
    for video_root in sorted(cp_root.glob("*/*/video/step_*")):
        if not any(video_root.glob("*.mp4")):
            continue
        try:
            step = int(video_root.name.removeprefix("step_"))
        except ValueError:
            continue
        exp_id, exp_mode = video_root.parts[-4:-2]
        exp_data, exp_seed = _experiment_metadata(exp_id)
        for gen_seed, videos in _videos_by_seed(video_root).items():
            runs.append({
                "model": "CP-DMD",
                "prompt": exp_mode,
                "exp": exp_id,
                "step": step,
                "exp_id": exp_id,
                "exp_mode": exp_mode,
                "exp_seed": exp_seed,
                "exp_data": exp_data,
                "exp_step": step,
                "gen_mode": exp_mode,
                "gen_seed": gen_seed,
                "gen_data": exp_data,
                "videos": videos,
                "strips": video_root.parent.parent / "strips" / video_root.name,
                "suffix": f"_seed{gen_seed}",
            })
    return runs


def make_strip(video: Path, output: Path, overwrite: bool = False) -> None:
    """Extract ten evenly spaced frames from a five-second video into one JPG."""
    if output.is_file() and not overwrite:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y" if overwrite else "-n",
        "-i", str(video), "-vf", "fps=2,scale=240:-2,tile=10x1", "-frames:v", "1",
        str(output),
    ]
    subprocess.run(command, check=True)


def make_strip_tree(video_root: Path, strip_root: Path,
                    overwrite: bool = False) -> list[Path]:
    """Convert every MP4 below video_root, preserving relative subdirectories."""
    outputs = []
    videos = sorted(video_root.rglob("*.mp4"))
    for video in tqdm(videos, desc=f"Strips: {video_root.name}", unit="video"):
        output = (strip_root / video.relative_to(video_root)).with_suffix(".jpg")
        make_strip(video, output, overwrite)
        outputs.append(output)
    if not outputs:
        raise FileNotFoundError(f"No MP4 files found under {video_root}")
    return outputs


def _gantt_axis() -> str:
    return '<div class="axis">' + "".join(f"<i>{index}</i>" for index in range(21)) + "</div>"


def _display(value: object, *, step: bool = False) -> str:
    if value is None:
        return "—"
    if step:
        return str(int(value))
    return str(value)


def _display_data(value: object) -> object:
    return "activitynet" if value == "activitynet_causal" else value


def _sheet(title: object, fields: tuple[tuple[str, object], ...], css_class: str) -> str:
    rows = "".join(
        f'<tr><th>{html.escape(label)}</th><td>{html.escape(_display(value))}</td></tr>'
        for label, value in fields
    )
    return (f'<table class="meta-sheet {css_class}"><thead><tr>'
            f'<th colspan="2">{html.escape(_display(title))}</th></tr></thead>'
            f'<tbody>{rows}</tbody></table>')


def _experiment_sheet(run: dict) -> str:
    return _sheet(run["model"], (
        ("mode", run["exp_mode"]),
        ("seed", run["exp_seed"]),
        ("data", _display_data(run["exp_data"])),
        ("step", _display(run["exp_step"], step=True)),
    ), "experiment-sheet")


def _generation_sheet(run: dict) -> str:
    return _sheet(_display_data(run["gen_data"]), (
        ("mode", run["gen_mode"]),
        ("seed", run["gen_seed"]),
    ), "generation-sheet")


def _experiment_key(run: dict) -> tuple[object, ...]:
    """Identify one trained model/experiment configuration."""
    return tuple(run[key] for key in (
        "model", "exp_id", "exp_mode", "exp_seed", "exp_data", "exp_step",
    ))


def _generation_key(run: dict) -> tuple[object, ...]:
    """Identify generations whose scheduling graph is shared across seeds."""
    return run["gen_mode"], run["gen_data"]


def _group_runs(runs: list[dict], key) -> list[list[dict]]:
    groups: dict[tuple[object, ...], list[dict]] = {}
    for run in runs:
        groups.setdefault(key(run), []).append(run)
    return [sorted(group, key=lambda run: run["gen_seed"]) for group in groups.values()]


def _active_runs(active: list[bool]) -> list[tuple[int, int]]:
    runs, start = [], None
    for index, value in enumerate((*active, False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    return runs


def _visibility_gantt(record: dict, mode: str, model: str, step: str = "—") -> str:
    item = schedule_record(record, mode=mode)
    rows = []
    for index, (event_id, description) in enumerate(zip(record["events_idx"], record["events_decs"])):
        active = [description in prompt for prompt in item.chunk_prompts]
        bars = "".join(
            f'<i class="{COLORS[index % len(COLORS)]}" '
            f'style="left:{start / 21 * 100:.4f}%;width:{(end - start) / 21 * 100:.4f}%">'
            f'E{event_id} · {html.escape(description)}</i>'
            for start, end in _active_runs(active)
        )
        rows.append(f'<div class="track">{bars}</div>')
    return f'<div class="schedule">{_gantt_axis()}{"".join(rows)}</div>'


def _event_gantt(record: dict) -> str:
    events = []
    for index, (event_id, times, description) in enumerate(zip(
            record["events_idx"], record["events_timestamps"], record["events_decs"]
    )):
        left, width = times[0] / 5 * 100, (times[1] - times[0]) / 5 * 100
        events.append(f'<div class="track"><i class="{COLORS[index % len(COLORS)]}" '
                      f'style="left:{left:.2f}%;width:{width:.2f}%">'
                      f'E{event_id} · {html.escape(description)}</i></div>')
    return (f'<div class="result raw"><div class="meta"><span><b>ground truth</b></span>'
            f'<span>dataset events</span></div>'
            f'<div class="visuals"><div class="schedule">{_gantt_axis()}'
            f'{"".join(events)}</div></div></div>')


def _select_runs(models: list[str] | None, prompts: list[str] | None,
                 experiments: list[str] | None, steps: list[int] | None) -> list[dict]:
    runs = [run for run in discover_runs()
            if (not models or run["model"] in models)
            and (not prompts or run["prompt"] in prompts)
            and (not experiments or run["exp"] in experiments)
            and (not steps or run["step"] in steps)]
    if not runs:
        raise ValueError("No discovered run matches the model/prompt/exp/step filters.")
    return runs


def build_report(dataset: Path, runs: list[dict], report: Path,
                 max_samples: int | None = None, overwrite: bool = False,
                 title: str = "ActNet Eval") -> Path:
    records = {}
    with dataset.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] == "test":
                records[record["sample_id"]] = record

    gen_data = _dataset_id(dataset)
    runs = [run for run in runs if run["gen_data"] == gen_data]
    runs = [run for run in runs if any(sample_id in records for sample_id in run["videos"])]
    sample_ids = [sample_id for sample_id in records
                  if any(sample_id in run["videos"] for run in runs)]
    if max_samples is not None:
        sample_ids = sample_ids[:max_samples]
    if not sample_ids:
        raise FileNotFoundError("No discovered videos match the test JSONL.")

    cards = []
    for sample_id in tqdm(sample_ids, desc="Report", unit="sample"):
        record = records[sample_id]
        results = []
        for experiment in _group_runs(runs, _experiment_key):
            available = [run for run in experiment if sample_id in run["videos"]]
            if not available:
                continue
            generation_groups = []
            show_experiment = True
            for variants in _group_runs(available, _generation_key):
                generation_rows = []
                for run in variants:
                    video = run["videos"][sample_id]
                    strip = run["strips"] / f"{sample_id}{run['suffix']}.jpg"
                    make_strip(video, strip, overwrite)
                    strip_relative = Path("..") / strip.relative_to(PROJECT_DIR)
                    video_relative = Path("..") / video.relative_to(PROJECT_DIR)
                    experiment_sheet = _experiment_sheet(run) if show_experiment else ""
                    show_experiment = False
                    generation_rows.append(
                        f'<div class="generation"><div class="meta-stack">'
                        f'{experiment_sheet}{_generation_sheet(run)}</div>'
                        f'<button class="strip" data-video="{video_relative}">'
                        f'<img src="{strip_relative}" alt="{html.escape(sample_id)} '
                        f'seed {run["gen_seed"]} strip"></button></div>'
                    )
                run = variants[0]
                generation_groups.append(
                    f'<div class="generation-config">{"".join(generation_rows)}'
                    f'<div class="shared-gantt">'
                    f'{_visibility_gantt(record, run["prompt"], run["model"])}'
                    f'</div></div>'
                )
            run = available[0]
            results.append(
                f'<div class="result" data-model="{html.escape(run["model"], quote=True)}">'
                f'<div class="run-content">{"".join(generation_groups)}</div></div><hr>'
            )
        cards.append(f'''<section class="sample"><code>{html.escape(sample_id)}</code>
<div class="initial"><b>Initial view</b><span>{html.escape(record["init_decs"])}</span></div>
{_event_gantt(record)}<hr>
{"".join(results)}
</section>''')

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<link rel="stylesheet" href="assets/style.css"><style>
:root{{--bg:#f3f6f9;--panel:#fff;--panel2:#f8fafc;--line:#cbd5df;--text:#17212b;--muted:#607080;--cyan:#087f8c;--orange:#f0a12b;--green:#42a66b;--red:#dc6074;--blue:#4f83d1;color-scheme:light}}body{{background:#f3f6f9;color:var(--text)}}.nav a:hover,.nav a.active{{background:#e7edf3;color:var(--text)}}
.filters{{position:sticky;top:0;z-index:3;display:flex;align-items:center;gap:8px;margin:0 8px;padding:7px 9px;background:#f3f6f9ee;border-bottom:1px solid var(--line);font:12px sans-serif;backdrop-filter:blur(6px)}}.filters select{{min-width:180px;padding:4px 7px;border:1px solid var(--line);border-radius:4px;background:#fff;color:var(--text)}}.wrap{{max-width:none;padding:0 8px 8px}}.sample{{margin:4px 0;padding:4px;background:#fff;border:1px solid var(--line)}}.sample>code{{display:block;color:var(--muted);line-height:18px}}.initial{{display:grid;grid-template-columns:88px 1fr;gap:4px;padding:3px 5px;border-left:3px solid var(--cyan);background:#47d7d10b;font-size:12px}}hr{{margin:3px 0;border:0;border-top:1px solid #aebdca}}.result{{margin:2px 0}}.run-content,.visuals{{min-width:0}}.generation-config+.generation-config{{margin-top:6px;padding-top:4px;border-top:1px solid var(--line)}}.generation{{display:grid;grid-template-columns:calc(84px + 5ch) minmax(0,1fr);gap:4px;align-items:stretch;margin-bottom:3px}}.meta-stack{{display:flex;min-width:0;overflow:hidden;flex-direction:column;justify-content:flex-end;align-self:stretch}}.meta-sheet{{width:100%;max-width:100%;overflow:hidden;table-layout:fixed;background:#fff;font:10px/1.25 monospace;color:var(--text)}}.meta-sheet th,.meta-sheet td{{max-width:0;padding:1px 3px;border:0;overflow:hidden;overflow-wrap:anywhere;text-align:left;vertical-align:top}}.meta-sheet thead th{{font-weight:700;text-align:center}}.meta-sheet tbody th{{width:34%;font-weight:400}}.generation-sheet{{margin-top:3px}}.strip{{display:block;width:100%;padding:0;margin:0;border:0;background:none;cursor:pointer}}.strip img{{display:block;width:100%}}.shared-gantt{{margin-left:calc(84px + 5ch + 4px)}}.raw{{display:grid;grid-template-columns:calc(84px + 5ch) minmax(0,1fr);gap:4px}}.raw .meta{{display:flex;flex-direction:column;align-self:start;font:10px/1.25 monospace;color:var(--muted);overflow:hidden}}.raw .meta span{{overflow-wrap:anywhere;white-space:normal}}.schedule{{display:grid;grid-template-columns:minmax(0,1fr);gap:1px;margin:2px 0}}.axis{{display:grid;grid-template-columns:repeat(21,1fr);font:8px monospace;color:var(--muted);text-align:center}}.axis i{{font-style:normal;border-left:1px solid #17212b18}}.track{{position:relative;height:15px;background:repeating-linear-gradient(90deg,#17212b18 0,#17212b18 1px,transparent 1px,transparent 4.7619%)}}.track i{{position:absolute;height:100%;padding:1px 3px;font:8px/13px sans-serif;color:#07111f;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-style:normal}}.e0{{background:var(--orange)}}.e1{{background:var(--red)}}.e2{{background:var(--blue)}}.e3{{background:var(--green)}}.e4{{background:var(--cyan)}}dialog{{width:min(1100px,96vw);padding:0;border:1px solid var(--line);background:#fff}}dialog::backdrop{{background:#000c}}dialog video{{display:block;width:100%;max-height:90vh}}dialog button{{position:absolute;right:4px;top:4px;z-index:2;border:0;background:#000b;color:white;font-size:20px;cursor:pointer}}
</style></head><body><nav class="nav"></nav><div class="filters"><label for="model-filter">Model</label><select id="model-filter"><option value="">All models</option></select></div><main class="wrap">{''.join(cards)}</main><dialog id="player"><button aria-label="close">×</button><video controls autoplay></video></dialog><script src="assets/app.js"></script><script>const d=document.querySelector('#player'),v=d.querySelector('video');document.querySelectorAll('button.strip').forEach(x=>x.onclick=()=>{{v.src=x.dataset.video;d.showModal();}});function closePlayer(){{v.pause();v.removeAttribute('src');v.load();d.close();}}d.querySelector('button').onclick=closePlayer;d.onclick=e=>{{if(e.target===d)closePlayer();}};const mf=document.querySelector('#model-filter'),rows=[...document.querySelectorAll('.result[data-model]')];[...new Set(rows.map(x=>x.dataset.model))].sort().forEach(model=>mf.add(new Option(model,model)));mf.onchange=()=>rows.forEach(row=>{{const show=!mf.value||row.dataset.model===mf.value;row.hidden=!show;if(row.nextElementSibling?.tagName==='HR')row.nextElementSibling.hidden=!show;}});</script></body></html>''',
                      encoding="utf-8")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build causal/current strips and visual report.")
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Build one report from this dataset; omit to build both evaluation reports.",
    )
    parser.add_argument("--model", nargs="+")
    parser.add_argument("--prompt", nargs="+", choices=("causal", "current"))
    parser.add_argument("--exp", nargs="+")
    parser.add_argument("--step", type=int, nargs="+")
    parser.add_argument("--report", type=Path,
                        help="Output for --dataset; omit to build both default reports.")
    parser.add_argument("--max-samples", type=int, default=10,
                        help="Number of dataset-ordered samples to show (default: 10).")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--video-root", type=Path,
                        help="Build a strip tree from this video directory instead of an HTML report.")
    parser.add_argument("--strip-root", type=Path,
                        help="Destination paired with --video-root; relative subdirectories are preserved.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.video_root:
        if args.strip_root is None:
            raise ValueError("--strip-root is required with --video-root.")
        outputs = make_strip_tree(args.video_root, args.strip_root, args.overwrite)
        print(f"Created or retained {len(outputs)} strips under {args.strip_root}")
        return
    runs = _select_runs(args.model, args.prompt, args.exp, args.step)
    if args.dataset is not None:
        report = args.report or PROJECT_DIR / "reports" / "actnet-eval.html"
        print(build_report(args.dataset, runs, report, args.max_samples, args.overwrite))
        return
    if args.report is not None:
        raise ValueError("--report requires --dataset.")
    defaults = (
        (PROJECT_DIR / "data" / "activitynet_causal_5s_test.jsonl",
         PROJECT_DIR / "reports" / "actnet-eval.html", "ActNet Eval"),
        (PROJECT_DIR / "data" / "eval_single_obj_test.jsonl",
         PROJECT_DIR / "reports" / "single-obj-eval.html", "Single Object Eval"),
    )
    for dataset, report, title in defaults:
        print(build_report(dataset, runs, report, args.max_samples, args.overwrite, title))


if __name__ == "__main__":
    main()
