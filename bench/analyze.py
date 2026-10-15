"""Analysis of a benchmark series: tables (CSV + Markdown) and figures (PNG + SVG).

    python bench/analyze.py bench/results/<date>_<host>

Every number comes from the raw files of the series; nothing is typed in.

* time per (project, configuration, m): mean, standard deviation, 95 %
  confidence interval (Student's t);
* speedup ``S(m) = T_serial / T(m)``, efficiency ``E = S / m``, Karp-Flatt
  metric ``e = (1/S - 1/m) / (1 - 1/m)`` and the Amdahl curve
  ``S(m) = 1 / (e + (1 - e)/m)`` with the mean ``e`` of a configuration;
* failures before and after re-runs; false failures = tests that failed in a
  parallel configuration but never failed in the serial runs of the series;
* runs killed after ``--run-timeout`` are excluded from the times and
  counted separately (``timeouts`` in the failure table);
* the method's one-time overhead (profiling + detection) and the break-even
  point: CI runs needed until the time saved per run against the best plain
  xdist configuration with the same m covers it;
* prediction error of the plan (smartpar runs);
* for projects with a ground truth (control project): precision/recall of
  detection, with the recall of file conflicts separately for "profile" and
  "profile + detect".
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import mean, stdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ORDER = ("serial", "xdist-load", "xdist-loadfile", "smartpar")
BASELINES = ("xdist-load", "xdist-loadfile")
LABELS = {
    "serial": "послідовно",
    "xdist-load": "xdist --dist load",
    "xdist-loadfile": "xdist --dist loadfile",
    "smartpar": "smartpar",
}
HATCHES = {"serial": "", "xdist-load": "//", "xdist-loadfile": "\\\\", "smartpar": "xx"}
MARKERS = {"serial": "o", "xdist-load": "s", "xdist-loadfile": "^", "smartpar": "D"}
LINESTYLES = {"serial": "-", "xdist-load": "--", "xdist-loadfile": "-.", "smartpar": "-"}
RUN_FILE = re.compile(r"^(?P<config>[a-z-]+)_m(?P<m>\d+)_r(?P<r>\d+)\.json$")

# Two-sided 95 % Student t quantiles, df = 1..30.
T95 = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
]  # fmt: skip


# --- statistics ----------------------------------------------------------------------


def t95(df: int) -> float:
    return T95[df - 1] if 1 <= df <= len(T95) else 1.96


def summarize(values: Sequence[float]) -> dict[str, float]:
    n = len(values)
    m = mean(values)
    sd = stdev(values) if n > 1 else 0.0
    ci = t95(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
    return {"n": n, "mean": m, "std": sd, "ci95": ci}


def karp_flatt(speedup: float, m: int) -> float | None:
    if m <= 1 or speedup <= 0:
        return None
    return (1 / speedup - 1 / m) / (1 - 1 / m)


def amdahl(e: float, m: float) -> float:
    return 1 / (e + (1 - e) / m)


def break_even_runs(overhead_s: float, baseline_s: float, smartpar_s: float) -> int | None:
    """CI runs until the saved time covers the one-time overhead; None if no saving."""
    gain = baseline_s - smartpar_s
    if gain <= 0:
        return None
    return max(1, math.ceil(overhead_s / gain))


def precision_recall(found: set, expected: set) -> tuple[float, float]:
    tp = len(found & expected)
    precision = tp / len(found) if found else 1.0
    recall = tp / len(expected) if expected else 1.0
    return precision, recall


# --- loading ---------------------------------------------------------------------------


def load_series(series: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict]]:
    env = json.loads((series / "env.json").read_text())
    records, prep = [], {}
    for project_dir in sorted((series / "raw").iterdir()):
        if not project_dir.is_dir():
            continue
        if (project_dir / "prep.json").exists():
            prep[project_dir.name] = json.loads((project_dir / "prep.json").read_text())
        for path in sorted(project_dir.iterdir()):
            if RUN_FILE.match(path.name):
                records.append(json.loads(path.read_text()))
    return env, records, prep


def group_times(records: Iterable[dict]) -> dict[tuple[str, str, int], list[float]]:
    """Wall times per (project, config, m); runs killed after the timeout are not
    measurements and are left out (counted in the failure table)."""
    out: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for r in records:
        if not r.get("timed_out"):
            out[(r["project"], r["config"], r["workers"])].append(r["wall_s"])
    return out


# --- tables ------------------------------------------------------------------------------


def time_table(records: list[dict]) -> list[dict[str, Any]]:
    times = group_times(records)
    rows = []
    serial = {p: mean(v) for (p, c, _m), v in times.items() if c == "serial"}
    for (project, config, m), values in sorted(
        times.items(), key=lambda kv: (kv[0][0], CONFIG_ORDER.index(kv[0][1]), kv[0][2])
    ):
        s = summarize(values)
        speedup = serial[project] / s["mean"] if project in serial else None
        rows.append(
            {
                "project": project,
                "config": config,
                "workers": m,
                "runs": s["n"],
                "mean_s": s["mean"],
                "std_s": s["std"],
                "ci95_s": s["ci95"],
                "speedup": speedup,
                "efficiency": speedup / m if speedup is not None else None,
                "karp_flatt": karp_flatt(speedup, m) if speedup is not None else None,
            }
        )
    return rows


def failure_table(
    records: list[dict], nod: dict[str, set[str]] | None = None
) -> list[dict[str, Any]]:
    """``nod``: per project, tests the detector classified as NOD; they are
    false failures by definition only if they happened not to fail serially,
    so a second column excludes them."""
    nod = nod or {}
    serial_failed: dict[str, set[str]] = defaultdict(set)
    for r in records:
        if r["config"] == "serial":
            serial_failed[r["project"]] |= set(r["failed_after_rerun"])
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for r in records:
        grouped[(r["project"], r["config"], r["workers"])].append(r)
    rows = []
    for (project, config, m), runs in sorted(
        grouped.items(), key=lambda kv: (kv[0][0], CONFIG_ORDER.index(kv[0][1]), kv[0][2])
    ):
        base = serial_failed[project]
        false_before = [set(r["failed_before_rerun"]) - base for r in runs]
        false_after = [set(r["failed_after_rerun"]) - base for r in runs]
        parallel = config != "serial"
        rows.append(
            {
                "project": project,
                "config": config,
                "workers": m,
                "runs": len(runs),
                "timeouts": sum(1 for r in runs if r.get("timed_out")),
                "failures_before_rerun_mean": mean(len(r["failed_before_rerun"]) for r in runs),
                "failures_after_rerun_mean": mean(len(r["failed_after_rerun"]) for r in runs),
                "false_failures_before_rerun": sum(map(len, false_before)) if parallel else "",
                "false_failures_after_rerun": sum(map(len, false_after)) if parallel else "",
                "false_failures_after_rerun_excl_nod": (
                    sum(len(f - nod.get(project, set())) for f in false_after) if parallel else ""
                ),
                "false_failure_tests": ";".join(sorted(set().union(*false_after)))
                if parallel
                else "",
            }
        )
    return rows


def detected_nod(series: Path) -> dict[str, set[str]]:
    out = {}
    for flaky in (series / "raw").glob("*/smartpar_master/flaky.json"):
        out[flaky.parents[1].name] = {n["test"] for n in json.loads(flaky.read_text())["nod"]}
    return out


def overhead_table(records: list[dict], prep: dict[str, dict]) -> list[dict[str, Any]]:
    times = {k: mean(v) for k, v in group_times(records).items()}
    rows = []
    for project, p in sorted(prep.items()):
        ms = sorted({m for (pr, c, m) in times if pr == project and c == "smartpar"})
        for m in ms:
            baselines = {c: times[(project, c, m)] for c in BASELINES if (project, c, m) in times}
            if not baselines:
                continue
            best = min(baselines, key=baselines.get)
            smart = times[(project, "smartpar", m)]
            runs = break_even_runs(p["overhead_s"], baselines[best], smart)
            rows.append(
                {
                    "project": project,
                    "workers": m,
                    "profile_s": p["profile_s"],
                    "detect_s": p["detect_s"],
                    "overhead_s": p["overhead_s"],
                    "warmup_s": p["warmup_s"],
                    "best_baseline": best,
                    "baseline_mean_s": baselines[best],
                    "smartpar_mean_s": smart,
                    "gain_per_run_s": baselines[best] - smart,
                    "break_even_runs": runs if runs is not None else "never",
                }
            )
    return rows


def prediction_table(records: list[dict]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for r in records:
        predicted = r.get("predicted_wall_s")
        if r["config"] == "smartpar" and predicted:
            grouped[(r["project"], r["workers"])].append(
                (r["wall_s"] - predicted) / predicted * 100
            )
    return [
        {
            "project": project,
            "workers": m,
            "runs": len(errors),
            "mean_error_pct": mean(errors),
            "mean_abs_error_pct": mean(abs(e) for e in errors),
        }
        for (project, m), errors in sorted(grouped.items())
    ]


def detection_table(series: Path, env: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for project, info in sorted(env["projects"].items()):
        gt_path = info.get("ground_truth")
        raw = series / "raw" / project
        if not gt_path or not (raw / "smartpar_master" / "flaky.json").exists():
            continue
        gt = json.loads((REPO_ROOT / gt_path).read_text())
        final = json.loads((raw / "smartpar_master" / "dependencies.json").read_text())
        after_profile = json.loads((raw / "deps_after_profile.json").read_text())
        flaky = json.loads((raw / "smartpar_master" / "flaky.json").read_text())

        def conflicts(deps: dict) -> set:
            return {(c["a"], c["b"], c["resource"], c["kind"]) for c in deps["conflicts"]}

        expected_conflicts = {
            (c["a"], c["b"], c["resource"], c["kind"]) for c in gt["file_conflicts"]
        }
        nod = {n["test"] for n in gt["nod"]}
        metrics = {
            "file conflicts (profile)": (conflicts(after_profile), expected_conflicts),
            "file conflicts (profile + detect)": (conflicts(final), expected_conflicts),
            "polluter -> victim": (
                {(p, v["victim"]) for v in flaky["od_victims"] for p in v["polluters"]},
                {(p["polluter"], p["victim"]) for p in gt["od_pairs"]},
            ),
            "state setter -> brittle": (
                {(s, b["test"]) for b in flaky["od_brittle"] for s in b["state_setters"]},
                {(b["setter"], b["test"]) for b in gt["brittle"]},
            ),
            "NOD": ({n["test"] for n in flaky["nod"]}, nod),
        }
        for name, (found, expected) in metrics.items():
            precision, recall = precision_recall(found, expected)
            rows.append(
                {
                    "project": project,
                    "metric": name,
                    "found": len(found),
                    "expected": len(expected),
                    "true_positives": len(found & expected),
                    "precision": precision,
                    "recall": recall,
                }
            )
    return rows


def write_table(rows: list[dict[str, Any]], out: Path, title: str) -> None:
    if not rows:
        return
    columns = list(rows[0])
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return "–" if math.isnan(v) else f"{v:.3f}"
        return "–" if v is None or v == "" else str(v)

    lines = [f"### {title}", "", "| " + " | ".join(columns) + " |"]
    lines.append("|" + "|".join("---" for _ in columns) + "|")
    lines += ["| " + " | ".join(fmt(r[c]) for c in columns) + " |" for r in rows]
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# --- figures -----------------------------------------------------------------------------


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
            "font.size": 9,
            "axes.grid": True,
            "grid.color": "0.85",
            "savefig.bbox": "tight",
        }
    )
    return plt


def save(fig: Any, out: Path) -> None:
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".svg"))


def fig_times(rows: list[dict], m: int, out: Path) -> None:
    plt = _plt()
    projects = sorted({r["project"] for r in rows})
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    width = 0.2
    for i, config in enumerate(CONFIG_ORDER):
        xs, ys, err = [], [], []
        for j, project in enumerate(projects):
            row = next(
                (
                    r
                    for r in rows
                    if r["project"] == project
                    and r["config"] == config
                    and r["workers"] == (1 if config == "serial" else m)
                ),
                None,
            )
            if row:
                xs.append(j + (i - 1.5) * width)
                ys.append(row["mean_s"])
                err.append(0 if math.isnan(row["ci95_s"]) else row["ci95_s"])
        ax.bar(
            xs,
            ys,
            width,
            yerr=err,
            capsize=2,
            label=LABELS[config],
            color="white",
            edgecolor="black",
            hatch=HATCHES[config],
        )
    ax.set_xticks(range(len(projects)), projects)
    ax.set_ylabel("Час виконання, с")
    ax.set_title(f"Час виконання тестів, m = {m} (послідовно – m = 1)")
    ax.legend(fontsize=7)
    save(fig, out)
    plt.close(fig)


def fig_gantt(series: Path, project: str, m: int, out: Path) -> bool:
    plt = _plt()
    raw = series / "raw" / project
    panels = [(c, raw / f"{c}_m{m}_r0.timing.jsonl") for c in ("xdist-load", "smartpar")]
    if not all(p.exists() for _c, p in panels):
        return False
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.2), sharex=True)
    for ax, (config, path) in zip(axes, panels, strict=True):
        tests = [json.loads(x) for x in path.read_text().splitlines()]
        t0 = min(t["start"] for t in tests)
        workers = sorted({t["worker"] for t in tests}, key=lambda w: int(re.sub(r"\D", "", w) or 0))
        for y, worker in enumerate(workers):
            bars = [
                (t["start"] - t0, max(t["end"] - t["start"], 1e-3))
                for t in tests
                if t["worker"] == worker
            ]
            ax.broken_barh(
                bars, (y - 0.35, 0.7), facecolors="0.6", edgecolors="black", linewidth=0.3
            )
        ax.set_yticks(range(len(workers)), workers)
        ax.set_ylabel("Воркер")
        ax.set_title(LABELS[config])
    axes[-1].set_xlabel("Час від початку, с")
    fig.suptitle(f"Завантаження воркерів, {project}, m = {m}")
    save(fig, out)
    plt.close(fig)
    return True


def fig_speedup(rows: list[dict], project: str, out: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ms_all = sorted({r["workers"] for r in rows if r["project"] == project})
    for config in CONFIG_ORDER[1:]:
        pts = sorted(
            (r["workers"], r["speedup"], r["karp_flatt"])
            for r in rows
            if r["project"] == project and r["config"] == config and r["speedup"]
        )
        if not pts:
            continue
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            marker=MARKERS[config],
            linestyle="none",
            color="black",
            label=LABELS[config],
        )
        es = [p[2] for p in pts if p[2] is not None]
        if es:
            e = mean(es)
            grid = [1 + i * (max(ms_all) - 1) / 50 for i in range(51)]
            ax.plot(
                grid,
                [amdahl(e, g) for g in grid],
                linestyle=LINESTYLES[config],
                color="black",
                linewidth=0.8,
                label=f"Амдал, e = {e:.3f}",
            )
    ax.plot(ms_all, ms_all, color="0.6", linewidth=0.8, linestyle=":", label="лінійне")
    ax.set_xticks(ms_all)
    ax.set_xlabel("Кількість воркерів m")
    ax.set_ylabel("Прискорення S(m)")
    ax.set_title(f"Прискорення відносно послідовного запуску, {project}")
    ax.legend(fontsize=7)
    save(fig, out)
    plt.close(fig)


def fig_table(
    rows: list[dict], columns: list[str], headers: list[str], title: str, out: Path
) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 0.4 + 0.25 * (len(rows) + 1)))
    ax.axis("off")
    cells = [
        [
            ("–" if r[c] == "" else (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c])))
            for c in columns
        ]
        for r in rows
    ]
    table = ax.table(cellText=cells, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    ax.set_title(title)
    save(fig, out)
    plt.close(fig)


def fig_detection(rows: list[dict], out: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    names = [f"{r['project']}: {r['metric']}" for r in rows]
    ys = range(len(rows))
    ax.barh(
        [y + 0.2 for y in ys],
        [r["precision"] for r in rows],
        0.4,
        color="white",
        edgecolor="black",
        hatch="//",
        label="точність (precision)",
    )
    ax.barh(
        [y - 0.2 for y in ys],
        [r["recall"] for r in rows],
        0.4,
        color="0.6",
        edgecolor="black",
        label="повнота (recall)",
    )
    ax.set_yticks(list(ys), names, fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Значення")
    ax.set_title("Виявлення залежностей на контрольному проєкті")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, out)
    plt.close(fig)


# --- main ----------------------------------------------------------------------------------


def analyze(series: Path) -> Path:
    env, records, prep = load_series(series)
    out = series / "analysis"
    out.mkdir(exist_ok=True)
    times = time_table(records)
    failures = failure_table(records, detected_nod(series))
    overhead = overhead_table(records, prep)
    prediction = prediction_table(records)
    detection = detection_table(series, env)
    write_table(times, out / "times", "Час виконання, прискорення, Карп-Флатт")
    write_table(failures, out / "failures", "Падіння до/після перезапусків і хибні падіння")
    write_table(overhead, out / "overhead", "Накладні витрати методу і точка окупності")
    write_table(prediction, out / "prediction", "Похибка прогнозу плану")
    write_table(detection, out / "detection", "Precision/recall виявлення (ground truth)")

    measured = sorted({r["workers"] for r in records if r["config"] != "serial"})
    if measured:
        m = 4 if 4 in measured else max(measured)
        fig_times(times, m, out / "fig1_times")
        for project in sorted({r["project"] for r in records}):
            fig_gantt(series, project, m, out / f"fig2_gantt_{project}")
            fig_speedup(times, project, out / f"fig3_speedup_{project}")
    parallel = [r for r in failures if r["config"] != "serial"]
    if parallel:
        fig_table(
            parallel,
            [
                "project",
                "config",
                "workers",
                "failures_before_rerun_mean",
                "failures_after_rerun_mean",
                "false_failures_after_rerun",
                "false_failures_after_rerun_excl_nod",
            ],
            [
                "проєкт",
                "конфігурація",
                "m",
                "падінь до",
                "падінь після",
                "хибних",
                "хибних без NOD",
            ],
            "Хибні падіння (впали паралельно, не падали послідовно)",
            out / "fig4_false_failures",
        )
    if detection:
        fig_detection(detection, out / "fig5_detection")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("series", type=Path, help="bench/results/<date>_<host>")
    args = parser.parse_args(argv)
    out = analyze(args.series)
    print(f"analysis written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
