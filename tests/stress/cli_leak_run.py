#!/usr/bin/env python3
"""Stress / leak test for the DMS engine through depthcli (no After Effects needed).

For every model x mapping mode x tile count, runs depthcli in ONE process over N frames (cache
disabled, so every frame runs inference) and samples the process RSS. Fails if a run crashes,
exceeds the timeout (default 10 min), or its RSS keeps growing after warm-up (leak).

  python3 -I tests/stress/cli_leak_run.py --cli build/linux-release/bin/depthcli \
      [--models-dir DIR_WITH_REAL_MODELS] [--frames 500] [--real-frames 60] [--timeout 600]

Models: the synthetic single-frame and temporal fixtures always; Depth Anything v2 Small too when
--models-dir contains it (fewer frames by default: CPU inference is ~0.6 s/frame).
Exit code 0 = all runs passed. A JSON report is written to <out>/stress_report.json.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FIXTURES = os.path.join(ROOT, "tests", "fixtures")


def rss_bytes(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def rss_bytes_any(proc):
    v = rss_bytes(proc.pid)
    if v is not None:
        return v
    try:  # Windows / macOS fallback
        import psutil  # type: ignore
        return psutil.Process(proc.pid).memory_info().rss
    except Exception:
        return None


def leak_verdict(samples, frames):
    """samples: list of (t, rss). Robust growth estimate after warm-up (first 25% of the run)."""
    if len(samples) < 20:
        return 0.0, 0.0, True, "too few samples (run too short to judge)"
    n = len(samples)
    tail = samples[n // 4:]
    t0 = tail[0][0]
    xs = [t - t0 for t, _ in tail]
    ys = [r for _, r in tail]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    var = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var if var > 0 else 0.0
    duration = xs[-1] if xs else 0.0
    projected = slope * duration  # growth over the measured window
    q = max(1, len(ys) // 10)
    warm = sorted(ys[:q])[q // 2]
    end = sorted(ys[-q:])[q // 2]
    growth = end - warm
    limit = max(48 * 2**20, 0.08 * warm)
    ok = growth <= limit and projected <= limit
    return growth, projected, ok, f"warm {warm/2**20:.1f} MB -> end {end/2**20:.1f} MB (growth {growth/2**20:+.1f} MB, trend {projected/2**20:+.1f} MB, limit {limit/2**20:.0f} MB)"


def run_one(cli, models_dir, model, mapping, tiles, inputs, frames, timeout, out_dir, log_dir):
    per_pass = len(inputs)
    repeat = max(1, math.ceil(frames / per_pass))
    cmd = [cli, "--models-dir", models_dir, "--log-dir", log_dir, "--log-level", "1", "--model", model, "--mapping", mapping,
           "--tiles", str(tiles), "--no-cache", "--repeat", str(repeat), "-q", "-o", out_dir, "--res", "518"]
    if mapping == "guided":
        cmd += ["--near", "10,10", "--far", "20,20", "--near-value", "0.9", "--far-value", "0.1"]
    for i in inputs:
        cmd += ["-i", i]
    t_start = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    samples = []
    timed_out = False
    while proc.poll() is None:
        r = rss_bytes_any(proc)
        if r is not None:
            samples.append((time.time() - t_start, r))
        if time.time() - t_start > timeout:
            timed_out = True
            proc.kill()
            break
        time.sleep(0.1)
    out, err = proc.communicate()
    elapsed = time.time() - t_start
    growth, projected, leak_ok, leak_msg = leak_verdict(samples, repeat * per_pass)
    crashed = proc.returncode is not None and proc.returncode < 0
    ok = (not timed_out) and proc.returncode == 0 and leak_ok
    return {
        "model": model, "mapping": mapping, "tiles": tiles, "frames": repeat * per_pass, "seconds": round(elapsed, 1),
        "returncode": proc.returncode, "crashed": crashed, "timed_out": timed_out, "rss_growth_mb": round(growth / 2**20, 2),
        "rss_trend_mb": round(projected / 2**20, 2), "peak_rss_mb": round(max((r for _, r in samples), default=0) / 2**20, 1),
        "leak_ok": leak_ok, "detail": leak_msg, "ok": ok, "stderr_tail": err.strip().splitlines()[-3:] if err else [],
        "summary": out.strip().splitlines()[-1] if out.strip() else "",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", required=True)
    ap.add_argument("--models-dir", default="")
    ap.add_argument("--frames", type=int, default=500)
    ap.add_argument("--real-frames", type=int, default=60)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "stress"))
    ap.add_argument("--quick", action="store_true", help="tiles 1 and 16 only, normalize + guided")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="dms_stress_")
    mdir = os.path.join(work, "models")
    os.makedirs(mdir)
    for f in ("synthetic_depth.onnx", "synthetic_temporal.onnx", "synthetic_normalizing.onnx"):
        shutil.copy(os.path.join(FIXTURES, "onnx", f), mdir)
    manifest = {"models": [
        {"id": "synthetic", "file": "synthetic_normalizing.onnx"},
        {"id": "synthetic-video", "kind": "temporal", "window": 8, "file": "synthetic_temporal.onnx"},
    ]}
    models = [("synthetic", args.frames), ("synthetic-video", args.frames)]
    real = os.path.join(args.models_dir, "depth_anything_v2_vits_dynamic.onnx") if args.models_dir else ""
    if real and os.path.isfile(real):
        os.symlink(real, os.path.join(mdir, "depth_anything_v2_vits_dynamic.onnx"))
        models.append(("da2-small", args.real_frames))
    with open(os.path.join(mdir, "models.json"), "w") as f:
        json.dump(manifest, f)

    seq = sorted(os.path.join(FIXTURES, "seq", n) for n in os.listdir(os.path.join(FIXTURES, "seq")))
    mappings = ["normalize", "guided"] if args.quick else ["raw", "normalize", "guided"]
    tiles_list = [1, 16] if args.quick else [1, 4, 16]
    results = []
    for model, frames in models:
        for mapping in mappings:
            for tiles in tiles_list:
                # Tiles need a frame large enough to split: use the 4K still for tiled runs of the fast models.
                inputs = seq if (tiles == 1 or model == "synthetic-video") else [os.path.join(FIXTURES, "odd_333x197.png"), os.path.join(FIXTURES, "scene_floor.png")]
                if model == "da2-small" and tiles > 1:
                    inputs = [os.path.join(FIXTURES, "scene_floor.png")]
                r = run_one(args.cli, mdir, model, mapping, tiles, inputs, frames, args.timeout, os.path.join(work, "out"), os.path.join(work, "logs"))
                results.append(r)
                status = "PASS" if r["ok"] else "FAIL"
                print(f"[{status}] {model:16s} {mapping:9s} tiles={tiles:<2d} frames={r['frames']:<4d} {r['seconds']:7.1f}s "
                      f"peak={r['peak_rss_mb']:7.1f}MB {r['detail']}", flush=True)
                if not r["ok"]:
                    print("        rc=%s timed_out=%s stderr=%s" % (r["returncode"], r["timed_out"], r["stderr_tail"]), flush=True)
    with open(os.path.join(args.out, "stress_report.json"), "w") as f:
        json.dump(results, f, indent=2)
    shutil.rmtree(work, ignore_errors=True)
    failed = [r for r in results if not r["ok"]]
    print(f"\n{len(results) - len(failed)}/{len(results)} runs passed; report: {os.path.join(args.out, 'stress_report.json')}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
