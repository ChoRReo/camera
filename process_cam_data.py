#!/usr/bin/env python3
"""Average polarization image bursts and estimate useful capture counts."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_DIRECTIONS = ("0", "45", "90", "135", "L", "R")
DEFAULT_MILESTONES = (1, 2, 4, 8, 16, 32, 64, 100, 128, 200, 256, 400, 512, 800)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each scene/direction folder, average all PNG frames and save one "
            "mean image. Also estimate random noise reduction versus frame count."
        )
    )
    parser.add_argument(
        "data_root",
        nargs="?",
        default="/Volumes/xuyifan_u/cam_data",
        help="Dataset root. Default: /Volumes/xuyifan_u/cam_data",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Where averaged images and summary JSON are written. Default: current directory",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        help="Optional scene names to process, for example: --scenes 00001 00002",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=list(DEFAULT_DIRECTIONS),
        help="Direction folder names. Default: 0 45 90 135 L R",
    )
    parser.add_argument(
        "--pattern",
        default="*.png",
        help="Image filename glob used recursively inside each direction. Default: *.png",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=16,
        help=(
            "Pixel stride for noise estimation. 16 samples a 2048x2048 image as a "
            "128x128 grid. Larger is faster, smaller is more accurate. Default: 16"
        ),
    )
    parser.add_argument(
        "--noise-thresholds",
        type=float,
        nargs="+",
        default=[1.0, 0.5],
        help="Average-noise thresholds in DN used for recommended frame counts. Default: 1.0 0.5",
    )
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="+",
        default=list(DEFAULT_MILESTONES),
        help="Frame counts compared against the full-set mean on sampled pixels.",
    )
    parser.add_argument(
        "--summary-name",
        default="cam_data_average_summary.json",
        help="Summary JSON filename. Default: cam_data_average_summary.json",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing averaged PNG files.",
    )
    return parser.parse_args()


def list_scenes(root: Path, selected: list[str] | None) -> list[Path]:
    if selected:
        scenes = [root / name for name in selected]
    else:
        scenes = sorted(p for p in root.iterdir() if p.is_dir())
    missing = [str(p) for p in scenes if not p.is_dir()]
    if missing:
        raise FileNotFoundError(f"Scene folder(s) not found: {', '.join(missing)}")
    return scenes


def open_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.uint8)


def sample_frame(frame: np.ndarray, stride: int) -> np.ndarray:
    if frame.ndim == 2:
        return frame[::stride, ::stride].astype(np.float64)
    return frame[::stride, ::stride, ...].astype(np.float64)


def output_name(scene: str, direction: str) -> str:
    safe_direction = direction.replace("/", "_")
    return f"{scene}_{safe_direction}_avg.png"


def process_group(
    scene_name: str,
    direction: str,
    files: list[Path],
    output_dir: Path,
    sample_stride: int,
    milestones: list[int],
    thresholds: list[float],
    overwrite: bool,
) -> dict:
    out_name = output_name(scene_name, direction)
    out_path = output_dir / out_name
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"{out_path} exists. Pass --overwrite to replace it.")

    nfiles = len(files)
    accumulator = None
    sample_mean = None
    sample_m2 = None
    sample_sum = None
    milestone_means = {}
    effective_milestones = sorted(m for m in set(milestones) if 1 <= m <= nfiles)
    started = time.time()

    for index, image_path in enumerate(files, start=1):
        frame = open_image(image_path)
        if accumulator is None:
            accumulator = np.zeros(frame.shape, dtype=np.uint64)
            sample = sample_frame(frame, sample_stride)
            sample_mean = np.zeros(sample.shape, dtype=np.float64)
            sample_m2 = np.zeros(sample.shape, dtype=np.float64)
            sample_sum = np.zeros(sample.shape, dtype=np.float64)
            image_shape = frame.shape
        elif frame.shape != image_shape:
            raise ValueError(
                f"Shape mismatch in {scene_name}/{direction}: "
                f"{image_path} has {frame.shape}, expected {image_shape}"
            )
        else:
            sample = sample_frame(frame, sample_stride)

        accumulator += frame.astype(np.uint64, copy=False)

        delta = sample - sample_mean
        sample_mean += delta / index
        sample_m2 += delta * (sample - sample_mean)
        sample_sum += sample
        if index in effective_milestones:
            milestone_means[str(index)] = (sample_sum / index).copy()

        if index % 200 == 0 or index == nfiles:
            print(f"{scene_name}/{direction}: {index}/{nfiles}", flush=True)

    averaged = np.rint(accumulator / nfiles).clip(0, 255).astype(np.uint8)
    Image.fromarray(averaged).save(out_path)

    sample_variance = sample_m2 / max(nfiles - 1, 1)
    single_frame_rms_noise = float(np.sqrt(np.mean(sample_variance)))
    final_sample_mean = sample_mean
    convergence = {}
    for milestone, mean_at_milestone in milestone_means.items():
        error = mean_at_milestone - final_sample_mean
        convergence[milestone] = float(np.sqrt(np.mean(error * error)))

    recommended = {}
    for threshold in thresholds:
        if threshold <= 0:
            continue
        recommended[str(threshold)] = int(math.ceil((single_frame_rms_noise / threshold) ** 2))

    return {
        "scene": scene_name,
        "direction": direction,
        "frames": nfiles,
        "output": out_name,
        "image_shape": list(image_shape),
        "sample_stride": sample_stride,
        "single_frame_rms_noise_dn_sampled": single_frame_rms_noise,
        "estimated_average_rms_noise_at_full_count_dn": single_frame_rms_noise / math.sqrt(nfiles),
        "recommended_frames_for_average_noise_threshold_dn": recommended,
        "rms_difference_to_full_mean_dn_sampled": convergence,
        "seconds": round(time.time() - started, 2),
    }


def main() -> None:
    args = parse_args()
    root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.sample_stride < 1:
        raise ValueError("--sample-stride must be >= 1")

    scenes = list_scenes(root, args.scenes)
    summary = {
        "data_root": str(root),
        "output_dir": str(output_dir),
        "directions": args.directions,
        "pattern": args.pattern,
        "groups": [],
    }

    started = time.time()
    for scene in scenes:
        for direction in args.directions:
            direction_dir = scene / direction
            if not direction_dir.is_dir():
                print(f"SKIP {scene.name}/{direction}: folder not found", flush=True)
                continue
            files = sorted(direction_dir.rglob(args.pattern))
            if not files:
                print(f"SKIP {scene.name}/{direction}: no images", flush=True)
                continue
            result = process_group(
                scene_name=scene.name,
                direction=direction,
                files=files,
                output_dir=output_dir,
                sample_stride=args.sample_stride,
                milestones=args.milestones,
                thresholds=args.noise_thresholds,
                overwrite=args.overwrite,
            )
            summary["groups"].append(result)
            recommendations = result["recommended_frames_for_average_noise_threshold_dn"]
            print(
                f"DONE {scene.name}/{direction}: saved {result['output']}; "
                f"single-frame noise={result['single_frame_rms_noise_dn_sampled']:.3f} DN; "
                f"recommended={recommendations}",
                flush=True,
            )

    summary["seconds_total"] = round(time.time() - started, 2)
    summary_path = output_dir / args.summary_name
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"SUMMARY {summary_path}", flush=True)


if __name__ == "__main__":
    main()
