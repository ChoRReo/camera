#!/usr/bin/env python3
"""Compare DoLP, DoCP, and AoP stability at different averaging frame counts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from compute_polarization_maps import DIRECTIONS, compute_maps, convert_channel


DEFAULT_COUNTS = (8, 16, 32, 64, 100, 128, 200, 256, 400, 512)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use raw polarization frame folders to test how many frames are enough. "
            "For each requested frame count, the script averages the first N frames "
            "per direction, computes DoLP/DoCP/AoP, and compares them with a full-count "
            "reference computed from all available frames or --reference-count frames."
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
        default="frame_count_comparison",
        help="Output directory for JSON/CSV summaries. Default: frame_count_comparison",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        help="Optional scene names to process, for example: --scenes 00001 00002",
    )
    parser.add_argument(
        "--directions",
        nargs="+",
        default=list(DIRECTIONS),
        help="Direction folder names. Default: 0 45 90 135 L R",
    )
    parser.add_argument(
        "--pattern",
        default="*.png",
        help="Image filename glob used recursively inside each direction. Default: *.png",
    )
    parser.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_COUNTS),
        help="Frame counts to compare against the reference. Default: 8 16 32 64 100 128 200 256 400 512",
    )
    parser.add_argument(
        "--reference-count",
        type=int,
        default=0,
        help="Reference frame count. 0 means use all common frames in the direction folders. Default: 0",
    )
    parser.add_argument(
        "--channel",
        choices=("luma", "mean", "r", "g", "b", "rgb"),
        default="luma",
        help="How RGB frames are converted before computing maps. Default: luma",
    )
    parser.add_argument(
        "--s0-mode",
        choices=("linear-average", "0-90", "45-135", "all-average"),
        default="linear-average",
        help="S0 estimate, matching compute_polarization_maps.py. Default: linear-average",
    )
    parser.add_argument(
        "--circular-sign",
        choices=("R-minus-L", "L-minus-R"),
        default="R-minus-L",
        help="DoCP sign convention. Default: R-minus-L",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1e-6,
        help="Small denominator floor to avoid divide-by-zero. Default: 1e-6",
    )
    parser.add_argument(
        "--metric-stride",
        type=int,
        default=4,
        help=(
            "Pixel stride used when computing error statistics. 1 uses every pixel; "
            "4 samples a 2048x2048 image as a 512x512 grid. Default: 4"
        ),
    )
    parser.add_argument(
        "--dolp-rms-threshold",
        type=float,
        default=0.005,
        help="DoLP RMS error threshold used for recommended count. Default: 0.005",
    )
    parser.add_argument(
        "--docp-rms-threshold",
        type=float,
        default=0.005,
        help="DoCP RMS error threshold used for recommended count. Default: 0.005",
    )
    parser.add_argument(
        "--aop-rms-deg-threshold",
        type=float,
        default=1.0,
        help="AoP RMS angular error threshold in degrees used for recommended count. Default: 1.0",
    )
    parser.add_argument(
        "--save-maps",
        action="store_true",
        help="Also save float32 .npy DoLP/DoCP/AoP maps for each tested count and the reference.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing outputs.",
    )
    return parser.parse_args()


def list_scenes(root: Path, selected: list[str] | None) -> list[Path]:
    scenes = [root / scene for scene in selected] if selected else sorted(p for p in root.iterdir() if p.is_dir())
    missing = [str(scene) for scene in scenes if not scene.is_dir()]
    if missing:
        raise FileNotFoundError(f"Scene folder(s) not found: {', '.join(missing)}")
    return scenes


def read_frame(path: Path, channel: str) -> np.ndarray:
    with Image.open(path) as image:
        frame = np.asarray(image)
    return convert_channel(frame.astype(np.float32), channel)


def list_direction_files(scene: Path, direction: str, pattern: str) -> list[Path]:
    direction_dir = scene / direction
    if not direction_dir.is_dir():
        raise FileNotFoundError(f"Missing direction folder: {direction_dir}")
    files = sorted(direction_dir.rglob(pattern))
    if not files:
        raise FileNotFoundError(f"No images found in {direction_dir} with pattern {pattern!r}")
    return files


def effective_reference_count(file_counts: dict[str, int], requested: int) -> int:
    common_count = min(file_counts.values())
    if requested <= 0:
        return common_count
    if requested > common_count:
        raise ValueError(
            f"--reference-count {requested} is larger than the smallest direction count {common_count}"
        )
    return requested


def normalize_counts(counts: list[int], reference_count: int) -> list[int]:
    valid = sorted(set(count for count in counts if 1 <= count <= reference_count))
    if not valid:
        raise ValueError(f"No requested --counts are between 1 and reference count {reference_count}")
    if reference_count not in valid:
        valid.append(reference_count)
    return sorted(valid)


def average_direction_at_counts(
    files: list[Path],
    counts: list[int],
    reference_count: int,
    channel: str,
    scene_name: str,
    direction: str,
) -> dict[int, np.ndarray]:
    requested = set(counts)
    averages = {}
    accumulator = None
    image_shape = None
    started = time.time()

    for index, path in enumerate(files[:reference_count], start=1):
        frame = read_frame(path, channel)
        if accumulator is None:
            image_shape = frame.shape
            accumulator = np.zeros(image_shape, dtype=np.float64)
        elif frame.shape != image_shape:
            raise ValueError(
                f"Shape mismatch in {scene_name}/{direction}: {path} has {frame.shape}, expected {image_shape}"
            )

        accumulator += frame
        if index in requested:
            averages[index] = (accumulator / index).astype(np.float32)

        if index % 200 == 0 or index == reference_count:
            print(f"{scene_name}/{direction}: {index}/{reference_count}", flush=True)

    print(f"DONE average {scene_name}/{direction} in {time.time() - started:.1f}s", flush=True)
    return averages


def sample_metric_grid(values: np.ndarray, stride: int) -> np.ndarray:
    if stride <= 1:
        return values
    if values.ndim == 2:
        return values[::stride, ::stride]
    if values.ndim == 3:
        return values[::stride, ::stride, ...]
    raise ValueError(f"Unsupported map shape: {values.shape}")


def scalar_error_stats(error: np.ndarray) -> dict[str, float]:
    finite = np.abs(error[np.isfinite(error)])
    if finite.size == 0:
        return {"mean_abs": float("nan"), "rms": float("nan"), "p95_abs": float("nan"), "max_abs": float("nan")}
    return {
        "mean_abs": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(finite * finite))),
        "p95_abs": float(np.percentile(finite, 95)),
        "max_abs": float(np.max(finite)),
    }


def aop_difference_rad(aop: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (aop - reference + math.pi / 2.0) % math.pi - math.pi / 2.0


def compare_maps(maps: dict[str, np.ndarray], reference: dict[str, np.ndarray], metric_stride: int) -> dict:
    dolp_error = sample_metric_grid(maps["dolp"] - reference["dolp"], metric_stride)
    docp_error = sample_metric_grid(maps["docp"] - reference["docp"], metric_stride)
    aop_error = sample_metric_grid(aop_difference_rad(maps["aop_rad"], reference["aop_rad"]), metric_stride)

    dolp = scalar_error_stats(dolp_error)
    docp = scalar_error_stats(docp_error)
    aop_rad = scalar_error_stats(aop_error)
    aop_deg = {key: float(value * 180.0 / math.pi) for key, value in aop_rad.items()}
    return {
        "dolp": dolp,
        "docp": docp,
        "aop_rad": aop_rad,
        "aop_deg": aop_deg,
    }


def build_maps_for_count(
    averages_by_direction: dict[str, dict[int, np.ndarray]],
    count: int,
    s0_mode: str,
    circular_sign: str,
    epsilon: float,
) -> dict[str, np.ndarray]:
    images = {direction: by_count[count] for direction, by_count in averages_by_direction.items()}
    return compute_maps(images, s0_mode, circular_sign, epsilon)


def save_maps(output_dir: Path, scene: str, count_label: str, maps: dict[str, np.ndarray]) -> dict[str, str]:
    map_dir = output_dir / scene / count_label
    map_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name in ("dolp", "docp", "aop_rad"):
        path = map_dir / f"{name}.npy"
        np.save(path, maps[name].astype(np.float32))
        outputs[name] = str(path.relative_to(output_dir))
    return outputs


def process_scene(scene: Path, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    files_by_direction = {
        direction: list_direction_files(scene, direction, args.pattern)
        for direction in args.directions
    }
    file_counts = {direction: len(files) for direction, files in files_by_direction.items()}
    reference_count = effective_reference_count(file_counts, args.reference_count)
    counts = normalize_counts(args.counts, reference_count)

    print(
        f"SCENE {scene.name}: reference={reference_count}, counts={counts}, "
        f"available={file_counts}",
        flush=True,
    )

    averages_by_direction = {}
    for direction, files in files_by_direction.items():
        averages_by_direction[direction] = average_direction_at_counts(
            files=files,
            counts=counts,
            reference_count=reference_count,
            channel=args.channel,
            scene_name=scene.name,
            direction=direction,
        )

    reference_maps = build_maps_for_count(
        averages_by_direction,
        reference_count,
        args.s0_mode,
        args.circular_sign,
        args.epsilon,
    )
    saved_maps = {}
    if args.save_maps:
        saved_maps["reference"] = save_maps(args.output_dir, scene.name, f"reference_{reference_count}", reference_maps)

    rows = []
    comparisons = []
    recommended_count = None
    for count in counts:
        maps = build_maps_for_count(
            averages_by_direction,
            count,
            args.s0_mode,
            args.circular_sign,
            args.epsilon,
        )
        metrics = compare_maps(maps, reference_maps, args.metric_stride)
        if args.save_maps:
            saved_maps[str(count)] = save_maps(args.output_dir, scene.name, f"count_{count}", maps)

        passed = (
            metrics["dolp"]["rms"] <= args.dolp_rms_threshold
            and metrics["docp"]["rms"] <= args.docp_rms_threshold
            and metrics["aop_deg"]["rms"] <= args.aop_rms_deg_threshold
        )
        if recommended_count is None and passed:
            recommended_count = count

        comparison = {
            "count": count,
            "is_reference": count == reference_count,
            "passes_thresholds": passed,
            "metrics": metrics,
        }
        comparisons.append(comparison)
        rows.append(
            {
                "scene": scene.name,
                "count": count,
                "reference_count": reference_count,
                "is_reference": count == reference_count,
                "passes_thresholds": passed,
                "dolp_rms": metrics["dolp"]["rms"],
                "dolp_p95_abs": metrics["dolp"]["p95_abs"],
                "docp_rms": metrics["docp"]["rms"],
                "docp_p95_abs": metrics["docp"]["p95_abs"],
                "aop_rms_deg": metrics["aop_deg"]["rms"],
                "aop_p95_abs_deg": metrics["aop_deg"]["p95_abs"],
                "aop_max_abs_deg": metrics["aop_deg"]["max_abs"],
            }
        )

        print(
            f"{scene.name} N={count}: DoLP rms={metrics['dolp']['rms']:.6f}, "
            f"DoCP rms={metrics['docp']['rms']:.6f}, AoP rms={metrics['aop_deg']['rms']:.3f} deg",
            flush=True,
        )

    scene_summary = {
        "scene": scene.name,
        "file_counts": file_counts,
        "reference_count": reference_count,
        "counts": counts,
        "channel": args.channel,
        "s0_mode": args.s0_mode,
        "circular_sign": args.circular_sign,
        "metric_stride": args.metric_stride,
        "thresholds": {
            "dolp_rms": args.dolp_rms_threshold,
            "docp_rms": args.docp_rms_threshold,
            "aop_rms_deg": args.aop_rms_deg_threshold,
        },
        "recommended_count": recommended_count,
        "comparisons": comparisons,
        "saved_maps": saved_maps,
    }
    return scene_summary, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "scene",
        "count",
        "reference_count",
        "is_reference",
        "passes_thresholds",
        "dolp_rms",
        "dolp_p95_abs",
        "docp_rms",
        "docp_p95_abs",
        "aop_rms_deg",
        "aop_p95_abs_deg",
        "aop_max_abs_deg",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.data_root = Path(args.data_root).expanduser().resolve()
    args.output_dir = Path(args.output_dir).expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive.")
    if args.metric_stride < 1:
        raise ValueError("--metric-stride must be >= 1.")
    if set(args.directions) != set(DIRECTIONS):
        raise ValueError("This comparison requires all six directions: 0 45 90 135 L R.")

    summary_path = args.output_dir / "frame_count_comparison_summary.json"
    csv_path = args.output_dir / "frame_count_comparison.csv"
    if not args.overwrite:
        existing = [str(path) for path in (summary_path, csv_path) if path.exists()]
        if existing:
            raise FileExistsError(f"Output exists: {', '.join(existing)}. Pass --overwrite.")

    scenes = list_scenes(args.data_root, args.scenes)
    summary = {
        "data_root": str(args.data_root),
        "output_dir": str(args.output_dir),
        "method": (
            "Each count uses the first N sorted frames in every direction. "
            "Errors are measured against maps computed from reference_count frames."
        ),
        "scenes": [],
    }
    rows = []
    started = time.time()

    for scene in scenes:
        scene_summary, scene_rows = process_scene(scene, args)
        summary["scenes"].append(scene_summary)
        rows.extend(scene_rows)

    summary["seconds_total"] = round(time.time() - started, 2)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    write_csv(csv_path, rows)
    print(f"SUMMARY {summary_path}", flush=True)
    print(f"CSV {csv_path}", flush=True)


if __name__ == "__main__":
    main()
