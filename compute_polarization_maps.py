#!/usr/bin/env python3
"""Compute DoLP, DoCP, and AoP maps from averaged polarization images."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


DIRECTIONS = ("0", "45", "90", "135", "L", "R")
LINEAR_DIRECTIONS = ("0", "45", "90", "135")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute polarization maps from averaged images. Supported layouts: "
            "0.png/45.png/... in one directory, <scene>/0.png/<scene>/45.png/... "
            "for multiple scenes, or the old <scene>_0_avg.png format.\n\n"
            "Stokes convention: S1 = I0 - I90, S2 = I45 - I135, "
            "DoLP = sqrt(S1^2 + S2^2) / S0, AoP = 0.5 * atan2(S2, S1). "
            "DoCP uses either S3 = IR - IL or S3 = IL - IR, controlled by "
            "--circular-sign."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        default="./",
        help="Directory containing averaged PNGs. Default: current directory",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=".",
        help="Where maps are saved. Default: current directory",
    )
    parser.add_argument(
        "--scenes",
        nargs="*",
        help="Optional scene prefixes to process, for example: --scenes 00001 00002",
    )
    parser.add_argument(
        "--suffix",
        default="_avg.png",
        help="Suffix for --input-layout prefixed. Default: _avg.png",
    )
    parser.add_argument(
        "--input-layout",
        choices=("auto", "flat", "scene-subdir", "prefixed"),
        default="auto",
        help=(
            "Averaged-image layout. flat reads 0.png, 45.png, ... from input_dir; "
            "scene-subdir reads <scene>/0.png, ...; prefixed reads <scene>_0_avg.png, ...; "
            "auto detects in that order. Default: auto"
        ),
    )
    parser.add_argument(
        "--channel",
        choices=("luma", "mean", "r", "g", "b", "rgb"),
        default="rgb",
        help=(
            "How RGB inputs are used before computing maps. rgb computes "
            "per-channel maps. Default: rgb"
        ),
    )
    parser.add_argument(
        "--s0-mode",
        choices=("linear-average", "0-90", "45-135", "all-average"),
        default="linear-average",
        help=(
            "S0 estimate. linear-average = 0.5*((I0+I90)+(I45+I135)); "
            "all-average also includes IL+IR. Default: linear-average"
        ),
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
        "--no-npy",
        action="store_true",
        help="Only save PNG visualizations, not float32 .npy arrays.",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Only save float32 .npy arrays, not PNG visualizations.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image).astype(np.float32)


def convert_channel(image: np.ndarray, channel: str) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    if channel == "rgb":
        return image[..., :3]
    if channel == "r":
        return image[..., 0]
    if channel == "g":
        return image[..., 1]
    if channel == "b":
        return image[..., 2]
    if channel == "mean":
        return image[..., :3].mean(axis=-1)
    if channel == "luma":
        rgb = image[..., :3]
        return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    raise ValueError(f"Unsupported channel mode: {channel}")


def direction_filename(direction: str) -> str:
    return f"{direction.replace('/', '_')}.png"


def has_direction_set(paths: dict[str, Path]) -> bool:
    return all(path.is_file() for path in paths.values())


def image_path(input_dir: Path, scene: str, direction: str, suffix: str, layout: str) -> Path:
    if layout == "flat":
        return input_dir / direction_filename(direction)
    if layout == "scene-subdir":
        return input_dir / scene / direction_filename(direction)
    if layout == "prefixed":
        return input_dir / f"{scene}_{direction}{suffix}"
    raise ValueError(f"Unsupported input layout: {layout}")


def prefixed_image_path(input_dir: Path, scene: str, direction: str, suffix: str) -> Path:
    return input_dir / f"{scene}_{direction}{suffix}"


def discover_prefixed_scenes(input_dir: Path, suffix: str) -> list[str]:
    marker = f"_0{suffix}"
    scenes = []
    for path in sorted(input_dir.glob(f"*{marker}")):
        name = path.name
        scenes.append(name[: -len(marker)])
    return scenes


def direction_paths(input_dir: Path, scene: str, suffix: str, layout: str) -> dict[str, Path]:
    return {
        direction: image_path(input_dir, scene, direction, suffix, layout)
        for direction in DIRECTIONS
    }


def discover_scene_subdirs(input_dir: Path) -> list[str]:
    scenes = []
    for path in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        paths = {direction: path / direction_filename(direction) for direction in DIRECTIONS}
        if has_direction_set(paths):
            scenes.append(path.name)
    return scenes


def flat_paths(input_dir: Path) -> dict[str, Path]:
    return {direction: input_dir / direction_filename(direction) for direction in DIRECTIONS}


def choose_layout_and_scenes(
    input_dir: Path,
    suffix: str,
    input_layout: str,
    selected_scenes: list[str] | None,
) -> tuple[str, list[str]]:
    if input_layout == "flat":
        if selected_scenes and len(selected_scenes) > 1:
            raise ValueError("--input-layout flat only supports one scene.")
        return "flat", selected_scenes or [""]
    if input_layout == "scene-subdir":
        scenes = selected_scenes or discover_scene_subdirs(input_dir)
        return "scene-subdir", scenes
    if input_layout == "prefixed":
        scenes = selected_scenes or discover_prefixed_scenes(input_dir, suffix)
        return "prefixed", scenes

    if selected_scenes:
        if len(selected_scenes) == 1 and has_direction_set(flat_paths(input_dir)):
            return "flat", selected_scenes
        scene_paths = [direction_paths(input_dir, scene, suffix, "scene-subdir") for scene in selected_scenes]
        if all(has_direction_set(paths) for paths in scene_paths):
            return "scene-subdir", selected_scenes
        prefixed_paths = [direction_paths(input_dir, scene, suffix, "prefixed") for scene in selected_scenes]
        if all(has_direction_set(paths) for paths in prefixed_paths):
            return "prefixed", selected_scenes
        return "scene-subdir", selected_scenes

    if has_direction_set(flat_paths(input_dir)):
        return "flat", [""]
    scene_subdirs = discover_scene_subdirs(input_dir)
    if scene_subdirs:
        return "scene-subdir", scene_subdirs
    return "prefixed", discover_prefixed_scenes(input_dir, suffix)


def validate_inputs(input_dir: Path, scene: str, suffix: str, layout: str) -> dict[str, Path]:
    paths = direction_paths(input_dir, scene, suffix, layout)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        scene_label = scene or "flat input"
        raise FileNotFoundError(f"Missing averaged image(s) for scene {scene_label}: {', '.join(missing)}")
    return paths


def estimate_s0(images: dict[str, np.ndarray], mode: str) -> np.ndarray:
    s0_090 = images["0"] + images["90"]
    s0_45135 = images["45"] + images["135"]
    if mode == "0-90":
        return s0_090
    if mode == "45-135":
        return s0_45135
    if mode == "linear-average":
        return 0.5 * (s0_090 + s0_45135)
    if mode == "all-average":
        return (s0_090 + s0_45135 + images["L"] + images["R"]) / 3.0
    raise ValueError(f"Unsupported S0 mode: {mode}")


def compute_maps(
    images: dict[str, np.ndarray],
    s0_mode: str,
    circular_sign: str,
    epsilon: float,
) -> dict[str, np.ndarray]:
    shapes = {name: image.shape for name, image in images.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"Input shapes do not match: {shapes}")

    s0 = estimate_s0(images, s0_mode)
    s1 = images["0"] - images["90"]
    s2 = images["45"] - images["135"]
    if circular_sign == "R-minus-L":
        s3 = images["R"] - images["L"]
    elif circular_sign == "L-minus-R":
        s3 = images["L"] - images["R"]
    else:
        raise ValueError(f"Unsupported circular sign: {circular_sign}")

    denominator = np.maximum(s0, epsilon)
    dolp = np.sqrt(s1 * s1 + s2 * s2) / denominator
    docp = s3 / denominator
    aop = np.mod(0.5 * np.arctan2(s2, s1), math.pi)

    return {
        "s0": s0.astype(np.float32),
        "s1": s1.astype(np.float32),
        "s2": s2.astype(np.float32),
        "s3": s3.astype(np.float32),
        "dolp": dolp.astype(np.float32),
        "docp": docp.astype(np.float32),
        "aop_rad": aop.astype(np.float32),
    }


def to_uint16_unit_interval(values: np.ndarray) -> np.ndarray:
    return np.rint(np.clip(values, 0.0, 1.0) * 65535.0).astype(np.uint16)


def to_uint16_signed_unit(values: np.ndarray) -> np.ndarray:
    scaled = (np.clip(values, -1.0, 1.0) + 1.0) * 0.5
    return np.rint(scaled * 65535.0).astype(np.uint16)


def to_uint16_aop(values: np.ndarray) -> np.ndarray:
    return np.rint(np.mod(values, math.pi) / math.pi * 65535.0).astype(np.uint16)


def save_png(path: Path, data: np.ndarray) -> None:
    if data.ndim == 2:
        Image.fromarray(data).save(path)
        return
    if data.ndim == 3 and data.shape[-1] == 3:
        preview = np.rint(data.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(preview).save(path)
        return
    raise ValueError(f"Cannot save PNG for data shape {data.shape}")


def write_outputs(
    output_dir: Path,
    scene: str,
    maps: dict[str, np.ndarray],
    save_npy: bool,
    save_png_files: bool,
    overwrite: bool,
) -> dict[str, str]:
    outputs = {}
    products = {
        "dolp": maps["dolp"],
        "docp": maps["docp"],
        "aop_rad": maps["aop_rad"],
    }

    for name, values in products.items():
        base_name = f"{scene}_{name}" if scene else name
        base = output_dir / base_name
        npy_path = base.with_suffix(".npy")
        png_path = base.with_suffix(".png")
        planned = []
        if save_npy:
            planned.append(npy_path)
        if save_png_files:
            planned.append(png_path)
        if not overwrite:
            existing = [str(path) for path in planned if path.exists()]
            if existing:
                raise FileExistsError(f"Output exists: {', '.join(existing)}. Pass --overwrite.")

        if save_npy:
            np.save(npy_path, values.astype(np.float32))
            outputs[f"{name}_npy"] = npy_path.name
        if save_png_files:
            if name == "dolp":
                png_data = to_uint16_unit_interval(values)
            elif name == "docp":
                png_data = to_uint16_signed_unit(values)
            elif name == "aop_rad":
                png_data = to_uint16_aop(values)
            else:
                raise ValueError(f"Unsupported product: {name}")
            save_png(png_path, png_data)
            outputs[f"{name}_png"] = png_path.name
    return outputs


def summarize_map(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan"), "mean": float("nan")}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
    }


def process_scene(
    input_dir: Path,
    output_dir: Path,
    scene: str,
    suffix: str,
    input_layout: str,
    channel: str,
    s0_mode: str,
    circular_sign: str,
    epsilon: float,
    save_npy: bool,
    save_png_files: bool,
    overwrite: bool,
) -> dict:
    paths = validate_inputs(input_dir, scene, suffix, input_layout)
    images = {
        direction: convert_channel(read_image(path), channel)
        for direction, path in paths.items()
    }
    maps = compute_maps(images, s0_mode, circular_sign, epsilon)
    outputs = write_outputs(output_dir, scene, maps, save_npy, save_png_files, overwrite)
    return {
        "scene": scene or "",
        "input_layout": input_layout,
        "channel": channel,
        "s0_mode": s0_mode,
        "circular_sign": circular_sign,
        "inputs": {direction: path.name for direction, path in paths.items()},
        "outputs": outputs,
        "stats": {
            "dolp": summarize_map(maps["dolp"]),
            "docp": summarize_map(maps["docp"]),
            "aop_rad": summarize_map(maps["aop_rad"]),
        },
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.no_npy and args.no_png:
        raise ValueError("At least one of NPY or PNG output must be enabled.")
    if args.epsilon <= 0:
        raise ValueError("--epsilon must be positive.")

    input_layout, scenes = choose_layout_and_scenes(
        input_dir=input_dir,
        suffix=args.suffix,
        input_layout=args.input_layout,
        selected_scenes=args.scenes,
    )
    if not scenes:
        raise FileNotFoundError(
            f"No complete averaged-image sets found in {input_dir} with layout {args.input_layout!r}"
        )
    summary_path = output_dir / "polarization_maps_summary.json"
    if summary_path.exists() and not args.overwrite:
        raise FileExistsError(f"{summary_path} exists. Pass --overwrite.")

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_layout": input_layout,
        "suffix": args.suffix,
        "png_encoding": {
            "dolp": "uint16, 0 maps to 0 and 1 maps to 65535",
            "docp": "uint16, -1 maps to 0, 0 maps to 32768, 1 maps to 65535",
            "aop_rad": "uint16, 0 maps to 0 and pi maps to 65535",
        },
        "scenes": [],
    }

    for scene in scenes:
        result = process_scene(
            input_dir=input_dir,
            output_dir=output_dir,
            scene=scene,
            suffix=args.suffix,
            input_layout=input_layout,
            channel=args.channel,
            s0_mode=args.s0_mode,
            circular_sign=args.circular_sign,
            epsilon=args.epsilon,
            save_npy=not args.no_npy,
            save_png_files=not args.no_png,
            overwrite=args.overwrite,
        )
        summary["scenes"].append(result)
        scene_label = scene or "flat input"
        print(
            f"DONE {scene_label}: saved DoLP/DoCP/AoP; "
            f"DoLP mean={result['stats']['dolp']['mean']:.4f}, "
            f"DoCP mean={result['stats']['docp']['mean']:.4f}",
            flush=True,
        )

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"SUMMARY {summary_path}", flush=True)


if __name__ == "__main__":
    main()
