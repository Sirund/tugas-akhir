"""
Visualize selected SAM3 segmentation outputs.

Reads segmentation_selected.csv + JSON shapes, draws overlays on filter source
images, and saves to {good,bad_outliers,bad_negative}/ subdirectories.

Usage:
    python -m scripts.dataset.visualize_segmentation \
        --filter_dir ../dataset/pipeline2.0/filter \
        --seg_dir ../dataset/pipeline2.0/segmentation \
        --selected_csv ../dataset/pipeline2.0/segmentation_selected.csv \
        --output_dir ../dataset/pipeline2.0/segmentation/visualization \
        --samples 20
"""

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TARGET_COLOR = (0, 255, 0)
IGNORE_COLOR = (0, 0, 255)


def draw_shapes(img: np.ndarray, shapes: list, title: str = "") -> np.ndarray:
    vis = img.copy()
    overlay = np.zeros_like(img)

    for i, shape in enumerate(shapes):
        pts = np.array(shape["points"], np.int32).reshape((-1, 1, 2))
        label = shape.get("label", "target")
        color = IGNORE_COLOR if label == "ignore" else TARGET_COLOR

        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(vis, [pts], True, color, thickness=2)

        M = cv2.moments(pts)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(
                vis, f"#{i} ({label})", (cx + 5, cy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA,
            )

    vis = cv2.addWeighted(vis, 1.0, overlay, 0.35, 0)

    if title:
        cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(
            vis, title, (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    if len(shapes) == 0:
        cv2.putText(
            vis, "NEGATIVE — no shapes", (20, img.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
        )

    return vis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize selected SAM3 segmentation outputs"
    )
    parser.add_argument(
        "--filter_dir",
        required=True,
        help="Filter directory with source images (e.g. pipeline2.0/filter)",
    )
    parser.add_argument(
        "--seg_dir",
        required=True,
        help="Segmentation directory with JSONs (e.g. pipeline2.0/segmentation)",
    )
    parser.add_argument(
        "--selected_csv",
        required=True,
        help="Path to segmentation_selected.csv",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Visualization output directory",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Max samples per category per subfolder (default: %(default)s)",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    filter_dir = Path(args.filter_dir)
    seg_dir = Path(args.seg_dir)
    vis_dir = Path(args.output_dir)
    samples_per_cat = args.samples

    entries = []
    with open(args.selected_csv) as f:
        for row in csv.DictReader(f):
            entries.append({
                "category": row["category"],
                "instance_id": row["instance_id"],
                "safety": row["safety"],
                "task": row["task"],
                "image_name": row["selected_image"],
                "num_target_shapes": int(row["num_target_shapes"]),
                "num_ignore_shapes": int(row.get("num_ignore_shapes", "0")),
            })

    logger.info("Loaded %d selected entries", len(entries))

    by_cat: dict = defaultdict(list)
    for e in entries:
        by_cat[e["category"]].append(e)

    for cat, cat_entries in by_cat.items():
        sorted_by_shapes = sorted(
            cat_entries, key=lambda r: r["num_target_shapes"], reverse=True
        )
        ones = [e for e in cat_entries if e["num_target_shapes"] == 1]
        negs = [e for e in cat_entries if e["num_target_shapes"] == 0]

        good = ones[:samples_per_cat]
        outliers = sorted_by_shapes[:samples_per_cat]
        negatives = negs[:samples_per_cat]

        for cat_name, samples in [
            ("good", good),
            ("bad_outliers", outliers),
            ("bad_negative", negatives),
        ]:
            out_dir = vis_dir / cat_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for entry in samples:
                stem = Path(entry["image_name"]).stem
                safety = entry["safety"]
                task = entry["task"]
                category = entry["category"]
                instance_id = entry["instance_id"]

                src_path = filter_dir / safety / entry["image_name"]
                img = cv2.imread(str(src_path))
                if img is None:
                    logger.warning("Cannot load %s", src_path)
                    continue

                json_path = seg_dir / task / safety / f"{category}_{instance_id}.json"
                shapes = []
                if json_path.exists():
                    with open(json_path) as f:
                        data = json.load(f)
                    shapes = data.get("shapes", [])

                n_target = entry["num_target_shapes"]
                n_ignore = entry["num_ignore_shapes"]
                title = f"{stem} | {task} | {n_target}S/{n_ignore}I"
                vis = draw_shapes(img, shapes, title=title)
                out_path = out_dir / f"{stem}.png"
                cv2.imwrite(str(out_path), vis)

        logger.info(
            "  %s: %d/%d/%d good/outlier/negative",
            cat, len(good), len(outliers), len(negatives),
        )

    logger.info("Visualization done — %s", vis_dir)


if __name__ == "__main__":
    main()
