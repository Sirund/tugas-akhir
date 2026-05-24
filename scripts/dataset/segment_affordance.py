"""
SAM3 affordance segmentation — hierarchy-based contours + ranking.

Pipeline 2.0 step: segment filtered images with SAM3, extract contours,
rank per (instance, task), and output best JSON.

Usage:
    python -m scripts.dataset.segment_affordance \
        --filter_dir ../dataset/pipeline2.0/filter \
        --output_dir ../dataset/pipeline2.0
"""

import argparse
import csv
import gc
import json as json_mod
import logging
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Sam3Model, Sam3Processor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAM3_RES = 288
OUTPUT_RES = 1024
SCALE = OUTPUT_RES / SAM3_RES

THRESHOLD_SAM3 = 0.5
THRESHOLD_VALID_FRAGMENT = 0.4
MASK_THRESHOLD = 0.5

CLOSE_KERNEL_SIZE = 7
OPEN_KERNEL_SIZE = 3
EPSILON_FACTOR = 0.002

SAMPLES_PER_CATEGORY = 20

# ---------------------------------------------------------------------------
# Prompt map (per-category-per-task, single-word)
# ---------------------------------------------------------------------------
CATEGORY_PROMPT_MAP: Dict[Tuple[str, str], str] = {
    ("scissors", "handover"): "blade",
    ("shears", "handover"): "blade",
    ("fork", "handover"): "prongs",
    ("pitchfork", "handover"): "prongs",
    ("knife", "handover"): "blade",
    ("steak_knife", "handover"): "blade",
    ("pocketknife", "handover"): "blade",
    ("scissors", "pick_and_place"): "handle",
    ("shears", "pick_and_place"): "handle",
    ("fork", "pick_and_place"): "handle",
    ("pitchfork", "pick_and_place"): "handle",
    ("knife", "pick_and_place"): "handle",
    ("steak_knife", "pick_and_place"): "handle",
    ("pocketknife", "pick_and_place"): "handle",
    ("kettle", "handover"): "handle",
    ("teakettle", "handover"): "handle",
    ("coffeepot", "handover"): "handle",
    ("cup", "handover"): "handle",
    ("cappuccino", "handover"): "handle",
    ("mug", "handover"): "handle",
    ("teacup", "handover"): "handle",
    ("spoon", "handover"): "handle",
    ("wooden_spoon", "handover"): "handle",
    ("soupspoon", "handover"): "handle",
    ("kettle", "pick_and_place"): "handle",
    ("teakettle", "pick_and_place"): "handle",
    ("coffeepot", "pick_and_place"): "handle",
    ("cup", "pick_and_place"): "handle",
    ("cappuccino", "pick_and_place"): "handle",
    ("mug", "pick_and_place"): "handle",
    ("teacup", "pick_and_place"): "handle",
    ("spoon", "pick_and_place"): "handle",
    ("wooden_spoon", "pick_and_place"): "handle",
    ("soupspoon", "pick_and_place"): "handle",
}

# Taxonomy for safety lookup
TAXONOMY = {
    "scissors": "dangerous", "shears": "dangerous", "fork": "dangerous",
    "pitchfork": "dangerous", "knife": "dangerous", "steak_knife": "dangerous",
    "pocketknife": "dangerous",
    "kettle": "non-dangerous", "teakettle": "non-dangerous",
    "coffeepot": "non-dangerous", "cup": "non-dangerous",
    "cappuccino": "non-dangerous", "mug": "non-dangerous",
    "teacup": "non-dangerous", "spoon": "non-dangerous",
    "wooden_spoon": "non-dangerous", "soupspoon": "non-dangerous",
}

# ---------------------------------------------------------------------------
# SAM3 global cache
# ---------------------------------------------------------------------------
_sam3_model = None
_sam3_processor = None
_device = None


def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _sam3_pretrained_name() -> str:
    snapshots = sorted(
        Path.home().glob(".cache/huggingface/hub/models--facebook--sam3/snapshots/*/")
    )
    if snapshots:
        return str(snapshots[-1])
    return "facebook/sam3"


def load_sam3() -> Tuple[Sam3Model, Sam3Processor]:
    global _sam3_model, _sam3_processor
    if _sam3_model is None:
        device = get_device()
        model_name = _sam3_pretrained_name()
        logger.info("Loading SAM3 from %s (fp16)", model_name)
        _sam3_model = Sam3Model.from_pretrained(
            model_name, torch_dtype=torch.float16, token=True
        ).to(device)
        _sam3_model.eval()
        _sam3_processor = Sam3Processor.from_pretrained(model_name, token=True)
        logger.info("SAM3 loaded in fp16")
    return _sam3_model, _sam3_processor


def unload_sam3() -> None:
    global _sam3_model, _sam3_processor
    if _sam3_model is not None:
        del _sam3_model
        _sam3_model = None
    if _sam3_processor is not None:
        del _sam3_processor
        _sam3_processor = None
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("SAM3 model unloaded, VRAM cleared")


# ---------------------------------------------------------------------------
# Image filename parsing
# ---------------------------------------------------------------------------

def parse_image_name(image_name: str) -> Tuple[str, str]:
    """Parse '{category}_{instance}_{frame}.png' → (category, instance_id).

    Handles multi-word categories like 'wooden_spoon', 'steak_knife' by
    matching against known taxonomy keys (longest match first).
    """
    stem = Path(image_name).stem
    parts = stem.split("_")
    sorted_cats = sorted(TAXONOMY.keys(), key=len, reverse=True)
    for cat in sorted_cats:
        prefix = cat.split("_")
        if len(parts) >= len(prefix) + 1 and parts[:len(prefix)] == prefix:
            instance_id = parts[len(prefix)]
            return cat, instance_id
    raise ValueError(f"Cannot parse image name: {image_name} (parts={parts})")


# ---------------------------------------------------------------------------
# SAM3 inference
# ---------------------------------------------------------------------------

def segment_with_sam3(
    image_pil: Image.Image,
    text_prompt: str,
) -> Tuple[List[np.ndarray], List[float]]:
    """Run SAM3 with threshold=0.5, returns (masks_list, scores_list).

    Each mask is uint8 (288, 288), values 0 or 255.
    """
    model, processor = load_sam3()
    device = get_device()

    inputs = processor(
        images=image_pil,
        text=text_prompt,
        return_tensors="pt",
    ).to(device, dtype=torch.float16)

    target_sizes = [[SAM3_RES, SAM3_RES]]

    with torch.inference_mode():
        outputs = model(**inputs)

    try:
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=THRESHOLD_SAM3,
            mask_threshold=MASK_THRESHOLD,
            target_sizes=target_sizes,
        )[0]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return [], []

    masks_raw = results.get("masks", [])
    scores_raw = results.get("scores", [])

    masks_np = []
    scores = []
    for m, s in zip(masks_raw, scores_raw):
        if torch.is_tensor(m):
            m = m.cpu().numpy()
        if torch.is_tensor(s):
            s = s.cpu().item()
        masks_np.append((m > 0.5).astype(np.uint8) * 255)
        scores.append(float(s))

    return masks_np, scores


# ---------------------------------------------------------------------------
# Contour extraction with hierarchy (RETR_CCOMP)
# ---------------------------------------------------------------------------

def _compute_area_quality(area_ratio: float) -> float:
    if 0.03 < area_ratio < 0.80:
        return 1.0
    if area_ratio <= 0.03:
        return area_ratio / 0.03
    return max(0.0, 1.0 - (area_ratio - 0.80) / 0.20)


def extract_shapes_with_hierarchy(
    mask_288: np.ndarray,
    image_name: str,
    target_label: str = "target",
    ignore_label: str = "ignore",
) -> List[Dict]:
    """Extract polygons with RETR_CCOMP hierarchy at 288p, scaled to 1024p.

    Parent contours → target_label, child (hole) contours → ignore_label.
    No approxPolyDP → raw contours from SAM3 mask are used.
    """
    if mask_288.ndim == 3:
        mask_288 = np.squeeze(mask_288)

    bin_uint8 = (mask_288.astype(np.float32) / 255.0 > 0.5).astype(np.uint8) * 255

    # MORPH_CLOSE(7×7) to merge nearby fragments
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (CLOSE_KERNEL_SIZE, CLOSE_KERNEL_SIZE)
    )
    merged = cv2.morphologyEx(bin_uint8, cv2.MORPH_CLOSE, kernel_close)

    # MORPH_OPEN(3×3) to remove noise
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (OPEN_KERNEL_SIZE, OPEN_KERNEL_SIZE)
    )
    cleaned = cv2.morphologyEx(merged, cv2.MORPH_OPEN, kernel_open)

    contours, hierarchy = cv2.findContours(
        cleaned, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )

    if hierarchy is None or len(contours) == 0:
        return []

    shapes: List[Dict] = []
    for i, contour in enumerate(contours):
        epsilon = EPSILON_FACTOR * cv2.arcLength(contour, True)
        simplified = cv2.approxPolyDP(contour, epsilon, True)

        scaled = (simplified.astype(np.float64) * SCALE).astype(np.int32)
        points = scaled.reshape(-1, 2).astype(float).tolist()

        if len(points) < 3:
            continue

        is_parent = hierarchy[0][i][3] == -1
        label = target_label if is_parent else ignore_label

        shapes.append({
            "label": label,
            "labels": [label],
            "shape_type": "polygon",
            "image_name": image_name,
            "points": points,
            "group_id": None,
            "group_ids": [None],
            "flags": {},
        })

    return shapes


# ---------------------------------------------------------------------------
# Combined scoring + shape extraction
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    shapes: List[Dict] = field(default_factory=list)
    best_score: float = 0.0
    area_ratio: float = 0.0
    total_score: float = 0.0
    num_valid_masks: int = 0
    num_target_shapes: int = 0
    num_ignore_shapes: int = 0


def process_sam3_output(
    masks: List[np.ndarray],
    scores: List[float],
    image_name: str,
) -> TaskResult:
    """Filter valid fragments, combine, extract hierarchy contours, score."""
    result = TaskResult()

    if not masks or not scores:
        return result

    paired = sorted(
        [(m, s) for m, s in zip(masks, scores)],
        key=lambda x: x[1],
        reverse=True,
    )

    result.best_score = paired[0][1]

    valid = [(m, s) for m, s in paired if s >= THRESHOLD_VALID_FRAGMENT]
    result.num_valid_masks = len(valid)

    if not valid:
        return result

    # Combine all valid masks into one binary mask (union)
    combined = np.zeros((SAM3_RES, SAM3_RES), dtype=np.uint8)
    for m, _ in valid:
        if m.ndim == 3:
            m = np.squeeze(m)
        combined = np.maximum(combined, (m > 0.5).astype(np.uint8) * 255)

    total_area_px = int(np.count_nonzero(combined))
    result.area_ratio = total_area_px / (SAM3_RES * SAM3_RES)
    area_quality = _compute_area_quality(result.area_ratio)

    result.total_score = 0.7 * result.best_score + 0.3 * area_quality

    # Extract contours with hierarchy
    result.shapes = extract_shapes_with_hierarchy(combined, image_name)

    result.num_target_shapes = sum(
        1 for s in result.shapes if s["label"] == "target"
    )
    result.num_ignore_shapes = sum(
        1 for s in result.shapes if s["label"] == "ignore"
    )

    return result


# ---------------------------------------------------------------------------
# Per-candidate data
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    image_path: Path
    image_name: str
    category: str
    instance_id: str
    safety: str
    prompt: str
    task: str
    result: TaskResult


# ---------------------------------------------------------------------------
# Walk filter images
# ---------------------------------------------------------------------------

def _walk_filter_images(filter_dir: Path) -> List[Tuple[str, Path]]:
    entries = []
    for safety in ["dangerous", "non-dangerous"]:
        safety_dir = filter_dir / safety
        if not safety_dir.is_dir():
            continue
        for img_path in sorted(safety_dir.iterdir()):
            if img_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
                entries.append((safety, img_path))
    return entries


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------

def _draw_shapes_vis(img: np.ndarray, shapes: List[dict], title: str = "") -> np.ndarray:
    vis = img.copy()
    overlay = np.zeros_like(img)

    for i, shape in enumerate(shapes):
        pts = np.array(shape["points"], np.int32).reshape((-1, 1, 2))
        label = shape.get("label", "target")
        color = (0, 0, 255) if label == "ignore" else (0, 255, 0)

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


def _generate_visualization(
    all_candidates: List[Candidate],
    filter_dir: Path,
    vis_dir: Path,
) -> None:
    """Pick 20 good/outlier/negative samples per category and save."""
    logger.info("Generating 20-sample visualization per category...")

    by_category: Dict[str, List[Candidate]] = defaultdict(list)
    for c in all_candidates:
        by_category[c.category].append(c)

    for cat, candidates in by_category.items():
        sorted_by_shapes = sorted(
            candidates,
            key=lambda c: c.result.num_target_shapes,
            reverse=True,
        )
        ones = [c for c in candidates if c.result.num_target_shapes == 1]
        negs = [c for c in candidates if c.result.num_target_shapes == 0]

        good = ones[:SAMPLES_PER_CATEGORY]
        outliers = sorted_by_shapes[:SAMPLES_PER_CATEGORY]
        negatives = negs[:SAMPLES_PER_CATEGORY]

        for cat_name, samples, label in [
            ("good", good, "good"),
            ("bad_outliers", outliers, "outlier"),
            ("bad_negative", negatives, "negative"),
        ]:
            out_dir = vis_dir / cat_name
            out_dir.mkdir(parents=True, exist_ok=True)

            for candidate in samples:
                src_path = candidate.image_path
                img = cv2.imread(str(src_path))
                if img is None:
                    logger.warning("Cannot load %s for visualization", src_path)
                    continue

                title = f"{candidate.image_name} | {candidate.task} | {candidate.result.num_target_shapes}S/{candidate.result.num_ignore_shapes}I | score={candidate.result.total_score:.3f}"
                vis = _draw_shapes_vis(img, candidate.result.shapes, title=title)
                out_path = out_dir / f"{Path(candidate.image_name).stem}.png"
                cv2.imwrite(str(out_path), vis)

        logger.info(
            "  %s: %d/%d/%d good/outlier/negative",
            cat, len(good), len(outliers), len(negatives),
        )

    logger.info("Visualization done — %s", vis_dir)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_segmentation_pipeline(
    filter_dir: Path,
    output_dir: Path,
) -> int:
    segmentation_dir = output_dir / "segmentation"
    segmentation_dir.mkdir(parents=True, exist_ok=True)

    # --- Phase A: collect all candidates from filter ---
    entries = _walk_filter_images(filter_dir)
    logger.info("Found %d images in %s", len(entries), filter_dir)

    all_candidates: List[Candidate] = []
    log_rows: List[Dict] = []
    total_processed = 0

    for safety, img_path in tqdm(entries, desc="SAM3 segmenting"):
        image_name = img_path.name
        try:
            category, instance_id = parse_image_name(image_name)
        except ValueError:
            logger.warning("Skipping unparseable filename: %s", image_name)
            continue

        safety_of_category = TAXONOMY.get(category)
        if safety_of_category is None:
            logger.warning("Unknown category '%s' in %s — skipped", category, image_name)
            continue

        if safety_of_category != safety:
            logger.warning(
                "Safety mismatch for %s: filter says '%s', taxonomy says '%s' — skipped",
                image_name, safety, safety_of_category,
            )
            continue

        tasks: List[str]
        if safety == "dangerous":
            tasks = ["handover", "pick_and_place"]
        else:
            tasks = ["handover", "pick_and_place"]

        prompts_used: List[str] = []
        task_results: List[TaskResult] = []

        for task in tasks:
            prompt = CATEGORY_PROMPT_MAP.get((category, task), "handle")
            prompts_used.append(prompt)

            try:
                img = Image.open(img_path).convert("RGB")
                masks, scores = segment_with_sam3(img, prompt)
                result = process_sam3_output(masks, scores, image_name)
            except Exception as e:
                logger.exception("Failed to segment %s (task=%s, prompt='%s')", image_name, task, prompt)
                result = TaskResult()

            task_results.append(result)

            candidate = Candidate(
                image_path=img_path,
                image_name=image_name,
                category=category,
                instance_id=instance_id,
                safety=safety,
                prompt=prompt,
                task=task,
                result=result,
            )
            all_candidates.append(candidate)

            log_rows.append({
                "timestamp": datetime.now().isoformat(),
                "image_name": image_name,
                "category": category,
                "instance_id": instance_id,
                "safety": safety,
                "task": task,
                "prompt": prompt,
                "best_score": f"{result.best_score:.4f}",
                "area_ratio": f"{result.area_ratio:.4f}",
                "total_score": f"{result.total_score:.4f}",
                "num_valid_masks": str(result.num_valid_masks),
                "num_target_shapes": str(result.num_target_shapes),
                "num_ignore_shapes": str(result.num_ignore_shapes),
            })

        total_processed += 1

        if total_processed % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # --- unload SAM3 before ranking ---
    unload_sam3()

    # --- Phase B: rank per (instance, task) and output best ---
    logger.info("Ranking %d candidates per (instance, task)...", len(all_candidates))

    cat_inst_map: Dict[Tuple[str, str, str], List[Candidate]] = defaultdict(list)
    for candidate in all_candidates:
        key = (candidate.category, candidate.instance_id, candidate.task)
        cat_inst_map[key].append(candidate)

    total_output = 0
    selected_log_rows: List[Dict] = []
    selected_candidates: List[Candidate] = []

    for (category, inst_id, task), candidates in cat_inst_map.items():
        best = max(candidates, key=lambda c: c.result.total_score)

        safety_out = best.safety
        json_name = f"{category}_{inst_id}.json"

        out_dir = segmentation_dir / task / safety_out
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / json_name

        lisa_json = {"shapes": best.result.shapes}

        with open(out_path, "w", encoding="utf-8") as f:
            json_mod.dump(lisa_json, f, indent=2, ensure_ascii=False)

        selected_candidates.append(best)

        selected_log_rows.append({
            "timestamp": datetime.now().isoformat(),
            "category": category,
            "instance_id": inst_id,
            "safety": safety_out,
            "task": task,
            "selected_image": best.image_name,
            "best_score": f"{best.result.best_score:.4f}",
            "area_ratio": f"{best.result.area_ratio:.4f}",
            "total_score": f"{best.result.total_score:.4f}",
            "num_target_shapes": str(best.result.num_target_shapes),
            "num_ignore_shapes": str(best.result.num_ignore_shapes),
            "dest_path": str(out_path),
        })
        total_output += 1

    # --- write CSV logs ---
    log_path = output_dir / "segmentation_log.csv"
    fieldnames = [
        "timestamp", "image_name", "category", "instance_id", "safety",
        "task", "prompt", "best_score", "area_ratio", "total_score",
        "num_valid_masks", "num_target_shapes", "num_ignore_shapes",
    ]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_rows)

    selected_log_path = output_dir / "segmentation_selected.csv"
    selected_fieldnames = [
        "timestamp", "category", "instance_id", "safety", "task",
        "selected_image", "best_score", "area_ratio", "total_score",
        "num_target_shapes", "num_ignore_shapes", "dest_path",
    ]
    with open(selected_log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=selected_fieldnames)
        writer.writeheader()
        writer.writerows(selected_log_rows)

    # --- summary ---
    num_candidates = len(all_candidates)
    num_images = len(entries)
    num_negative = sum(
        1 for c in all_candidates if c.result.num_target_shapes == 0
    )
    logger.info(
        "Segmentation complete — %d candidates from %d images, %d outputs, %d negatives "
        "(no target shapes) — CSV at %s",
        num_candidates, num_images, total_output, num_negative, log_path,
    )

    # --- Phase C: visualization (only selected outputs, not all candidates) ---
    vis_dir = segmentation_dir / "visualization"
    _generate_visualization(selected_candidates, filter_dir, vis_dir)

    return total_output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SAM3 affordance segmentation v4 — hierarchy-based contours + ranking",
    )
    parser.add_argument(
        "--filter_dir",
        default="../dataset/pipeline2.0/filter",
        help="Filter directory (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        default="../dataset/pipeline2.0",
        help="Output directory for pipeline2.0 (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help='Torch device (e.g. "cuda:0", "cpu")',
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> None:
    parsed = parse_args(args)
    logging.basicConfig(
        level=getattr(logging, parsed.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    global _device
    if parsed.device:
        _device = torch.device(parsed.device)

    logger.info(
        "Starting SAM3 segmentation v4 — filter=%s output=%s device=%s",
        parsed.filter_dir, parsed.output_dir,
        parsed.device or "auto",
    )

    count = run_segmentation_pipeline(
        filter_dir=Path(parsed.filter_dir),
        output_dir=Path(parsed.output_dir),
    )
    logger.info("Done — %d JSON outputs written to segmentation/", count)


if __name__ == "__main__":
    main()
