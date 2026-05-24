"""
SigLIP quality filtering — threshold-based, no task split.

Pipeline 2.0 step: filter rendered images by SigLIP cosine similarity.

Usage:
    python -m scripts.dataset.filter_images \
        --source_dir ../dataset/render2.0 \
        --output_dir ../dataset/pipeline2.0 \
        --min_similarity 0.10
"""

import argparse
import csv
import gc
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoProcessor

# ---------------------------------------------------------------------------
# TAXONOMY: Objaverse category -> (safety_class, base_category)
# ---------------------------------------------------------------------------
TAXONOMY = {
    "scissors": ("dangerous", "scissors"),
    "shears": ("dangerous", "scissors"),
    "fork": ("dangerous", "fork"),
    "pitchfork": ("dangerous", "fork"),
    "knife": ("dangerous", "knife"),
    "steak_knife": ("dangerous", "knife"),
    "pocketknife": ("dangerous", "knife"),
    "kettle": ("non-dangerous", "kettle"),
    "teakettle": ("non-dangerous", "kettle"),
    "coffeepot": ("non-dangerous", "kettle"),
    "cup": ("non-dangerous", "cup"),
    "cappuccino": ("non-dangerous", "cup"),
    "mug": ("non-dangerous", "cup"),
    "teacup": ("non-dangerous", "cup"),
    "spoon": ("non-dangerous", "spoon"),
    "wooden_spoon": ("non-dangerous", "spoon"),
    "soupspoon": ("non-dangerous", "spoon"),
}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model cache
# ---------------------------------------------------------------------------
_siglip_model = None
_siglip_processor = None
_device = None


def get_device() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def load_siglip() -> Tuple[AutoModel, AutoProcessor]:
    global _siglip_model, _siglip_processor
    if _siglip_model is None:
        logger.info("Loading SigLIP model: google/siglip2-so400m-patch14-384 (fp16)")
        device = get_device()
        _siglip_model = AutoModel.from_pretrained(
            "google/siglip2-so400m-patch14-384",
            torch_dtype=torch.float16,
        ).to(device)
        _siglip_model.eval()
        _siglip_processor = AutoProcessor.from_pretrained(
            "google/siglip2-so400m-patch14-384"
        )
        logger.info("SigLIP model loaded in fp16")
    return _siglip_model, _siglip_processor


def unload_siglip() -> None:
    global _siglip_model, _siglip_processor
    if _siglip_model is not None:
        del _siglip_model
        _siglip_model = None
    if _siglip_processor is not None:
        del _siglip_processor
        _siglip_processor = None
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("SigLIP model unloaded, VRAM cleared")


# ---------------------------------------------------------------------------
# SigLIP scoring — threshold only, no top-k, no fallback
# ---------------------------------------------------------------------------

def _compute_text_embedding(text_prompt: str) -> np.ndarray:
    """Compute text embedding once, reused across all images in a category."""
    model, processor = load_siglip()
    device = get_device()
    text_inputs = processor(
        text=[text_prompt], padding="max_length", return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        text_out = model.text_model(**text_inputs)
    text_embed = F.normalize(text_out.pooler_output, p=2, dim=-1)
    return text_embed.cpu().numpy()


def _score_image_list(
    image_paths: List[Path],
    text_np: np.ndarray,
) -> List[Dict]:
    """Score multiple images against a precomputed text embedding.

    Returns list sorted by score descending.
    """
    model, processor = load_siglip()
    device = get_device()

    all_feats = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        img_in = processor(images=img, return_tensors="pt").to(device)
        with torch.no_grad():
            img_out = model.vision_model(**img_in)
            img_embed = F.normalize(img_out.pooler_output, p=2, dim=-1)
        all_feats.append(img_embed.cpu().numpy().squeeze(0))

    if not all_feats:
        return []

    all_feats_np = np.array(all_feats)
    scores = (all_feats_np @ text_np.T).squeeze(-1)

    results = [
        {"path": p, "score": float(s)}
        for p, s in zip(image_paths, scores)
    ]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Copy helper
# ---------------------------------------------------------------------------

def _copy_filtered(src: Path, dest_dir: Path, dest_name: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dst = dest_dir / dest_name
    shutil.copy2(str(src), str(dst))
    return dst


# ---------------------------------------------------------------------------
# Instance collection from render2.0
# ---------------------------------------------------------------------------

def _collect_instances(source_dir: Path, category: str) -> Dict[str, List[Path]]:
    cat_dir = source_dir / category
    if not cat_dir.is_dir():
        return {}

    instances = {}
    for child in sorted(cat_dir.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        images = sorted(child.glob("*.png")) + sorted(child.glob("*.jpg"))
        if images:
            instances[child.name] = images
    return instances


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_filter_pipeline(
    source_dir: Path,
    output_dir: Path,
    categories: Optional[List[str]] = None,
    min_similarity: float = 0.10,
) -> int:
    filter_dir = output_dir / "filter"
    filter_dir.mkdir(parents=True, exist_ok=True)

    if categories is None:
        categories = list(TAXONOMY.keys())

    total_copied = 0
    log_entries: List[Dict] = []

    for raw_category in categories:
        info = TAXONOMY.get(raw_category)
        if info is None:
            logger.warning("Unknown category '%s' — skipped", raw_category)
            continue
        safety_class, base_category = info
        safety_dir = "dangerous" if safety_class == "dangerous" else "non-dangerous"

        instances = _collect_instances(source_dir, raw_category)
        if not instances:
            logger.warning("No rendered images found for '%s'", raw_category)
            continue

        prompt = f"a {base_category}"
        logger.info(
            "Processing '%s' (%s) — %d instances, prompt='%s'",
            raw_category, safety_class, len(instances), prompt,
        )

        # Compute text embedding once per category
        text_np = _compute_text_embedding(prompt)

        for instance_id, image_paths in instances.items():
            if not image_paths:
                continue

            results = _score_image_list(image_paths, text_np)

            for r in results:
                frame = Path(r["path"]).name
                score = r["score"]
                passed = score >= min_similarity

                if passed:
                    dest_name = f"{raw_category}_{instance_id}_{frame}"
                    dest_path = _copy_filtered(
                        r["path"],
                        filter_dir / safety_dir,
                        dest_name,
                    )
                else:
                    dest_path = Path("")

                log_entries.append({
                    "timestamp": datetime.now().isoformat(),
                    "category": raw_category,
                    "instance_id": instance_id,
                    "safety": safety_dir,
                    "frame": frame,
                    "score": f"{score:.6f}",
                    "threshold": min_similarity,
                    "passed": str(passed),
                    "source_path": str(r["path"]),
                    "dest_path": str(dest_path),
                })
                if passed:
                    total_copied += 1

    # --- write CSV log ---
    log_path = output_dir / "filter_log.csv"
    fieldnames = [
        "timestamp", "category", "instance_id", "safety", "frame",
        "score", "threshold", "passed", "source_path", "dest_path",
    ]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(log_entries)

    # --- summary ---
    passed_count = sum(1 for e in log_entries if e["passed"] == "True")
    failed_count = len(log_entries) - passed_count
    logger.info(
        "Filter complete — %d passed, %d below threshold (%.1f%% pass rate) — CSV at %s",
        passed_count, failed_count,
        (passed_count / len(log_entries) * 100) if log_entries else 0,
        log_path,
    )

    # --- unload SigLIP ---
    unload_siglip()

    return total_copied


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SigLIP quality filter for pipeline 2.0 — threshold-based, no task split",
    )
    parser.add_argument(
        "--source_dir",
        default="../dataset/render2.0",
        help="Root directory of rendered images (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir",
        default="../dataset/pipeline2.0",
        help="Output directory for pipeline2.0 (default: %(default)s)",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Specific categories to process (default: all in TAXONOMY)",
    )
    parser.add_argument(
        "--min_similarity",
        type=float,
        default=0.10,
        help="Minimum SigLIP cosine similarity threshold (default: %(default)s)",
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

    logger.info(
        "Starting filter v2 — source=%s output=%s threshold=%.2f categories=%s",
        parsed.source_dir, parsed.output_dir, parsed.min_similarity,
        parsed.categories or "ALL",
    )
    count = run_filter_pipeline(
        source_dir=Path(parsed.source_dir),
        output_dir=Path(parsed.output_dir),
        categories=parsed.categories,
        min_similarity=parsed.min_similarity,
    )
    logger.info("Done — %d images copied to filter/", count)


if __name__ == "__main__":
    main()
