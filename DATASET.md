# Robotic Affordance Dataset — Pipeline 2.0

Dataset untuk fine-tune LISA (Large Language Instructed Segmentation Assistant) pada robotic affordance understanding. Input berupa image + natural language instruction, output berupa segmentation mask affordance region + text explanation.

---

## 1. Taxonomy (18 categories)

| Safety Class | Categories | Count |
|-------------|-----------|:-----:|
| **Dangerous** | fork, scissors, shears, knife, pocketknife, steak_knife, pitchfork | 7 |
| **Non-Dangerous** | spoon, soupspoon, wooden_spoon, teakettle, cup, teacup, mug, kettle, cappuccino, coffeepot | 11 |

### Affordance Rules

| Object Type | Task Type | Affordance Region | SAM3 Prompt |
|------------|-----------|-------------------|-------------|
| Dangerous | handover | Non-handle / sharp edge / blade | `"blade"` |
| Dangerous | pick_and_place | Handle / safe grip | `"handle"` |
| Non-dangerous | handover | Handle / safe grip | `"handle"` |
| Non-dangerous | pick_and_place | Handle / safe grip | `"handle"` |

---

## 2. Pipeline Overview

```
render/objaverse/          # Engine rendering (Blender)
  ├── objaverse_download_script.py
  ├── texture_filter.py
  └── blender_script.py

scripts/dataset/           # Pipeline aktif
  ├── filter_images.py       # SigLIP filtering
  ├── segment_affordance.py  # SAM3 segmentation
  └── visualize_segmentation.py  # Visualisasi output
```

### Alur Pipeline

```
Download .glb ──► Filter Texture ──► Blender Render ──► SigLIP Filter ──► SAM3 Segment
(objaverse)       (texture_filter)   (16 views/obj)    (threshold 0.10)   (threshold 0.5)
```

### Pipeline Detail

| Step | Script | Input | Output |
|------|--------|-------|--------|
| 1. Download | `render/objaverse/objaverse_download_script.py` | Objaverse LVIS | `dataset/obj_models/{category}/*.glb` |
| 2. Filter Texture | `render/objaverse/texture_filter.py` (Blender) | `.glb` files | Filtered `.glb` (only textured) |
| 3. Render | `render/objaverse/blender_script.py` (Blender) | `.glb` files | `dataset/render2.0/{category}/{id}/{frame:03d}.png` |
| 4. SigLIP Filter | `scripts/dataset/filter_images.py` | `render2.0/` | `dataset/pipeline2.0/filter/{safety}/` (5048 images) |
| 5. SAM3 Segment | `scripts/dataset/segment_affordance.py` | `filter/` | `dataset/pipeline2.0/segmentation/{task}/{safety}/` (690 JSONs) |
| 6. Visualize | `scripts/dataset/visualize_segmentation.py` | segmentation CSVs | `visualization/{good,bad_outliers,bad_negative}/` |

---

## 3. Folder Structure

```
/media/sirund/gacor/Kuliah/Tugas Akhir/
├── dataset/
│   ├── obj_models/               # Downloaded .glb files (per category)
│   ├── render2.0/                # Blender renderings (16 frames/instance)
│   │   ├── {category}/
│   │   │   ├── {instance}/
│   │   │   │   ├── 000.png           # RGB rendering
│   │   │   │   ├── 000.npy           # Camera RT matrix
│   │   │   │   ├── ...
│   │   │   │   ├── 015.png
│   │   │   │   ├── cam_K.npy         # Camera intrinsic matrix
│   │   │   │   └── metadata.json     # Mesh metadata
│   │   │   └── ...
│   │   └── ...
│   └── pipeline2.0/               # Pipeline 2.0 output
│       ├── filter/                # SigLIP-filtered images (5048)
│       │   ├── dangerous/         # (2451 images)
│       │   └── non-dangerous/     # (2597 images)
│       ├── segmentation/          # SAM3 JSON output (690)
│       │   ├── handover/
│       │   │   ├── dangerous/     # (276 files)
│       │   │   └── non-dangerous/ # (69 files)
│       │   ├── pick_and_place/
│       │   │   ├── dangerous/     # (276 files)
│       │   │   └── non-dangerous/ # (69 files)
│       │   └── visualization/     # Overlay images
│       │       ├── good/          # 1 target shape
│       │       ├── bad_outliers/  # Most shapes
│       │       └── bad_negative/  # 0 shapes (negatives)
│       ├── filter_log.csv         # SigLIP results (5728 rows)
│       ├── segmentation_log.csv   # All candidates (10096 rows)
│       └── segmentation_selected.csv  # Best per instance-task (690 rows)
├── render/
│   └── objaverse/                 # ❗ Rendering engine
│       ├── blender_script.py
│       ├── objaverse_download_script.py
│       ├── texture_filter.py
│       ├── used_categories.py      # 18 categories
│       ├── all_categories.py       # All LVIS (reference)
│       └── adjustments.md          # Blender 4→5 migration notes
└── tugas-akhir/
    ├── DATASET.md
    ├── AGENTS.md
    └── scripts/
        ├── dataset/                # ❗ Active pipeline
        │   ├── filter_images.py
        │   ├── segment_affordance.py
        │   └── visualize_segmentation.py
        └── train/                  # ❗ Training/eval (future)
            └── __init__.py
```

---

## 4. JSON Output Format

```json
{
  "shapes": [
    {
      "label": "target",
      "group_id": 0,
      "description": "",
      "shape_type": "polygon",
      "flags": {},
      "points": [[x1, y1], [x2, y2], ...],
      "labels": ["target"]
    }
  ]
}
```

- `label: "target"` = affordance region (parent contour via RETR_CCOMP)
- `label: "ignore"` = hole/child contour (future use for LISA ignore region)
- Points: closed polygon, `approxPolyDP(epsilon=0.002)`, scaled to 1024p

---

## 5. Usage

### Rendering (from `/media/sirund/gacor/Kuliah/Tugas Akhir/render/objaverse/`)

```bash
# 1. Download 50 models per category
python objaverse_download_script.py --data_root ../../dataset --n 50

# 2. Filter out models without textures
blender --background \
    --python texture_filter.py -- \
    --data_root ../../dataset

# 3. Render 16 views per model
blender --background \
    --python blender_script.py -- \
    --data_root ../../dataset \
    --engine BLENDER_EEVEE \
    --num_renders 16 \
    --only_northern_hemisphere
```

### Filter + Segment (from `/media/sirund/gacor/Kuliah/Tugas Akhir/tugas-akhir/`)

```bash
# SigLIP filter (threshold 0.10)
python -m scripts.dataset.filter_images \
    --source_dir ../dataset/render2.0 \
    --output_dir ../dataset/pipeline2.0 \
    --min_similarity 0.10

# SAM3 segmentation
python -m scripts.dataset.segment_affordance \
    --filter_dir ../dataset/pipeline2.0/filter \
    --output_dir ../dataset/pipeline2.0

# Visualize selected outputs
python -m scripts.dataset.visualize_segmentation \
    --filter_dir ../dataset/pipeline2.0/filter \
    --seg_dir ../dataset/pipeline2.0/segmentation \
    --selected_csv ../dataset/pipeline2.0/segmentation_selected.csv \
    --output_dir ../dataset/pipeline2.0/segmentation/visualization \
    --samples 20
```

### Filter with specific categories

```bash
python -m scripts.dataset.filter_images \
    --source_dir ../dataset/render2.0 \
    --output_dir ../dataset/pipeline2.0 \
    --categories fork knife mug \
    --min_similarity 0.10
```

---

## 6. Current Status (Pipeline 2.0)

| Metric | Value |
|--------|:-----:|
| Source images (render2.0) | 5728 |
| Passed SigLIP filter | 5048 (88.4%) |
| Segmentation outputs | 690 JSONs |
| Clean (1 target shape) | 578 (83.8%) |
| Oversegmented (2+ shapes) | 81 (11.7%) |
| Negatives (0 shapes) | 31 (4.5%) |

---

## 7. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Two-pass** (filter all → segment all) | Avoid 5728× model load/unload overhead (~16-48h) |
| **SigLIP threshold 0.10** | Balances pass rate vs quality; 88.4% pass |
| **SAM3 threshold 0.5, no cascade** | High precision; 4.5% negatives acceptable |
| **RETR_CCOMP** (not RETR_EXTERNAL) | Parent=target, child=ignore (required by LISA) |
| **MORPH_CLOSE 7×7** | Merges nearby fragments before contour extraction |
| **Score = 0.7×best_mask + 0.3×area** | Ranks candidates per (instance, task) |
| **Inline TAXONOMY** | No external config dependency, each script is self-contained |
