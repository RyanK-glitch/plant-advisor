"""Run the classifier over an image folder and cache raw logits, 1280-d features and labels.

Every reliability script (calibration, abstention, OOD, field test) reads these .npz caches,
so the slow model pass happens once per image set.

Folder layouts:
    labelled   - one sub-folder per class, e.g. data/processed/test/maize_common_rust/*.jpg
    unlabelled - any images anywhere below the folder (use --ood); labels are stored as -1

Usage (from the project folder, with the venv active):
    python scripts/evaluation/extract_outputs.py --images data/processed/val   --name val
    python scripts/evaluation/extract_outputs.py --images data/processed/test  --name test
    python scripts/evaluation/extract_outputs.py --images data/processed/train --name train
    python scripts/evaluation/extract_outputs.py --images path/to/other_crops --name ood_other_crops --ood
    python scripts/evaluation/extract_outputs.py --images path/to/plantdoc --name field_plantdoc \
        --mapping scripts/evaluation/config/plantdoc_mapping.json
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
from src.labels import parse_prediction  # noqa: E402
from src.reliability import LogitCapture, softmax  # noqa: E402

DEFAULT_WEIGHTS = PROJECT_DIR / "runs/classify/crop_model_runs/yolo11m_staples_480_split70/weights/best.pt"
CACHE_DIR = PROJECT_DIR / "reports/reliability/cache"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def skipped_folder(name, skip_folders):
    return any(name == s or name.startswith(s + "_") for s in skip_folders)


def collect(folder, ood, mapping, per_class, seed, exclude=(), skip_folders=()):
    folder = Path(folder)
    items = []
    if ood:
        groups = {}
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES and not any(x in path.name for x in exclude) \
                    and not skipped_folder(path.parent.name, skip_folders):
                groups.setdefault(path.parent, []).append(path)
        for group, paths in groups.items():
            items += [(p, None, group.name) for p in paths]
    else:
        for class_dir in sorted(p for p in folder.iterdir() if p.is_dir() and not skipped_folder(p.name, skip_folders)):
            if mapping is not None:
                target = mapping.get(class_dir.name)
                if not target:
                    print(f"  skipped folder (not in mapping): {class_dir.name}")
                    continue
            else:
                target = parse_prediction(class_dir.name)[1]
            paths = sorted(p for p in class_dir.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES
                           and not any(x in p.name for x in exclude))
            items += [(p, target, class_dir.name) for p in paths]

    if per_class:
        random.Random(seed).shuffle(items)
        counts, kept = {}, []
        for item in items:
            key = item[2]
            if counts.get(key, 0) < per_class:
                counts[key] = counts.get(key, 0) + 1
                kept.append(item)
        items = sorted(kept, key=lambda item: str(item[0]))
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--name", required=True, help="cache name, e.g. val, test, train, ood_other_crops")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--ood", action="store_true", help="unlabelled images (no class folders needed)")
    parser.add_argument("--mapping", help="JSON {folder name: canonical class key} for external datasets")
    parser.add_argument("--per-class", type=int, default=0, help="random sample per folder (0 = all)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="cpu, mps or 0 (default: Ultralytics picks)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", default=str(CACHE_DIR), help="where the .npz cache is written")
    parser.add_argument("--exclude", default="", help="comma list: skip files whose name contains any of these")
    parser.add_argument("--skip-folders", default="", help="comma list: skip folders named X or starting with X_")
    args = parser.parse_args()

    mapping = json.loads(Path(args.mapping).read_text()) if args.mapping else None
    split = lambda v: [x for x in v.split(",") if x]
    items = collect(args.images, args.ood, mapping, args.per_class, args.seed,
                    exclude=split(args.exclude), skip_folders=split(args.skip_folders))
    if not items:
        sys.exit("No images found.")

    model = YOLO(args.weights)
    names = [model.names[i] for i in range(len(model.names))]
    keys = [parse_prediction(name)[1] for name in names]
    imgsz = model.overrides.get("imgsz", 480)
    capture = LogitCapture(model, imgsz=imgsz, device=args.device)

    labels = []
    for _, target, _ in items:
        if target is None:
            labels.append(-1)
        elif target in keys:
            labels.append(keys.index(target))
        else:
            sys.exit(f"Class '{target}' is not one of the model's classes: {keys}")

    all_logits, all_feats, max_gap = [], [], 0.0
    paths = [str(p) for p, _, _ in items]
    print(f"{args.name}: {len(paths)} images, imgsz={imgsz}")
    for start in range(0, len(paths), args.batch):
        results = model.predict(paths[start:start + args.batch], imgsz=imgsz, verbose=False,
                                device=args.device)
        logits, feats = capture.pop()
        # Sanity check: softmax of the captured logits must equal Ultralytics' own probabilities
        ul_probs = np.stack([r.probs.data.float().cpu().numpy() for r in results])
        max_gap = max(max_gap, float(np.abs(softmax(logits) - ul_probs).max()))
        all_logits.append(logits)
        all_feats.append(feats)
        print(f"  {min(start + args.batch, len(paths))}/{len(paths)}", end="\r", flush=True)
    print()
    if max_gap > 1e-3:
        print(f"WARNING: captured logits differ from Ultralytics probabilities (max gap {max_gap:.4f})")

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{args.name}.npz"
    np.savez_compressed(
        out,
        logits=np.concatenate(all_logits).astype(np.float32),
        features=np.concatenate(all_feats).astype(np.float16),
        labels=np.array(labels, dtype=np.int64),
        paths=np.array(paths),
        groups=np.array([g for _, _, g in items]),
        class_keys=np.array(keys),
        weights=np.array(str(args.weights)),
    )
    print(f"Saved {out}  (logit check max gap {max_gap:.2e})")


if __name__ == "__main__":
    main()
