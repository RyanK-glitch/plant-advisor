"""Clean and split data/processed into train (70%), val (15%) and test (15%).

Pools every image of each class from the existing split folders, removes exact
duplicate images (same file bytes) so no image can appear in two splits, then
makes a stratified 70/15/15 split so every class keeps the same proportions.

The new split is built in a temporary folder and only replaces data/processed
once it is complete. Run from the project root:

    python scripts/data_prep/split_dataset.py --force
"""

import argparse
import csv
import hashlib
import random
import shutil
from pathlib import Path

SOURCE_DIR = Path("data/processed")
BUILD_DIR = Path("data/processed_build")
REPORT_DIR = Path("reports/tables")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42


def file_md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def collect_unique_images(class_name):
    # Sort first so the kept copy and the shuffle are the same on every run
    images = sorted(
        path
        for split in ("train", "val", "test")
        for path in (SOURCE_DIR / split / class_name).glob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )

    unique, seen = [], set()
    for path in images:
        digest = file_md5(path)
        if digest not in seen:
            seen.add(digest)
            unique.append((path, digest))
    return unique, len(images) - len(unique)


def split_counts(total):
    # Val and test get the rounded share; train takes the remainder
    n_val = round(total * RATIOS["val"])
    n_test = round(total * RATIOS["test"])
    return {"train": total - n_val - n_test, "val": n_val, "test": n_test}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="delete and rebuild an existing output folder")
    args = parser.parse_args()

    if not args.force:
        raise SystemExit(f"This replaces {SOURCE_DIR} with a new split. Re-run with --force to continue.")
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)

    classes = sorted(path.name for path in (SOURCE_DIR / "train").iterdir() if path.is_dir())
    rng = random.Random(SEED)
    manifest, summary = [], []

    for class_name in classes:
        unique, duplicates = collect_unique_images(class_name)
        rng.shuffle(unique)
        counts = split_counts(len(unique))

        start = 0
        for split, count in counts.items():
            target_dir = BUILD_DIR / split / class_name
            target_dir.mkdir(parents=True, exist_ok=True)

            for path, digest in unique[start:start + count]:
                target = target_dir / path.name
                # Different images can share a file name across the old train and val folders
                if target.exists():
                    target = target_dir / f"{path.parent.parent.name}_{path.name}"
                shutil.copy2(path, target)
                manifest.append([split, class_name, target.name, str(path), digest])
            start += count

        summary.append([class_name, len(unique) + duplicates, duplicates, len(unique),
                        counts["train"], counts["val"], counts["test"]])
        print(f"✅ {class_name:28} unique {len(unique):5}  ->  "
              f"train {counts['train']:4}  val {counts['val']:4}  test {counts['test']:4}"
              f"  (removed {duplicates} duplicates)")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_DIR / "split_manifest.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "class", "file", "source", "md5"])
        writer.writerows(manifest)

    with open(REPORT_DIR / "split_summary.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "original_files", "duplicates_removed", "unique",
                         "train", "val", "test"])
        writer.writerows(summary)
        totals = [sum(row[i] for row in summary) for i in range(1, 7)]
        writer.writerow(["TOTAL", *totals])

    # Swap in the new split only after every file has been copied
    shutil.rmtree(SOURCE_DIR)
    BUILD_DIR.rename(SOURCE_DIR)

    print(f"\n📁 Cleaned dataset written to {SOURCE_DIR}")
    print(f"📄 Per-image manifest: {REPORT_DIR / 'split_manifest.csv'}")
    print(f"📊 Summary table: {REPORT_DIR / 'split_summary.csv'}")
    print(f"   Total: {totals[0]} files, {totals[1]} duplicates removed, "
          f"train {totals[3]} / val {totals[4]} / test {totals[5]}")


if __name__ == "__main__":
    main()
