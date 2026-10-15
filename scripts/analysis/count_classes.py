from pathlib import Path
import csv

# Dataset location
DATASET = Path("data/processed")

# Supported image formats
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Find dataset folders
splits = {
    "Train": DATASET / "train",
    "Val": DATASET / "val",
    "Test": DATASET / "test"
}

# Support alternative validation folder names
if not splits["Val"].exists():
    for name in ("valid", "validation"):
        if (DATASET / name).exists():
            splits["Val"] = DATASET / name
            break

# Collect all class names
classes = set()

for folder in splits.values():
    if folder.exists():
        classes.update(
            p.name for p in folder.iterdir()
            if p.is_dir()
        )

if not classes:
    raise SystemExit("No class folders found inside dataset/")

# Count images for each class
results = []

for name in sorted(classes):
    row = {"Class": name}

    for split, folder in splits.items():
        if not folder.exists():
            row[split] = "NOT CREATED"
        else:
            class_folder = folder / name
            row[split] = sum(
                1 for p in class_folder.rglob("*")
                if p.is_file() and p.suffix.lower() in EXTENSIONS
            ) if class_folder.exists() else 0

    results.append(row)

# Display class distribution
print("\nCLASS DISTRIBUTION")
print(f"{'Class':<48} {'Train':>10} {'Val':>10} {'Test':>14}")
print("-" * 86)

for row in results:
    print(
        f"{row['Class']:<48} "
        f"{str(row['Train']):>10} "
        f"{str(row['Val']):>10} "
        f"{str(row['Test']):>14}"
    )

print("-" * 86)

# Print dataset totals
for split, folder in splits.items():
    if folder.exists():
        total = sum(
            row[split] for row in results
            if isinstance(row[split], int)
        )
        print(f"{split} total: {total}")
    else:
        print(f"{split} dataset: NOT CREATED")

print(f"Total classes: {len(results)}")

# Export results to CSV
with open("reports/tables/class_distribution.csv", "w", newline="") as file:
    writer = csv.DictWriter(
        file, fieldnames=["Class", "Train", "Val", "Test"]
    )
    writer.writeheader()
    writer.writerows(results)

print("\nSaved: reports/tables/class_distribution.csv")
