from pathlib import Path
import shutil
import random

SOURCE = Path("data/processed")
OUTPUT = Path("data/processed_pruned")

TRAIN_CAP = 1200
VAL_CAP = 300

SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp"
}

random.seed(SEED)


def prune_split(split_name, cap):

    source_split = SOURCE / split_name
    output_split = OUTPUT / split_name

    print(f"\n--- {split_name.upper()} ---")

    for class_folder in sorted(source_split.iterdir()):

        if not class_folder.is_dir():
            continue

        class_name = class_folder.name

        images = [
            f for f in class_folder.rglob("*")
            if f.is_file()
            and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        original_count = len(images)

        random.shuffle(images)

        # Keep all images if class is already below the cap
        if original_count > cap:
            selected = images[:cap]
        else:
            selected = images

        destination = output_split / class_name

        destination.mkdir(
            parents=True,
            exist_ok=True
        )

        for image in selected:

            shutil.copy2(
                image,
                destination / image.name
            )

        print(
            f"{class_name:30} "
            f"{original_count:5} -> "
            f"{len(selected):5}"
        )


# Remove old output if rerunning
if OUTPUT.exists():
    shutil.rmtree(OUTPUT)

prune_split("train", TRAIN_CAP)
prune_split("val", VAL_CAP)

print("\n------------------------------")
print("PRUNING COMPLETE")
print("------------------------------")

print(
    f"New dataset created at:\n"
    f"{OUTPUT.resolve()}"
)