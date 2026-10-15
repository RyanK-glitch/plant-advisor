import os
import shutil

# Dataset folders
dataset_splits = [
    "data/processed/train",
    "data/processed/val"
]

# Map duplicate folders into the final 14 classes
class_mapping = {
    "corn_corn___common_rust": "maize_common_rust",
    "corn_maize_common_rust": "maize_common_rust",

    "corn_corn___gray_leaf_spot": "maize_gray_leaf_spot",
    "corn_maize_gray_leaf_spot": "maize_gray_leaf_spot",

    "corn_corn___northern_leaf_blight": "maize_northern_leaf_blight",
    "corn_maize_northern_leaf_blight": "maize_northern_leaf_blight",

    "corn_corn___healthy": "maize_healthy",
    "corn_maize_healthy": "maize_healthy",

    "potato_potato___early_blight": "potato_early_blight",
    "potato_potato___late_blight": "potato_late_blight",
    "potato_potato___healthy": "potato_healthy",

    "rice_rice___brown_spot": "rice_brown_spot",
    "rice_rice___leaf_blast": "rice_leaf_blast",
    "rice_rice___neck_blast": "rice_neck_blast",
    "rice_rice___healthy": "rice_healthy",

    "wheat_wheat___brown_rust": "wheat_brown_rust",
    "wheat_wheat___yellow_rust": "wheat_yellow_rust",
    "wheat_wheat___healthy": "wheat_healthy"
}


def merge_classes(split_path):
    # Merge duplicate folders into the correct class
    for old_class, new_class in class_mapping.items():
        old_path = os.path.join(split_path, old_class)
        new_path = os.path.join(split_path, new_class)

        # Skip folders that do not exist
        if not os.path.exists(old_path):
            continue

        # Create the final class folder
        os.makedirs(new_path, exist_ok=True)

        print(f"🔄 {old_class} -> {new_class}")

        # Move images into the final class folder
        for file_name in os.listdir(old_path):
            source = os.path.join(old_path, file_name)

            if not os.path.isfile(source):
                continue

            destination = os.path.join(new_path, file_name)

            # Prevent duplicate filenames from being overwritten
            if os.path.exists(destination):
                name, ext = os.path.splitext(file_name)
                counter = 1

                while os.path.exists(destination):
                    destination = os.path.join(
                        new_path,
                        f"{name}_{counter}{ext}"
                    )
                    counter += 1

            shutil.move(source, destination)

        # Remove old duplicate folder
        shutil.rmtree(old_path)


def show_classes(split_path):
    # Display final dataset classes
    classes = sorted([
        name for name in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, name))
    ])

    print(f"\n📁 {split_path}")

    for class_name in classes:
        print(f" - {class_name}")

    print(f"Total classes: {len(classes)}")


print("🌿 Fixing dataset classes...")

# Process train and validation datasets
for split_path in dataset_splits:
    if os.path.exists(split_path):
        merge_classes(split_path)
    else:
        print(f"❌ Folder not found: {split_path}")

# Show final classes
for split_path in dataset_splits:
    if os.path.exists(split_path):
        show_classes(split_path)

print("\n✅ Finished.")
