import os
import matplotlib.pyplot as plt

# Training and validation dataset folders
train_dir = "data/processed/train"
val_dir = "data/processed/val"

# Folder where evaluation results will be saved
output_dir = "reports/figures"

# Supported image file formats
image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


def count_images(folder_path):
    # Store the number of images found for each class
    class_counts = {}

    # Check whether the dataset folder exists
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return class_counts

    # Go through each class folder
    for class_name in sorted(os.listdir(folder_path)):
        class_path = os.path.join(folder_path, class_name)

        # Ignore anything that is not a folder
        if not os.path.isdir(class_path):
            continue

        # Count only supported image files
        image_count = len([
            file_name
            for file_name in os.listdir(class_path)
            if file_name.lower().endswith(image_extensions)
        ])

        # Save the image count for this class
        class_counts[class_name] = image_count

    return class_counts


# Create the evaluation folder if it does not exist
os.makedirs(output_dir, exist_ok=True)

print("📊 Analyzing dataset distribution...")

# Count images in the training dataset
train_counts = count_images(train_dir)

# Count images in the validation dataset
val_counts = count_images(val_dir)

# Combine all class names from train and validation folders
classes = sorted(
    set(train_counts.keys()) |
    set(val_counts.keys())
)

# Stop if no classes were found
if not classes:
    print("❌ No dataset classes were found.")
    exit()

# Prepare image counts for plotting
train_values = [
    train_counts.get(class_name, 0)
    for class_name in classes
]

val_values = [
    val_counts.get(class_name, 0)
    for class_name in classes
]

# Calculate total number of images
total_train = sum(train_values)
total_val = sum(val_values)
total_images = total_train + total_val

# Display dataset statistics in the terminal
print("\nDataset Distribution")
print("-" * 75)

for class_name, train_count, val_count in zip(
    classes,
    train_values,
    val_values
):
    class_total = train_count + val_count

    print(
        f"{class_name:<35} "
        f"Train: {train_count:<6} "
        f"Val: {val_count:<6} "
        f"Total: {class_total}"
    )

print("-" * 75)

print(f"Total Training Images:   {total_train}")
print(f"Total Validation Images: {total_val}")
print(f"Total Dataset Images:    {total_images}")
print(f"Total Classes:           {len(classes)}")

# Calculate train and validation percentages
if total_images > 0:
    train_percentage = (total_train / total_images) * 100
    val_percentage = (total_val / total_images) * 100

    print(f"Training Percentage:     {train_percentage:.2f}%")
    print(f"Validation Percentage:   {val_percentage:.2f}%")

# Create positions for the grouped bar chart
x_positions = list(range(len(classes)))
bar_width = 0.4

# Create the dataset distribution chart
plt.figure(figsize=(15, 8))

# Plot training image counts
plt.bar(
    [
        x - bar_width / 2
        for x in x_positions
    ],
    train_values,
    width=bar_width,
    label="Training"
)

# Plot validation image counts
plt.bar(
    [
        x + bar_width / 2
        for x in x_positions
    ],
    val_values,
    width=bar_width,
    label="Validation"
)

# Add chart title and axis labels
plt.title(
    "Plant Disease Dataset Distribution",
    fontsize=16
)

plt.xlabel(
    "Disease Classes",
    fontsize=12
)

plt.ylabel(
    "Number of Images",
    fontsize=12
)

# Display class names on the x-axis
plt.xticks(
    x_positions,
    classes,
    rotation=45,
    ha="right"
)

# Display chart legend
plt.legend()

# Add horizontal grid lines for readability
plt.grid(
    axis="y",
    alpha=0.3
)

# Prevent labels from being cut off
plt.tight_layout()

# Save the chart inside the evaluation folder
output_path = os.path.join(
    output_dir,
    "dataset_distribution.png"
)

plt.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

# Display the chart
plt.show()

print(
    f"\n✅ Dataset distribution chart saved to "
    f"'{output_path}'."
)