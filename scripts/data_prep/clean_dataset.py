import time
import os
import pandas as pd
from cleanvision import Imagelab

if __name__ == '__main__':
    dataset_path = "data/processed/val"

    print("Initializing Imagelab for your dataset...")
    start_time = time.time()

    # Initialize and find issues
    imagelab = Imagelab(data_path=dataset_path)
    print("Finding issues...")
    imagelab.find_issues(n_jobs=1)

    elapsed = time.time() - start_time
    print(f"Analysis finished in {elapsed:.2f} seconds!")

    # 1. Save the full issues table to a CSV file
    csv_output_path = "reports/tables/dataset_issues_report.csv"
    imagelab.issues.to_csv(csv_output_path)
    print(f"Full report saved to {csv_output_path}")

    # 2. Automatically clean exact duplicates and blurry images
    df = imagelab.issues
    duplicates_to_drop = df[df['is_exact_duplicates_issue'] == True].index
    blurry_to_drop = df[df['is_blurry_issue'] == True].index

    images_to_remove = set(list(duplicates_to_drop) + list(blurry_to_drop))
    print(f"Total problematic images targeted for removal: {len(images_to_remove)}")

    deleted_count = 0
    for img_path in images_to_remove:
        if os.path.exists(img_path):
            os.remove(img_path)
            deleted_count += 1

    print(f"Cleanup complete! Successfully deleted {deleted_count} files.")
