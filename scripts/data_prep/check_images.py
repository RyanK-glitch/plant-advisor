import time
from cleanvision import Imagelab

if __name__ == '__main__':
    dataset_path = "data/processed/val"

    print("Initializing Imagelab for your dataset...")
    start_time = time.time()

    # Initialize imagelab with your dataset
    imagelab = Imagelab(data_path=dataset_path)

    print("Finding issues...")
    imagelab.find_issues(n_jobs=1)

    elapsed = time.time() - start_time
    print(f"Finished in {elapsed:.2f} seconds!")

    # 1. Print the interactive/detailed report summary in terminal
    print("\n--- Generating Full Report ---")
    imagelab.report()

    # 2. Save the full issues table to a CSV file to view it in Excel or a spreadsheet app
    csv_output_path = "reports/tables/dataset_issues_report.csv"
    imagelab.issues.to_csv(csv_output_path)
    print(f"\nFull report successfully saved to {csv_output_path}!")
