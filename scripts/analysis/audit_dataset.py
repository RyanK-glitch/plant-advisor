import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def audit_dataset():
    base_dir = "data/processed"
    splits = ["train", "val"]
    
    data = []
    
    print("🔍 Scanning filtered dataset directories...")
    for split in splits:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            print(f"⚠️ Warning: Directory '{split_dir}' not found.")
            continue
            
        classes = sorted(os.listdir(split_dir))
        for cls in classes:
            cls_path = os.path.join(split_dir, cls)
            if os.path.isdir(cls_path):
                # Count valid image extensions
                images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                count = len(images)
                data.append({
                    "Split": split.capitalize(),
                    "Class": cls,
                    "Count": count
                })
                
    if not data:
        print("❌ No dataset structure found. Please check your path.")
        return
        
    df = pd.DataFrame(data)
    
    # Print clean text table summary
    print("\n📊 ========================================")
    print("         DATASET AUDIT SUMMARY          ")
    print("========================================")
    print(df.to_string(index=False))
    print("----------------------------------------")
    print(f"Total Images in Training: {df[df['Split'] == 'Train']['Count'].sum()}")
    print(f"Total Images in Validation: {df[df['Split'] == 'Val']['Count'].sum()}")
    print(f"Grand Total Dataset Size: {df['Count'].sum()}")
    print("========================================\n")
    
    # Generate publication-quality chart for dissertation
    print("📈 Generating high-resolution distribution chart...")
    plt.figure(figsize=(14, 7))
    sns.set_theme(style="whitegrid")
    
    ax = sns.barplot(
        data=df, 
        x="Class", 
        y="Count", 
        hue="Split", 
        palette="viridis"
    )
    
    plt.title("Filtered Dataset Class Distribution (Train vs Validation)", fontsize=14, fontweight='bold', pad=15)
    plt.xlabel("Plant Disease Class", fontsize=12, labelpad=10)
    plt.ylabel("Number of Images", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.legend(title="Dataset Split")
    plt.tight_layout()
    
    # Save as high-DPI image for dissertation report
    output_filename = "reports/figures/dataset_audit_distribution.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"💾 Success! Saved chart as '{output_filename}' in your project root folder.")
    
    plt.show()

if __name__ == "__main__":
    audit_dataset()
