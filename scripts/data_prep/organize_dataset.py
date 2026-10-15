import os
import shutil
import random

def organize_raw_datasets():
    output_base = "data/processed"
    valid_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")
    
    # Create train/val base directories
    for split in ["train", "val"]:
        os.makedirs(os.path.join(output_base, split), exist_ok=True)

    # 1. Process PlantVillage_Raw (Already split into train/val)
    pv_raw = "data/raw/PlantVillage"
    if os.path.exists(pv_raw):
        for split in ["train", "val"]:
            split_path = os.path.join(pv_raw, split)
            if os.path.exists(split_path):
                for class_folder in os.listdir(split_path):
                    class_src = os.path.join(split_path, class_folder)
                    if os.path.isdir(class_src):
                        clean_name = class_folder.lower().replace("___", "_").replace(" ", "_")
                        dst_dir = os.path.join(output_base, split, clean_name)
                        os.makedirs(dst_dir, exist_ok=True)
                        
                        images = [f for f in os.listdir(class_src) if f.endswith(valid_extensions)]
                        for img in images:
                            shutil.copy(os.path.join(class_src, img), os.path.join(dst_dir, img))
                        print(f"✅ PlantVillage [{split}]: Copied {len(images)} images for '{clean_name}'")

    # 2. Process IndianCrops_Raw (Raw folders requiring 80/20 train/val split)
    ic_raw = "data/raw/IndianCrops"
    if os.path.exists(ic_raw):
        for crop_folder in os.listdir(ic_raw):
            crop_path = os.path.join(ic_raw, crop_folder)
            if os.path.isdir(crop_path):
                subfolders = [d for d in os.listdir(crop_path) if os.path.isdir(os.path.join(crop_path, d))]
                
                if subfolders:
                    for disease_folder in subfolders:
                        disease_src = os.path.join(crop_path, disease_folder)
                        clean_name = f"{crop_folder.lower()}_{disease_folder.lower().replace(' ', '_')}"
                        
                        images = [f for f in os.listdir(disease_src) if f.endswith(valid_extensions)]
                        if not images:
                            continue
                            
                        random.shuffle(images)
                        split_idx = int(len(images) * 0.8)
                        
                        train_imgs = images[:split_idx]
                        val_imgs = images[split_idx:]
                        
                        train_dst = os.path.join(output_base, "train", clean_name)
                        val_dst = os.path.join(output_base, "val", clean_name)
                        os.makedirs(train_dst, exist_ok=True)
                        os.makedirs(val_dst, exist_ok=True)
                        
                        for img in train_imgs:
                            shutil.copy(os.path.join(disease_src, img), os.path.join(train_dst, img))
                        for img in val_imgs:
                            shutil.copy(os.path.join(disease_src, img), os.path.join(val_dst, img))
                            
                        print(f"✅ IndianCrops [Split]: '{clean_name}' -> {len(train_imgs)} train, {len(val_imgs)} val")

    print("\n🎉 Dataset consolidation and formatting complete!")

if __name__ == "__main__":
    organize_raw_datasets()
