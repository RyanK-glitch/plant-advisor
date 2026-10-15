import os
import shutil
import random

def filter_and_organize_dataset(target_root="data/processed", split_ratio=0.8):
    # 🧹 Clear out old dataset folder to prevent mixing with unwanted crops (like apple/tomato)
    if os.path.exists(target_root):
        print(f"🧹 Clearing old '{target_root}' directory...")
        shutil.rmtree(target_root)
        
    for split in ["train", "val"]:
        os.makedirs(os.path.join(target_root, split), exist_ok=True)

    valid_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

    # ==========================================
    # 1. PLANTVILLAGE CLASSES (Pre-split: Potato & Maize/Corn)
    # ==========================================
    plantvillage_classes = {
        "Potato": [
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy"
        ],
        "Maize": [
            "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight",
            "Corn_(maize)___healthy"
        ]
    }

    pv_source = "data/raw/PlantVillage"
    if os.path.exists(pv_source):
        print("\n🌿 Processing PlantVillage_Raw (Pre-split)...")
        for split in ["train", "val"]:
            src_split_dir = os.path.join(pv_source, split)
            dst_split_dir = os.path.join(target_root, split)
            
            if not os.path.exists(src_split_dir):
                continue
                
            for crop, classes in plantvillage_classes.items():
                for class_name in classes:
                    src_class_path = os.path.join(src_split_dir, class_name)
                    # Normalize folder name for YOLO classification
                    clean_name = class_name.lower().replace("___", "_").replace("(", "").replace(")", "").replace(" ", "_")
                    dst_class_path = os.path.join(dst_split_dir, clean_name)
                    
                    if os.path.exists(src_class_path):
                        os.makedirs(dst_class_path, exist_ok=True)
                        images = [f for f in os.listdir(src_class_path) if f.endswith(valid_extensions)]
                        for img in images:
                            shutil.copy(os.path.join(src_class_path, img), os.path.join(dst_class_path, img))
                        print(f"  ✅ [PV - {split}] {clean_name}: Copied {len(images)} images")
                    else:
                        print(f"  ⚠️ Warning: Folder '{class_name}' not found in PlantVillage {split} set.")

    # ==========================================
    # 2. INDIAN CROPS CLASSES (Raw folders: Rice & Wheat requiring 80/20 split)
    # ==========================================
    indian_crops_source = "data/raw/IndianCrops"
    if os.path.exists(indian_crops_source):
        print("\n🌾 Processing IndianCrops_Raw (Raw 80/20 split)...")
        
        # Scan subfolders under IndianCrops_Raw
        for crop_folder in os.listdir(indian_crops_source):
            crop_path = os.path.join(indian_crops_source, crop_folder)
            if not os.path.isdir(crop_path):
                continue
                
            # Filter strictly for Rice and Wheat (Potato/Maize handled by PlantVillage or included if present)
            crop_lower = crop_folder.lower()
            if not any(target in crop_lower for target in ["rice", "wheat", "potato", "corn", "maize"]):
                print(f"  ⏩ Skipping non-target folder: {crop_folder}")
                continue
                
            # Loop through disease subfolders within the crop folder
            disease_subfolders = [d for d in os.listdir(crop_path) if os.path.isdir(os.path.join(crop_path, d))]
            for disease in disease_subfolders:
                disease_src = os.path.join(crop_path, disease)
                clean_name = f"{crop_folder.lower()}_{disease.lower().replace(' ', '_')}"
                
                images = [f for f in os.listdir(disease_src) if f.endswith(valid_extensions)]
                if not images:
                    continue
                    
                random.shuffle(images)
                split_idx = int(len(images) * split_ratio)
                
                train_imgs = images[:split_idx]
                val_imgs = images[split_idx:]
                
                train_dst = os.path.join(target_root, "train", clean_name)
                val_dst = os.path.join(target_root, "val", clean_name)
                os.makedirs(train_dst, exist_ok=True)
                os.makedirs(val_dst, exist_ok=True)
                
                for img in train_imgs:
                    shutil.copy(os.path.join(disease_src, img), os.path.join(train_dst, img))
                for img in val_imgs:
                    shutil.copy(os.path.join(disease_src, img), os.path.join(val_dst, img))
                    
                print(f"  ✅ [IC - Split] '{clean_name}' -> {len(train_imgs)} train, {len(val_imgs)} val")

    print("\n🎉 Master 4-crop dataset filtering and organization complete!")

if __name__ == "__main__":
    filter_and_organize_dataset()
