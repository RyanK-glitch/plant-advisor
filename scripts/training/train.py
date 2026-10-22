import torch
from pathlib import Path
from ultralytics import YOLO


def train_crop_model():
    # Select the best available hardware acceleration
    if torch.backends.mps.is_available():
        device = "mps"
        print("🚀 Using Apple Silicon MPS Acceleration")
    elif torch.cuda.is_available():
        device = 0
        print("🚀 Using NVIDIA CUDA GPU Acceleration")
    else:
        device = "cpu"
        print("⚠️ Using CPU. Training will be slow")

    # Load the pretrained YOLO11m small classification model
    print("📥 Loading YOLO11m classification model...")
    model = YOLO("models/yolo11m-cls.pt")

    # Start training the plant disease classification model
    print("🏋️ Starting model training...")

    results = model.train(
        data="data/processed",                    # Cleaned, deduplicated 70/15/15 split with train, val and test folders
        epochs=40,                         # Maximum training cycles; the previous 20-epoch run was still improving when it ended
        patience=10,                       # Stops training if validation accuracy does not improve for ten epochs
        imgsz=480,                         # Resizes images to 480 x 480 to preserve smaller disease lesions and rust pustules
        batch=16,                          # Processes 16 images before each weight update
        device=device,                     # Uses Apple MPS, NVIDIA CUDA or CPU according to hardware availability
        workers=4,                         # Uses four parallel workers to load and preprocess the images

        optimizer="AdamW",                 # Stable optimizer with decoupled weight decay for transfer learning
        lr0=0.0005,                        # Low initial learning rate for careful fine-tuning of pretrained weights
        lrf=0.1,                           # Final learning rate will be 10% of the initial learning rate
        weight_decay=0.0005,               # Penalizes excessively large weights and helps reduce overfitting
        warmup_epochs=2,                   # Gradually increases the learning rate during the first two epochs
        cos_lr=True,                       # Uses cosine learning-rate reduction for smoother convergence
        dropout=0.1,                       # Randomly disables 10% of classifier neurons to reduce overfitting

        degrees=15.0,                      # Randomly rotates images by up to 15 degrees
        translate=0.05,                    # Randomly shifts images by up to 5% horizontally or vertically
        scale=0.15,                        # Randomly zooms images in or out by up to 15%
        fliplr=0.5,                        # Applies horizontal flipping with a 50% probability
        flipud=0.5,                        # Applies vertical flipping with a 50% probability
        hsv_h=0.01,                        # Applies small hue changes while preserving disease colours
        hsv_s=0.30,                        # Changes colour saturation to simulate different camera conditions
        hsv_v=0.20,                        # Changes image brightness to simulate different lighting conditions
        auto_augment="randaugment",        # Applies automatic classification-specific image transformations
        erasing=0.10,                      # Randomly hides small areas to reduce reliance on one particular leaf region

        seed=42,                           # Uses the same random seed to make experiments more reproducible
        deterministic=True,                # Requests consistent training behaviour for dissertation comparison
        plots=True,                        # Generates accuracy, loss and confusion-matrix plots
        save=True,                         # Saves the best and final model weights
        project="crop_model_runs",         # Main directory used to store training experiments
        name="yolo11m_staples_480_split70",  # New run name so the earlier model is not overwritten
        exist_ok=True,                     # Allows the existing experiment directory to be reused
        verbose=True                       # Displays detailed training information in the terminal
    )

    # Get the location of the best-performing model
    best_model_path = Path(model.trainer.save_dir) / "weights" / "best.pt"

    print("✅ Training completed successfully!")
    print(f"✅ Training results saved to: {results.save_dir}")
    print(f"✅ Best model saved at: {best_model_path}")

    # Evaluate the best model once on the held-out test set, which training never saw
    print("🧪 Evaluating the best model on the test set...")
    test_model = YOLO(str(best_model_path))
    test_metrics = test_model.val(
        data="data/processed",
        split="test",
        imgsz=480,
        batch=16,
        device=device,
        plots=True,
        project="crop_model_runs",
        name="yolo11m_staples_480_split70_test",
        exist_ok=True,
    )
    print(f"✅ Test top-1 accuracy: {test_metrics.top1:.4f}")
    print(f"✅ Test top-5 accuracy: {test_metrics.top5:.4f}")


if __name__ == "__main__":
    train_crop_model()