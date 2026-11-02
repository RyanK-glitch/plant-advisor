#!/usr/bin/env bash
# Full re-evaluation for the field-fine-tuned model, plus old-vs-new comparison.
#
# Run from the project folder with the virtual environment active:
#     cd ~/Desktop/"Plant-disease_advisor_india copy"
#     source .venv/bin/activate
#     bash scripts/evaluation/evaluate_field_model.sh
#
# Writes:
#     reports/reliability_field_ft/      calibration, abstention, unknown-input and field-test results (new model)
#     reports/model_comparison/          old vs new: accuracy, macro-F1, per-class F1, confusion matrices
#     models/reliability.json + models/ood_feature_bank.npy   now calibrated for the NEW model
#     models/reliability_staples.json + models/ood_feature_bank_staples.npy   backup of the old model's settings
# The old model's results in reports/reliability/ are kept.
set -euo pipefail
cd "$(dirname "$0")/../.."

NEW="runs/classify/crop_model_runs/yolo11m_field_ft/weights/best.pt"
OLD="runs/classify/crop_model_runs/yolo11m_staples_480_split70/weights/best.pt"
FT="../plant-field-training"
C="reports/reliability_field_ft/cache"
OUT="reports/reliability_field_ft"
DEV=$(python -c "import torch; print('mps' if torch.backends.mps.is_available() else 'cpu')")
EXTRACT="python scripts/evaluation/extract_outputs.py --weights $NEW --cache-dir $C --device $DEV"

for p in "$NEW" "$OLD" "$FT/dataset/val" "$FT/test_field" "$FT/dataset/train" "$FT/train_images"; do
  [ -e "$p" ] || { echo "Missing: $p"; exit 1; }
done

echo "== 1/6  New model: run over every image set (device: $DEV)"
$EXTRACT --images data/processed/val   --name val_lab
$EXTRACT --images "$FT/dataset/val"    --name val_field  --exclude lab__,__dup
$EXTRACT --images data/processed/test  --name test_lab
$EXTRACT --images "$FT/test_field"     --name test_field
$EXTRACT --images data/processed/train --name train_lab   --per-class 120
$EXTRACT --images "$FT/dataset/train"  --name train_field --exclude lab__,__dup --per-class 150
# Unknown inputs: other crops (lab style) and rice problems the model was never trained on (field style)
$EXTRACT --images data/raw/PlantVillage/train --name ood_plantvillage_other_crops --ood --per-class 30 --skip-folders Corn,Potato
$EXTRACT --images "$FT/train_images" --name ood_unseen_rice_problems --ood --per-class 30 --skip-folders blast,brown_spot,normal

echo "== 2/6  Old model on the new field test set (for the comparison)"
python scripts/evaluation/extract_outputs.py --weights "$OLD" --cache-dir reports/reliability/cache \
    --device "$DEV" --images "$FT/test_field" --name test_field

echo "== 3/6  Back up the old model's reliability settings"
[ -f models/reliability_staples.json ] || cp models/reliability.json models/reliability_staples.json
[ -f models/ood_feature_bank_staples.npy ] || cp models/ood_feature_bank.npy models/ood_feature_bank_staples.npy

echo "== 4/6  Calibration and abstention (fit on lab + field val, report on lab + field test)"
python scripts/evaluation/calibration_abstention.py --cache-dir $C --out-dir $OUT \
    --val val_lab,val_field --test test_lab,test_field --model "$NEW" --target-risk 0.01

echo "== 5/6  Unknown-input detection and the full gate"
python scripts/evaluation/ood_detection.py --cache-dir $C --out-dir $OUT \
    --train train_lab,train_field --val val_lab,val_field --test test_lab,test_field --model "$NEW"
python scripts/evaluation/field_test.py --cache-dir $C --out-dir $OUT --field test_field --in-domain test_lab --skip-dedup

echo "== 6/6  Old vs new comparison"
python scripts/evaluation/compare_models.py

echo
echo "Done. The app now uses the new model WITH the reliability checks switched on."
"
