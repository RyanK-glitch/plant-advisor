# 🌿 Plant Disease Safe-RAG Advisory System (India)

An AI-assisted crop disease advisor for Indian farmers. A farmer uploads a photo of a leaf; the system identifies the disease, checks treatment advice against verified Indian regulatory records (PPQS / CIBRC), and explains what to do in plain language, in English or one of ten Indian languages.

> Final-year Computer Science dissertation project by **Ryan K**.

**Live app:** <https://plant-advisor.victoriousbeach-0c1589c9.southeastasia.azurecontainerapps.io>
**Docker image:** [`ryankglitch/plant-advisor`](https://hub.docker.com/r/ryankglitch/plant-advisor) (linux/amd64 and linux/arm64)

---

## Contents

1. [Motivation](#motivation)
2. [How it works](#how-it-works)
3. [Features](#features)
4. [Supported crops and diseases](#supported-crops-and-diseases)
5. [Model and results](#model-and-results)
6. [Project structure](#project-structure)
7. [Installation](#installation)
8. [Running the app](#running-the-app)
9. [Running with Docker](#running-with-docker)
10. [Deploying to Azure](#deploying-to-azure)
11. [Configuration and maintenance](#configuration-and-maintenance)
12. [Re-training and evaluation](#re-training-and-evaluation)
13. [Safety design](#safety-design)
14. [Limitations](#limitations)
15. [Technology stack](#technology-stack)
16. [Acknowledgements and data sources](#acknowledgements-and-data-sources)
17. [Disclaimer](#disclaimer)

---

## Motivation

General-purpose chatbots can give confident but wrong pesticide advice: invented products, incorrect doses, or chemicals not registered for that crop in India. For a farmer, following that advice can damage the crop, harm health, or leave unsafe residues on food.

This project tests a **"Safe-RAG"** approach. The language model is never trusted to decide *what* chemical or dose to recommend. Those values come only from a verified table built from official PPQS / CIBRC sources and are enforced by a deterministic rule engine. The language model's role is limited to *explaining* the verified evidence clearly.

## How it works

```
 Leaf photo
     │
     ▼
┌───────────────────────┐
│ 1. YOLO11m classifier │  → crop + disease label, confidence score
└───────────────────────┘
     │
     ▼
┌───────────────────────┐
│ 2. RAG retrieval      │  → matching evidence from the knowledge base
│    (ChromaDB + BGE)   │     (advisory PDFs + verified treatment tables)
└───────────────────────┘
     │
     ▼
┌───────────────────────┐
│ 3. Rule engine        │  → APPROVED / MANUAL CHECK / REJECTED
│    (deterministic)    │     with exact chemical, dose, PHI, re-entry
└───────────────────────┘
     │
     ▼
┌───────────────────────┐
│ 4. Gemini LLM         │  → farmer-friendly report, grounded only in
│    (with fallback)    │     steps 2 and 3, in the chosen language
└───────────────────────┘
```

1. **Image classification.** A YOLO11m classification model, fine-tuned on 14 crop-disease classes, predicts the disease and a confidence score.
2. **Retrieval.** The prediction is used to query a ChromaDB vector store. Records tagged with the exact disease key are fetched first, then supporting semantic matches; results are filtered so only evidence for the detected crop and disease is kept.
3. **Rule engine.** `src/rule_engine.py` looks up the disease in a hard-coded table of audited PPQS / CIBRC rules and returns one of three statuses:
   - **APPROVED**: a verified chemical, dose, timing, maximum applications, pre-harvest interval (PHI) and re-entry period, with the source URL. Healthy classes are also approved, with *no chemical advised*.
   - **WARNING (manual check)**: a partial match where the source does not exactly cover the disease; no automatic dose is given.
   - **REJECTED**: no verified rule exists.
4. **Report generation.** Gemini writes an action-focused report using only the retrieved evidence and the rule output. If no evidence was retrieved, no report is generated.

## Features

- 📷 **Leaf image diagnosis** with a visual confidence meter (high / moderate / low).
- 🛡️ **Regulatory safety card** showing status, chemical, dose, timing, PHI, re-entry period and a link to the official source.
- 📚 **Transparent evidence**: the retrieved RAG passages can be expanded and inspected.
- 🌐 **Multilingual reports** in English, Hindi, Tamil, Telugu, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi and Odia. Chemical names, doses, numbers and URLs are kept exactly as verified, so they still match the product label.
- 🔊 **Read-aloud reports** in the selected language (all except Odia).
- 🌾 **Crop selector**: the farmer can state the crop, which limits the diagnosis to that crop's diseases and blocks advice when the photo looks like a different crop.
- 🌦️ **Weather-based disease risk** *(prototype)*: an optional location gives a 4-day Low / Moderate / High risk outlook and the next suitable spray window, from the free Open-Meteo forecast.
- ✅ **Field feedback**: farmers can confirm or correct each diagnosis, building a dataset for retraining.
- 🔁 **Automatic model fallback**: if a Gemini model is busy, rate-limited or retired, the app immediately tries the next free-tier model and shows which one produced the report.
- 🎨 **Custom HTML / CSS interface** on top of Streamlit, with a responsive layout for phones.

## Supported crops and diseases

| Crop   | Classes                                                    |
|--------|------------------------------------------------------------|
| Maize  | Common rust, Gray leaf spot, Northern leaf blight, Healthy |
| Potato | Early blight, Late blight, Healthy                         |
| Rice   | Brown spot, Leaf blast, Neck blast, Healthy                |
| Wheat  | Brown rust, Yellow rust, Healthy                           |

**14 classes** in total.

## Model and results

### Dataset

15,216 source images were pooled, **3,335 exact duplicates were removed** so that no image can appear in two splits, and the remaining 11,881 unique images were given a stratified 70 / 15 / 15 split (seed 42).

| Split      | Images |
|------------|-------:|
| Train      | 8,323  |
| Validation | 1,779  |
| Test       | 1,779  |

Per-class counts are in `reports/tables/split_summary.csv`.

### Training

| Setting        | Value                                             |
|----------------|---------------------------------------------------|
| Base model     | `yolo11m-cls.pt` (Ultralytics, pretrained)        |
| Image size     | 480 × 480                                         |
| Epochs         | 40 (early-stopping patience 10)                   |
| Batch size     | 16                                                |
| Optimiser      | AdamW, lr₀ = 0.0005, 2 warm-up epochs, cosine LR  |
| Regularisation | Dropout 0.1, weight decay 0.0005                  |
| Augmentation   | RandAugment, rotation, flips, HSV, erasing        |
| Hardware       | Apple Silicon (MPS)                               |
| Seed           | 42 (deterministic)                                |

Best validation top-1 accuracy was **97.58 %** (top-5: 100 %) at epoch 40.

### Results on the held-out test set

The test split was not used for training or model selection.

| Metric             | Value       |
|--------------------|-------------|
| Accuracy           | **97.13 %** |
| Macro F1           | 0.968       |
| Weighted F1        | 0.971       |
| Misclassified      | 51 / 1,779  |

Most errors are between the three rice classes, which are the weakest:

| Class            | Precision | Recall | F1    |
|------------------|----------:|-------:|------:|
| Rice brown spot  | 0.878     | 0.859  | 0.868 |
| Rice leaf blast  | 0.879     | 0.885  | 0.882 |
| Rice healthy     | 0.911     | 0.917  | 0.914 |

Four classes score a perfect 1.000 F1 (maize common rust, maize healthy, rice neck blast and wheat healthy) and every other class is above 0.94. The full per-class report, confusion matrix and list of misclassified images are in `reports/evaluation_yolo11m_staples_480_split70/`; training curves are in `reports/figures/`.

## Project structure

Folders marked *(local only)* are listed in `.gitignore` and are not in the repository.

```
.
├── .devcontainer/
│   └── devcontainer.json          # VS Code / Codespaces dev container
├── .streamlit/
│   ├── config.toml                # Streamlit theme
│   └── secrets.toml               # GEMINI_API_KEY (local only)
├── assets/
│   └── style.css                  # Custom styling for the interface
├── chroma_db/                     # Persisted vector store, built from knowledge_base/
├── data/                          # (local only)
│   ├── raw/                       # Source datasets (PlantVillage, Jain 2026)
│   ├── processed/                 # Final 14-class dataset
│   │   ├── train/                 #   70 %
│   │   ├── val/                   #   15 %
│   │   └── test/                  #   15 %
│   ├── feedback/                  # Farmer feedback database and consented photos
│   └── feedback_dataset/          # Feedback photos exported for retraining
├── knowledge_base/
│   ├── ADV-001 … ADV-014_*.pdf    # One advisory per class (14 PDFs)
│   ├── Crop_Disease_RAG_Advisory_India.xlsx
│   └── India_PPQS_CIBRC_Verified_Treatment_Table.xlsx
├── models/
│   └── yolo11m-cls.pt             # Pretrained base model used for fine-tuning
├── reports/
│   ├── evaluation_yolo11m_staples_480_split70/
│   │   ├── classification_report.csv
│   │   ├── confusion_matrix.csv
│   │   ├── confusion_matrix.png
│   │   ├── misclassified.csv
│   │   └── YOLO11m_Test_Evaluation.xlsx
│   ├── figures/                   # Dataset distribution and training curves
│   └── tables/                    # Class counts, split manifest, dataset issues, feedback summary
├── runs/classify/crop_model_runs/
│   └── yolo11m_staples_480_split70/
│       └── weights/
│           └── best.pt            # Trained model loaded by the app
├── scripts/                       # Offline pipeline, run from the project folder
│   ├── analysis/
│   │   ├── audit_dataset.py       # Class distribution chart
│   │   ├── count_classes.py       # Images per class and split
│   │   └── export_feedback.py     # Field accuracy and retraining photos from feedback
│   ├── data_prep/
│   │   ├── check_images.py        # Find unreadable images
│   │   ├── clean_dataset.py       # CleanVision quality checks
│   │   ├── filter_dataset.py      # Keep the four target crops
│   │   ├── fix_dataset_classes.py # Merge and rename class folders
│   │   ├── organize_dataset.py    # Copy raw datasets into data/processed
│   │   ├── prune_dataset.py       # Optional: cap images per class
│   │   └── split_dataset.py       # Remove duplicates, stratified 70/15/15 split
│   ├── evaluation/
│   │   ├── evaluate_model.py      # Test-set metrics and confusion matrix
│   │   ├── plot_training_history.py
│   │   └── visualize_dataset.py
│   └── training/
│       └── train.py               # Fine-tune YOLO11m on data/processed
├── src/                           # Application modules
│   ├── feedback.py                # Stores "Was this correct?" answers (SQLite)
│   ├── input_guard.py             # Crop selector check
│   ├── labels.py                  # Shared label parsing
│   ├── rag_pipeline.py            # Knowledge-base ingestion and retrieval (ChromaDB)
│   ├── read_aloud.py              # Reads the report aloud with gTTS
│   ├── rule_engine.py             # Deterministic PPQS / CIBRC safety rules
│   ├── ui_components.py           # HTML components (hero, cards, sidebar, footer)
│   └── weather_risk.py            # Open-Meteo forecast, disease risk and spray window
├── .dockerignore
├── .env                           # GEMINI_API_KEY for Docker (local only)
├── .env.example                   # Template for .env
├── .gitignore
├── Dockerfile                     # Container image for the app
├── README.md
├── app.py                         # Streamlit entry point: diagnosis → RAG → rules → Gemini report
├── docker-compose.yml             # Local container run with a persistent feedback volume
└── requirements.txt
```

## Installation

**Requirements:** Python 3.9 or newer (tested on 3.11 and 3.12) and about 5 GB of free disk space for the dependencies, models and embeddings.

```bash
# 1. Clone the repository
git clone https://github.com/RyanK-glitch/plant-advisor.git
cd plant-advisor

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**4. Add a Gemini API key.** Get a free key from [Google AI Studio](https://aistudio.google.com/apikey) and add it to `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-key-here"
```

Alternatively, set it as an environment variable: `export GEMINI_API_KEY="your-key-here"`. Both `secrets.toml` and `.env` are already in `.gitignore`; never commit a real key.

**5. Check the trained model is present** at `runs/classify/crop_model_runs/yolo11m_staples_480_split70/weights/best.pt`. To use a model stored elsewhere, set:

```bash
export YOLO_MODEL_PATH="/full/path/to/best.pt"
```

The app will not start without a trained model; it deliberately refuses to fall back to the untrained base model.

> **Moved or copied the project folder?** A virtual environment stores its own absolute path, so a copied `.venv` breaks with `bad interpreter: no such file or directory`. Delete `.venv` and repeat steps 2–3, or start the app with `python -m streamlit run app.py`.

## Running the app

```bash
streamlit run app.py
```

Then open the URL shown in the terminal (usually <http://localhost:8501>):

1. Choose a report language.
2. Optionally choose the crop and enter a location for the weather risk.
3. Upload a clear photo of a single leaf (JPG or PNG).
4. Click **🔍 Run Safe-RAG Analysis**.
5. Review the diagnosis, the regulatory safety check, the retrieved evidence and the advisory report.
6. Optionally press **🔊 Read the report aloud** or answer **Was this correct?**

On first launch, the BGE embedding model (`BAAI/bge-large-en-v1.5`, about 1.3 GB) is downloaded from Hugging Face, which can take a few minutes.

## Running with Docker

The image uses CPU-only PyTorch, runs as a non-root user, includes a health check, and has the embedding model built in, so it starts without downloading anything. The Gemini key is passed in at runtime and is never stored in the image.

**Option A: pull the published image**

```bash
docker run -d -p 8501:8501 -e GEMINI_API_KEY=your-key-here ryankglitch/plant-advisor:v3
```

**Option B: build from source with Docker Compose**

```bash
cp .env.example .env              # then put your real key in .env
docker compose up -d --build      # app at http://localhost:8501
docker compose logs -f app        # follow the logs
docker compose down               # stop (feedback volume is kept)
```

Compose stores farmer feedback in a named volume (`feedback`), so it survives rebuilds. If port 8501 is taken, run with `HOST_PORT=8502 docker compose up -d`.

**Publishing a multi-platform image** (Intel/AMD servers and Apple Silicon):

```bash
docker buildx build --platform linux/amd64,linux/arm64 \
  -t ryankglitch/plant-advisor:v3 --push .
```

## Deploying to Azure

The live app runs on **Azure Container Apps** from the Docker Hub image.

| Setting          | Value                                                       |
|------------------|-------------------------------------------------------------|
| Resource group   | `plant-rg` (Southeast Asia)                                 |
| Environment      | `plant-env`                                                 |
| Container app    | `plant-advisor`, 2 vCPU / 4 GiB, port 8501, 0–1 replicas    |
| Image            | `docker.io/ryankglitch/plant-advisor:v3`                    |
| Feedback storage | Azure Files share `feedback` in storage account `plantadvisorfeedback`, mounted at `/app/data/feedback` |

Because feedback is stored on Azure Files, it survives restarts, scale-to-zero and new revisions. Browse it in the portal under **Storage accounts → plantadvisorfeedback → File shares → feedback**; `feedback.db` appears after the first submission.

To deploy a new image version:

```bash
az containerapp update -n plant-advisor -g plant-rg \
  --image docker.io/ryankglitch/plant-advisor:<tag>
```

With a minimum of 0 replicas the app shuts down when idle, so the first visit after a quiet period takes longer while the models load.

## Configuration and maintenance

### Environment variables

| Variable          | Purpose                                            | Default                                    |
|-------------------|----------------------------------------------------|--------------------------------------------|
| `GEMINI_API_KEY`  | Google Gemini key for report generation (required) | read from `.streamlit/secrets.toml`        |
| `YOLO_MODEL_PATH` | Path to the trained classifier                     | `runs/…/yolo11m_staples_480_split70/weights/best.pt` |
| `FEEDBACK_DIR`    | Where the feedback database and photos are stored  | `data/feedback`                            |

### Rebuilding the knowledge base

Run this whenever files in `knowledge_base/` change:

```bash
python src/rag_pipeline.py
```

It deletes the old `chroma_db/`, loads every PDF, CSV and XLSX file, splits them into chunks of 900 characters (120 overlap) and stores them in a fresh vector database. Each spreadsheet row is stored as its own record with its `crop_key`, `disease_key`, `regulatory_status` and `source_url`, so exact matches can be retrieved by disease. Rebuild the Docker image afterwards, since `chroma_db/` is copied into it.

### Weather risk

Type a village, town or district under **Location** and pick the match. The app fetches an hourly forecast from [Open-Meteo](https://open-meteo.com/) (free, no API key) and shows:

- **Disease risk** for the next 4 days, from hours of high humidity or rain within each disease's favourable temperature range (`DISEASE_WEATHER` in `src/weather_risk.py`). Potato late blight uses the Hutton criteria, with a mean-temperature ceiling of 25 °C added for Indian conditions.
- **Next spray window** (only when a chemical is advised): daytime, wind under 15 km/h, below 35 °C and no rain for the following 6 hours.

The weather only affects risk and timing advice. It never changes the chemical, dose or rule status. The thresholds are indicative and should be checked against ICAR guidance.

### Read-aloud reports

Audio comes from Google Text-to-Speech (gTTS) in `src/read_aloud.py`. It needs an internet connection and covers every report language except Odia. Markdown and web addresses are removed before the text is spoken.

### Field feedback

After each diagnosis the farmer can answer **Yes**, **No** (and pick the real disease) or **Not sure**, add a comment and choose to share the photo. Answers are stored in `feedback.db` (SQLite) inside `FEEDBACK_DIR`. Shared photos are resized to 1024 px and saved without EXIF metadata, so phone GPS coordinates are never kept.

```bash
# Field accuracy per class        -> reports/tables/feedback_summary.csv
# Shared photos with a known class -> data/feedback_dataset/<class>/ (review before training)
python scripts/analysis/export_feedback.py
```

### Changing the Gemini models

The fallback order is defined by `GEMINI_MODELS` near the top of `app.py`. The app moves to the next model on errors 404, 408, 429, 500, 502, 503 and 504, or on a 60-second timeout. It stops immediately on other errors (such as an invalid API key), since those would affect every model.

## Re-training and evaluation

Run the steps in this order from the project folder. The raw datasets must first be placed in `data/raw/`.

```bash
# 1. Build the dataset (filter_dataset.py clears data/processed first)
python scripts/data_prep/filter_dataset.py
python scripts/data_prep/organize_dataset.py
python scripts/data_prep/fix_dataset_classes.py

# 2. Check and clean image quality
python scripts/data_prep/check_images.py
python scripts/data_prep/clean_dataset.py

# 3. Remove duplicates and make the 70/15/15 train/val/test split
python scripts/data_prep/split_dataset.py
#    Optional: cap images per class into data/processed_pruned
python scripts/data_prep/prune_dataset.py

# 4. Inspect the dataset
python scripts/analysis/count_classes.py
python scripts/analysis/audit_dataset.py

# 5. Train (automatically uses Apple MPS, NVIDIA CUDA or CPU)
python scripts/training/train.py

# 6. Evaluate on the held-out test split and plot results into reports/
python scripts/evaluation/evaluate_model.py
python scripts/evaluation/plot_training_history.py
python scripts/evaluation/visualize_dataset.py
```

The evaluation and data-cleaning scripts also use `matplotlib`, `seaborn`, `scikit-learn` and `cleanvision`; install them with `pip install matplotlib seaborn scikit-learn cleanvision` if needed.

## Safety design

Key decisions that keep the advice safe:

- **The LLM never chooses a chemical or dose.** These come only from `src/rule_engine.py`, whose rules were audited against PPQS / CIBRC sources on 22 September 2026.
- **Healthy predictions never trigger pesticide advice.**
- **Partial matches give no automatic dose** and are flagged for manual label checking.
- **No evidence, no report.** If retrieval finds nothing for the detected disease, the report is not generated.
- **Values are protected in translation.** Chemical names, doses, units, numbers and URLs stay exactly as verified in every language.
- **The image result is presented as an initial assessment.** Farmers are told to confirm in the field and to follow the registered product label.

## Limitations

- **Test images are not field photos.** The 97.13 % test accuracy comes from the same sources as the training data. Real-world accuracy on farmers' phone photos is likely to be lower, especially with cluttered backgrounds, multiple leaves or poor lighting. The field feedback feature is designed to measure this.
- **Rice classes are the weakest** (F1 0.87–0.91), mainly confusion between brown spot, leaf blast and healthy leaves.
- **Dataset bias.** Many images come from PlantVillage, which was photographed under controlled conditions.
- **Limited scope.** Only 4 crops and 14 classes are supported. Any other crop or disease will be forced into one of these classes.
- **Weather risk is not a validated forecast.** The thresholds come from general infection criteria and have not been tested against Indian outbreak records.
- **Rules can go out of date.** Pesticide registrations change; the rule table must be re-audited periodically.
- **Depends on external APIs.** Report generation needs internet access and the Gemini free tier, which has rate limits and changing model availability.
- **Translation quality** has not been formally evaluated by native speakers for every language.

## Technology stack

| Area             | Tools                                                  |
|------------------|--------------------------------------------------------|
| Image model      | Ultralytics YOLO11m-cls, PyTorch                       |
| Retrieval        | LangChain, ChromaDB, BAAI/bge-large-en-v1.5 embeddings |
| Document loading | PyPDF, pandas, openpyxl                                |
| Language model   | Google Gemini (`google-genai` SDK)                     |
| Speech           | gTTS                                                   |
| Interface        | Streamlit, custom HTML / CSS                           |
| Data quality     | CleanVision, scikit-learn, matplotlib, seaborn         |
| Deployment       | Docker, Docker Hub, Azure Container Apps, Azure Files  |

## Acknowledgements and data sources

- **PlantVillage** dataset: maize and potato images.
- **Jain (2026)** dataset: additional Indian crop images. *(Add full citation.)*
- **Directorate of Plant Protection, Quarantine & Storage (PPQS)** and the **Central Insecticides Board & Registration Committee (CIBRC)**, Government of India: treatment and registration records.
- **ICAR** guidance on integrated pest management.
- [Ultralytics](https://github.com/ultralytics/ultralytics), [LangChain](https://www.langchain.com/), [ChromaDB](https://www.trychroma.com/), [Streamlit](https://streamlit.io/), [Google Gemini](https://ai.google.dev/) and [Open-Meteo](https://open-meteo.com/).

*Please replace the placeholder citations above with the full references used in the dissertation, including the source of the wheat images.*

## Disclaimer

This software is an academic prototype built for a final-year dissertation. It is **not** a substitute for advice from a qualified agronomist or a local agricultural extension officer. Always confirm a diagnosis in the field and read the registered product label before applying any chemical.
