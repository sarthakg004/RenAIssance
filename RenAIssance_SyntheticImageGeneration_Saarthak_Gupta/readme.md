# Synthetic text generation with Generative Models

For Historical Manuscripts, limited availability of data a limiting factor in training OCR or Layout Detection models. This project propose a solution to this bottleneck by using a GAN architecture to generate synthetic renaissance style Image Data which can be used to improve OCR model performance on historical Spanish texts. This project is part of the HumanAI Foundation initiative and was developed during Google Summer of Code 2025.

## Table of Contents

- Overview  
- Project Structure  
- First-time Setup (Windows)  
- Data and Weights  
- Data Generation Pipeline  
- Model Building & Training  
- Inference & Synthetic Page Generation  
- Using the Notebooks  
---

## Overview

Historical OCR is challenged by scarce data and variable degradation in old manuscripts. This project:

1. Builds a data‑generation pipeline to split, enhance, detect, and align word images from scanned PDFs and transcripts.
2. Implements a Pix2Pix‑style WGAN (U‑Net generator + PatchGAN Critic) to translate clean synthetic text into realistic Renaissance‑era handwriting.
3. Provides a procedural synthetic page generator that simulates paper aging, ink bleed, and layout irregularities.
4. Build an CNN-biLSTM OCR model using the synthetic data and finetune on original data.
5. Used the synthetic data to finetune CRAFT text detection model.

---

## Project Structure

```
RenAIssance_SyntheticImageGeneration_Saarthak_Gupta/
├─ experimentation.ipynb            # End-to-end data → model → inference walkthrough
├─ OCR.ipynb                        # OCR baseline/evaluation workflow (CRAFT + OCR)
├─ finetune-craft-custom.ipynb      # Fine-tune CRAFT text detector
├─ CRAFT-pytorch/                   # CRAFT detector code and weights
│  └─ weights/
├─ fonts/                           # Historical-style fonts (e.g., RomanAntique.ttf)
├─ models/                          # Pretrained weights (e.g., RealESRGAN_x4plus.pth)
├─ data/                            # You will place your books and transcripts here
│  └─ 1_raw/
│     ├─ books/                     # Input PDFs
│     └─ transcripts/               # Input DOCX transcripts
├─ src/
│  ├─ data_generation.py            # End-to-end data pipeline (hard-coded default paths)
│  ├─ data_utils.py                 # Splitting, preprocessing, detection, mapping, dataset building
│  ├─ model_utils.py                # GAN models, train loop, visualization, inference pipeline
│  └─ model.py                      # Training + inference pipeline runner
└─ assets/                          # Figures used in this README
```

Notes on paths:
- This README and notebooks use a direct `data/...` layout (e.g., `data/1_raw/...`). Some scripts (e.g., `src/data_generation.py`, `src/model.py`) 

---

## First-time Setup (Windows)

Prerequisites:
- Python 3.8+ (3.10 recommended)
- Git
- GPU optional (CUDA speeds training but is not required)

1) Create and activate a virtual environment

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

2) Install core dependencies

```bat
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
pip install pillow numpy pandas matplotlib opencv-python scikit-image tqdm python-docx pymupdf basicsr realesrgan pytesseract
```
or 
```
pip install -r requirements
```

3) Install Tesseract OCR (required for mapping words)
- Download and install Tesseract for Windows (default path: `C:\Program Files\Tesseract-OCR\tesseract.exe`).
- The code already points to this path in `src/data_utils.py`. If you installed elsewhere, update the line:
  `pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"`.


---

## Data and Weights

- Download data and required model files (PDFs, transcripts, optional prepared outputs) from:   [Share Link](https://iitbhu365-my.sharepoint.com/:f:/g/personal/saarthak_gupta_mec22_iitbhu365_onmicrosoft_com/Eg2oNtFNhwVInbgs-FOrrHoBomkr7nfbY9VYHgx8eWjPcQ?e=qmazE5)

Place all the downloaded directory in your working repository.

### Required folders to download from Google Drive

Download the following folders from the shared drive and place them directly inside `RenAIssance_SyntheticImageGeneration_Saarthak_Gupta/`:

1) `data/`
  - Contains the pipeline structure used by `src/data_generation.py` and the notebooks.
  - Expected subfolders after running the pipeline: `2_splitted`, `3_processed`, `4_bounding_boxes`, `5_mapped`, `6_word_data`, `final_dataset`, `grid_dataset`.

2) `OCR_data_original/`
  - Original OCR pages and related artifacts used by `OCR.ipynb` for baseline comparisons.
  - Example: `OCR_data_original/1_raw/...`

3) `OCR_data_synthetic/`
  - Synthetic OCR data produced by the pipeline/notebooks for model training and evaluation.
  - Example: `OCR_data_synthetic/3_preprocessed/...`

4) `synthetic_finetune_data/`
  - Data (bboxes/pages/transcripts) to fine‑tune CRAFT in `finetune-craft-custom.ipynb`.
  - Example: `synthetic_finetune_data/{bboxes,pages,transcripts}/...`

Resulting layout (partial):
```
RenAIssance_SyntheticImageGeneration_Saarthak_Gupta/
├─ data/
├─ OCR_data_original/
├─ OCR_data_synthetic/
└─ synthetic_finetune_data/
```

---

## Data Generation Pipeline

All functions below are implemented in `src/data_utils.py` and orchestrated by `src/data_generation.py`.

### 1) Splitting Pages

- `process_books_with_transcripts(...)` splits multi‑page PDFs into per‑page images and produces per‑page transcripts.
- `copy_all_images()` and `copy_all_transcripts()` normalize naming to `Book_{book}_{page}.ext`.

![alt text](assets/image2.png)

### 2) Image Preprocessing

- `correct_skew`, `ensure_300ppi`, `remove_bleed_dual_layer`, `denoise_image`, `sharpen_image`, `enhance_contrast`, `binarize_image`, `morphological_operations`, `upscale`  
- `process_multiple_books(...)` applies a configurable sequence per book.

![alt text](assets/preprocessing.png)

### 3) Text Detection

- `text_detection(input_root, output_root, model_path)` runs CRAFT to find word‑level bounding boxes and writes `.txt` files.

![alt text](assets/plots/plot7.png)

### 4) Aligning Detection with Transcript

- `mapping_bounding_boxes(...)` uses PyTesseract + string similarity to align bounding boxes to words (threshold configurable).

![alt text](assets/mapping.png)

### 5) Dataset Creation

- `extract_and_process_all_regions(...)` crops word images and creates `words.csv`.
- `resize_and_pad(...)` standardizes word image size (default 64×128).
- `generate_text_image_dataset(...)` renders source font images to pair with targets and writes `data.csv`.

![alt text](assets/final_data.png)

### 6) Grid Construction

- `create_image_grids(...)` builds N×M grids (e.g., 4×2) for Pix2Pix training and saves `grid_info.csv`.

---

## Model Building & Training

All model definitions and training helpers live in `src/model_utils.py`.

Generator & Discriminator:
- `UNetGenerator` — U‑Net with skip connections for image‑to‑image translation.
- `PatchDiscriminator` — 70×70 PatchGAN for local realism.

![alt text](assets/model.png)

Training loop:
- `train_pix2pix(csv_file, epochs=100, batch_size=32, lr=2e-4, save_dir="...")`
  - Generator loss = GAN (BCE) + 100× L1
  - Discriminator loss = 0.5 × (real + fake)
- `plot_gan_history(csv_path)` and `visualize_pix2pix_results(...)` for monitoring and samples.

![alt text](assets/training_history_history.png)

---

## Inference & Synthetic Page Generation

- `GANInferencePipeline` supports:
  - `render_single_word`, `create_grid_from_words`
  - `generate_handwriting(words)` → grid input + generated handwriting
  - `save_results(...)` to persist images

![alt text](assets/test_samples.png)

Use it to generate new word samples and full synthetic pages.

![alt text](assets/pages.png)

---

## Using the Notebooks

Recommended order for a first run:

1) `SyntheticDataGeneration.ipynb` (end-to-end)
  - Update any paths in the first few cells if your data differs from `data/...` or if you are using a custom layout.
   - Run through data splitting → preprocessing → detection → mapping → dataset → grid → training → inference.
   - The following notebook generate data to train the WGAN model from the available PDF and also contain the code to generate synthetic page images using algorithmic degradation.

2) `OCR.ipynb`
   -  OCR experiments and evaluation using CRAFT + OCR over your processed pages(visible improvement in performance when using synthetic data).

3) `finetune-craft-custom.ipynb`
   - Fine‑tune CRAFT on your own annotations. Ensure training/val lists and data roots are set.

---
