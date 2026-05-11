<div align="center">

#  Pneumonia Detection System

**Deep learning pipeline for automated pneumonia detection from chest X-rays**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Gradio](https://img.shields.io/badge/Gradio-UI-F97316?style=flat-square)](https://gradio.app)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

---

## Overview

This project implements a full computer vision pipeline for **binary classification of chest X-ray images** — distinguishing normal lungs from pneumonia cases. The system combines classical image processing techniques (CLAHE, morphological segmentation) with deep learning (VGG16 transfer learning) and explainability tools (Grad-CAM), wrapped in an interactive Gradio web interface.

> **Disclaimer:** This system is a research and educational tool. It is not a substitute for professional medical diagnosis. Always consult a qualified radiologist.

---

## Results

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~96%   |
| Precision | ~94%   |
| Recall    | ~97%   |
| F1 Score  | ~95.5% |
| AUC-ROC   | ~0.98  |

---

## Project Structure

```
pneumonia-detection/
│
├── config.py                    # Global settings — paths, hyperparameters, thresholds
├── main.py                      # CLI entry point
├── requirements.txt             # Python dependencies
│
├── src/
│   ├── preprocessing.py         # CLAHE · Median Filter · Normalization · Augmentation
│   ├── segmentation.py          # Otsu Thresholding · Morphology · Lung Contour Extraction
│   ├── dataset.py               # Data loading · train/val/test splits · generators
│   ├── model.py                 # Custom CNN · VGG16 transfer learning head
│   ├── train.py                 # Training loop · callbacks · checkpointing
│   ├── evaluate.py              # Metrics · Confusion Matrix · ROC Curve
│   └── gradcam.py               # Grad-CAM heatmap · batch visualization
│
├── app/
│   └── app.py                   # Gradio web interface
│
├── notebooks/
│   └── Pneumonia_Detection_Colab.ipynb
│
├── data/
│   ├── raw/                     # Original Kaggle dataset
│   │   ├── train/
│   │   │   ├── NORMAL/
│   │   │   └── PNEUMONIA/
│   │   ├── val/
│   │   └── test/
│   └── processed/               # Preprocessed images
│
├── models/
│   ├── best_model.h5            # Best checkpoint (highest val accuracy)
│   └── final_model.h5           # Final epoch weights
│
└── outputs/
    ├── figures/
    │   ├── preprocessing_pipeline.png
    │   ├── segmentation_pipeline.png
    │   ├── training_curves.png
    │   ├── confusion_matrix.png
    │   ├── roc_curve.png
    │   └── gradcam_samples/
    └── results/
        └── metrics_report.csv
```

---

## Pipeline Architecture

```
Raw Chest X-Ray
      │
      ▼
┌─────────────────────────────┐
│     Preprocessing           │
│  · Grayscale conversion     │
│  · Resize → 224×224         │
│  · CLAHE enhancement        │
│  · Median noise removal     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│     Lung Segmentation       │
│  · Otsu thresholding        │
│  · Morphological opening    │
│  · Contour detection        │
│  · Region of interest crop  │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│     Classification Model    │
│  · VGG16 (ImageNet weights) │
│  · Custom dense head        │
│  · Binary sigmoid output    │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│     Grad-CAM Explainability │
│  · Gradient computation     │
│  · Activation heatmap       │
│  · Focus region overlay     │
└─────────────────────────────┘
```

---

## Techniques

### Image Processing (DIP)

| Technique | Purpose |
|-----------|---------|
| **CLAHE** (Contrast Limited Adaptive Histogram Equalization) | Enhance local contrast in X-ray images without amplifying noise |
| **Median Filter** | Remove salt-and-pepper noise while preserving anatomical edges |
| **Data Augmentation** | Rotation ±10°, horizontal flip, zoom ±10%, width/height shift — increases effective dataset size and reduces overfitting |

### Computer Vision (CV)

| Technique | Purpose |
|-----------|---------|
| **Otsu Thresholding** | Automatic foreground/background separation using optimal global threshold |
| **Morphological Operations** | Opening (remove small artifacts) + Closing (fill holes in lung mask) |
| **Contour Detection** | Isolate the two largest contours corresponding to left and right lungs |

### Deep Learning

| Component | Detail |
|-----------|--------|
| **Custom CNN** | 4 convolutional blocks (Conv2D → BatchNorm → MaxPool → Dropout), trained from scratch |
| **VGG16 Transfer Learning** | ImageNet pre-trained backbone, frozen during initial training, fine-tuned in later epochs |
| **Grad-CAM** | Computes gradients of the predicted class score w.r.t. the last Conv layer — produces a spatial heatmap highlighting regions most influential to the decision |

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/pneumonia-detection.git
cd pneumonia-detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

Download the [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) dataset from Kaggle.

```bash
# Using the Kaggle CLI
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/raw/
```

| Split      | Normal | Pneumonia | Total  |
|------------|--------|-----------|--------|
| Train      | 1,341  | 3,875     | 5,216  |
| Validation | 8      | 8         | 16     |
| Test       | 234    | 390       | 624    |

> The validation set is small by design in the original Kaggle split. The training script optionally carves a larger validation split from the training data.

---

## Usage

All commands go through `main.py`.

### Train

```bash
# Train a custom CNN from scratch
python main.py --mode train

# Train with VGG16 transfer learning
python main.py --mode train --model vgg16
```

### Evaluate

```bash
python main.py --mode evaluate
```

Outputs: accuracy, precision, recall, F1, AUC, confusion matrix, and ROC curve saved to `outputs/figures/`.

### Run the Web Interface

```bash
# Local only
python main.py --mode demo

# Generate a public Gradio share link
python main.py --mode demo --share
```

### Analyze a Single Image

```bash
python main.py --mode pipeline --image path/to/xray.jpeg
```

Runs the full pipeline (preprocessing → segmentation → prediction → Grad-CAM) on one image and saves all figures.

### Project Info

```bash
python main.py --mode info
```

---

## Grad-CAM — How It Works

Grad-CAM (Gradient-weighted Class Activation Mapping) answers the question: *"Which regions of the image drove the model's decision?"*

1. **Forward pass** — run the image through the model and record the prediction.
2. **Gradient computation** — compute the gradient of the predicted class score with respect to the output of the last convolutional layer using `tf.GradientTape`.
3. **Channel weighting** — apply Global Average Pooling on the gradients to get one importance weight per feature map channel.
4. **Heatmap** — take a weighted sum of the feature maps, apply ReLU, and normalize to [0, 1].
5. **Overlay** — upsample the heatmap to the original image size and blend it using a thermal colormap.

```
High activation (red/white)  →  region strongly influenced the decision
Low activation  (dark blue)  →  region had little effect on the decision
```

---

## Configuration

All tunable parameters live in `config.py`. Key settings:

```python
IMG_SIZE        = (224, 224)      # Input resolution for the model
THRESHOLD       = 0.5             # Sigmoid threshold for PNEUMONIA vs NORMAL
GRADCAM_ALPHA   = 0.55            # Heatmap opacity in overlay (0–1)
BEST_MODEL_PATH = "models/best_model.h5"
FIGURES_DIR     = "outputs/figures"
CLASS_NAMES     = ["NORMAL", "PNEUMONIA"]

# App
APP_TITLE       = "Pneumonia Detection System"
APP_PORT        = 7860
APP_SHARE       = False
```

---

## Requirements

```
tensorflow>=2.10
opencv-python
numpy
matplotlib
Pillow
gradio
scikit-learn
```

Full list in `requirements.txt`.

---

## Project Roadmap

- [x] Image preprocessing pipeline (CLAHE + Median Filter)
- [x] Lung segmentation (Otsu + Morphology + Contours)
- [x] Custom CNN classifier
- [x] VGG16 transfer learning
- [x] Grad-CAM explainability
- [x] Gradio web interface
- [ ] Multi-class classification (bacterial vs. viral pneumonia)
- [ ] DICOM format support
- [ ] REST API deployment (FastAPI)
- [ ] Docker containerization

---

## License

This project is released under the [MIT License](LICENSE).

---

<div align="center">
<sub>Built for academic purposes · Not for clinical use</sub>
</div>