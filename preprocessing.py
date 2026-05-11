from datetime import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_SIZE, CLAHE_CLIP, CLAHE_GRID, MEDIAN_BLUR, FIGURES_DIR



def load_image(path):

    img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"{path}")
    return img



def resize_image(img, size=IMG_SIZE):

    h, w = img.shape[:2]
    interpolation = cv2.INTER_AREA if (h > size[0] or w > size[1]) else cv2.INTER_LINEAR
    return cv2.resize(img, size, interpolation=interpolation)



def apply_clahe(img, clip_limit=CLAHE_CLIP, tile_grid=CLAHE_GRID):

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)



def remove_noise_medianBlur(img, kernel_size=MEDIAN_BLUR):

    return cv2.medianBlur(img, kernel_size)


def apply_histogram_eq(img):

    return cv2.equalizeHist(img)


def apply_gaussian_blur(img, kernel_size=(3, 3)):

    return cv2.GaussianBlur(img, kernel_size, 0)



def normalize(img):

    return img.astype(np.float32) / 255.0



def preprocess_image(path):

    img = load_image(path)
    img = resize_image(img)
    enhanced = apply_clahe(img)
        
    normalized = normalize(enhanced)      
    return enhanced , normalized


def visualize_preprocessing_pipeline(image_path, save=False,filename="pipeline_output.png"):
    original = load_image(image_path)
    resized  = resize_image(original)
    hist_eq  = apply_histogram_eq(resized)
    clahe    = apply_clahe(resized)
    gaussian = apply_gaussian_blur(clahe)
    median   = remove_noise_medianBlur(clahe)

    steps = [
        (original,  "Original"),
        (resized,   "Resize\n224x224"),
        (hist_eq,   "Hist EQ"),
        (clahe,     "CLAHE"),
        (gaussian,  "Gaussian"),
        (median,    "Median"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(25, 6)) 
    fig.suptitle("Preprocessing Pipeline", fontsize=18, fontweight='bold', y=1.1)

    for ax, (img, title) in zip(axes, steps): 
        ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=12, pad=10)
        ax.axis('off')

        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        ax_inset = ax.inset_axes([0.0, -0.4, 1.0, 0.3]) 
        ax_inset.fill_between(range(256), hist.flatten(), color='steelblue', alpha=0.7)
        ax_inset.set_xlim([0, 256])
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_facecolor('#f0f0f0')

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, filename)
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f"Saved: {path}")
    
    plt.show()

