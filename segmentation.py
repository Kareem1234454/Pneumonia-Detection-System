import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FIGURES_DIR
from src.preprocessing import load_image, resize_image, apply_clahe , preprocess_image

#-------------------------------------------------------------

def otsu_threshold(img):

    threshold_value, binary_img = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return threshold_value, binary_img

def morphological_cleanup(binary_img):

    kernel_small = np.ones((5, 5), np.uint8)
    kernel_large = np.ones((15, 15), np.uint8)

    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel_small)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large)

    return closed



def find_lung_contours(binary_img):

    contours, _ = cv2.findContours(
        binary_img,
        cv2.RETR_EXTERNAL,          
        cv2.CHAIN_APPROX_SIMPLE     
    )

    if len(contours) < 2:
        return contours, contours

    top2 = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
    return contours, top2



def create_lung_mask(img_shape, lung_contours):

    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.drawContours(mask, lung_contours, -1, 255, -1)       
    return mask



def apply_mask(img, mask):

    return cv2.bitwise_and(img, img, mask=mask)



def plot_segmentation_pipeline(image_path, save=True):
 
    img = load_image(image_path)
    img = resize_image(img)
    img = apply_clahe(img)


    thresh_val, binary   = otsu_threshold(img)
    cleaned              = morphological_cleanup(binary)
    _, lung_contours     = find_lung_contours(cleaned)
    mask                 = create_lung_mask(img.shape, lung_contours)
    segmented            = apply_mask(img, mask)

    img_with_contours = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_with_contours, lung_contours, -1, (0, 255, 0), 2)

    steps = [
        (img,               f" \nPreprocessed"),
        (binary,            f"Otsu Threshold\n(T={thresh_val:.0f})"),
        (cleaned,           " \nMorphological Ops"),
        (img_with_contours, "Contours\n", True),
        (mask,              "Lung Mask\n Mask "),
        (segmented,         "  \nSegmented Lungs"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(25,6))
    fig.suptitle("   — Lung Segmentation Pipeline",
                 fontsize=16, fontweight='bold')

    for i, (ax, step) in enumerate(zip(axes.flatten(), steps)):
        if len(step) == 3:    
            ax.imshow(cv2.cvtColor(step[0], cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(step[0], cmap='gray')
        ax.set_title(step[1], fontsize=12, pad=8)
        ax.axis('off')

    plt.tight_layout()

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "segmentation_pipeline.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f" Saved  : {path}")
    plt.show()

#-------------------------------------------------------------


def segment_lungs(img):

    h, w = img.shape

    y1 = int(h * 0.10);  y2 = int(h * 0.90)
    x1 = int(w * 0.05);  x2 = int(w * 0.95)
    roi_mask = np.zeros_like(img)
    roi_mask[y1:y2, x1:x2] = 255

    roi_img  = cv2.bitwise_and(img, img, mask=roi_mask)
    inverted = cv2.bitwise_not(roi_img)

    _, thresh = cv2.threshold(inverted, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel  = np.ones((7,7), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN,  kernel)

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    final_mask = roi_mask.copy()    

    if contours:
        min_area = h * w * 0.02
        max_area = h * w * 0.45
        valid    = [c for c in contours
                    if min_area < cv2.contourArea(c) < max_area]
        valid    = sorted(valid,
                          key=cv2.contourArea, reverse=True)[:2]

        if valid:     
            contour_mask = np.zeros_like(img)
            cv2.drawContours(contour_mask, valid, -1, 255, -1)
            final_mask = cv2.bitwise_and(contour_mask, roi_mask)

    lung_only  = cv2.bitwise_and(img, img, mask=final_mask)
    result_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    result_contours, _ = cv2.findContours(
        final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_img, result_contours, -1, (0,255,0), 2)

    return {
        'original':    img,
        'roi_mask':    roi_mask,
        'final_mask':  final_mask,
        'lung_only':   lung_only,
        'contour_img': result_img,
    }

def compare_segmentation(normal_img, pneumonia_img, save=True):

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Segmentation: Normal vs Pneumonia', fontsize=15, fontweight='bold')

    for row, (img, label) in enumerate([(normal_img, 'NORMAL'), (pneumonia_img, 'PNEUMONIA')]):
        result = segment_lungs(img)

        steps = [
            (result['original'],              'Original After Preprocessing'),
            (result['roi_mask'],              'ROI Mask'),
            (result['lung_only'],             'Lung Only'),
            (result['contour_img'][:,:,::-1], 'Final Result'),
        ]

        color = 'green' if label == 'NORMAL' else 'red'

        for col, (image, step_title) in enumerate(steps):
            cmap = 'gray' if col < 3 else None
            if cmap:
                axes[row, col].imshow(image, cmap=cmap)
            else:
                axes[row, col].imshow(image)
            if row == 0:
                axes[row, col].set_title(step_title, fontsize=11, fontweight='bold')
            axes[row, col].set_ylabel(label, color=color, fontweight='bold', fontsize=11)
            axes[row, col].set_yticks([])
            axes[row, col].set_xticks([])

    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, 'segmentation_comparison.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f" Saved: {path}")

    plt.show()
    return fig

