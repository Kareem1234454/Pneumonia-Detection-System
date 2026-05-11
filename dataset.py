import os
import sys
import cv2
import numpy as np
import glob
from tqdm import tqdm                    
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (DATA_DIR, PROC_DIR)
from src.preprocessing import preprocess_image
from src.segmentation  import segment_lungs


def gray_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

def process_single_image(img_path):

    enhanced, _ = preprocess_image(img_path)

    seg_result  = segment_lungs(enhanced)
    lung_only   = seg_result['lung_only']  
    lung_only = gray_to_rgb(lung_only)          

    return lung_only, enhanced



def build_processed_dataset(
        raw_dir=None,
        out_dir=None,
        splits=('train', 'test', 'val'),
        classes=('NORMAL', 'PNEUMONIA')):

    raw_dir = raw_dir or DATA_DIR
    out_dir = out_dir or PROC_DIR

    total_ok  = 0
    total_err = 0

    for split in splits:
        for cls in classes:
            in_path  = os.path.join(raw_dir, split, cls)
            out_path = os.path.join(out_dir, split, cls)

            if not os.path.exists(in_path):
                print(f"  Skipping (not found): {in_path}")
                continue

            os.makedirs(out_path, exist_ok=True)

            images = (glob.glob(os.path.join(in_path, '*.jpeg')) +
                      glob.glob(os.path.join(in_path, '*.jpg'))  +
                      glob.glob(os.path.join(in_path, '*.png')))

            print(f"\n {split}/{cls}: {len(images)} images")

            for img_path in tqdm(images, desc=f"{split}/{cls}"):
                filename  = os.path.basename(img_path)
                save_path = os.path.join(out_path, filename)

                if os.path.exists(save_path):
                    total_ok += 1
                    continue

                try:
                    lung_only, _ = process_single_image(img_path)
                    cv2.imwrite(save_path, lung_only)
                    total_ok += 1
                except Exception as e:
                    print(f"\n   Error: {filename} → {e}")
                    total_err += 1

    print(f"\n{'='*50}")
    print(f" Done!  Success: {total_ok}  |  Errors: {total_err}")
    print(f" Saved to: {out_dir}")
    print(f"{'='*50}")

    return total_ok, total_err

if __name__ == "__main__":
    build_processed_dataset()

