

import os

# ----------------------------------------------------------
#   — Paths
# ----------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR    = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
APP_DIR     = os.path.join(BASE_DIR, "app")

# ----------------------------------------------------------
#     — Image Settings
# ----------------------------------------------------------
IMG_SIZE    = (224, 224)      
IMG_CHANNELS = 3             
CLAHE_CLIP  = 2.0          
CLAHE_GRID  = (8, 8)         
MEDIAN_BLUR = 3           
VGG_IMG_SIZE   = (224, 224)   
VGG_INPUT_SHAPE = (224, 224, 3)  
# ----------------------------------------------------------
#    — Training Settings
# ----------------------------------------------------------
BATCH_SIZE    = 32
EPOCHS        = 20
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 0.2
RANDOM_SEED   = 42
PATIENCE      = 5

# ----------------------------------------------------------
#    — Model Settings
# ----------------------------------------------------------
MODEL_NAME      = "pneumonia_cnn"
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")
FINAL_MODEL_PATH= os.path.join(MODEL_DIR, "final_model.h5")
DROPOUT_RATE    = 0.5
NUM_CLASSES     = 2
CLASS_NAMES     = ["NORMAL", "PNEUMONIA"]

# ----------------------------------------------------------
#    — Evaluation Settings
# ----------------------------------------------------------
THRESHOLD       = 0.75         

# ----------------------------------------------------------
#   Grad-CAM
# ----------------------------------------------------------
GRADCAM_LAYER   = "conv2d_3"  
GRADCAM_ALPHA   = 0.4         

# ----------------------------------------------------------
#    App
# ----------------------------------------------------------
APP_TITLE       = "Pneumonia Detection System"
APP_DESCRIPTION = "Upload a chest X-ray image to detect pneumonia using deep learning and Grad-CAM explainability."
APP_PORT        = 7860
APP_SHARE       = False    

# ----------------------------------------------------------
#    — General
# ----------------------------------------------------------
VERBOSE = True                  
