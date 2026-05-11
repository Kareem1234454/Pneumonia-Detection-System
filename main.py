import argparse
import os
import sys
from src.transfer_learning import train_vgg16


#conda activate pneumonia_env


# ===========================================================
# 1. Evaluate Mode
# ===========================================================
def run_evaluate(args):
    import tensorflow as tf
    from src.evaluate  import evaluate_model
    from config        import BEST_MODEL_PATH, MODEL_DIR

    vgg_path    = os.path.join(MODEL_DIR, 'vgg16_finetuned.h5')
    frozen_path = os.path.join(MODEL_DIR, 'vgg16_best.h5')

    if os.path.exists(vgg_path):
        model_path = vgg_path
        print(f" Loading: vgg16_finetuned.h5")

    elif os.path.exists(frozen_path):
        model_path = frozen_path
        print(f" Loading: vgg16_best.h5")

    else:
        print(" No trained model found!")
        print("   Run training first:")
        print("   python main.py --mode train")
        return

    model = tf.keras.models.load_model(model_path)
    metrics, *_ = evaluate_model(model)
    return metrics


# ===========================================================
# 2. Demo Mode (Gradio)
# ===========================================================
def run_demo(args):
    from app.app import build_interface
    from config import APP_PORT, APP_SHARE

    demo = build_interface()
    print(f"\n     : http://localhost:{APP_PORT}")
    demo.launch(
        server_port=APP_PORT,
        share=args.share or APP_SHARE,
        show_error=True
    )


# ===========================================================
# 3. Pipeline Mode — اختبار على صورة واحدة
# ===========================================================
def run_pipeline(args):

    import numpy as np
    import tensorflow as tf

    from src.preprocessing import preprocess, plot_preprocessing_steps
    from src.segmentation import segment_lungs, plot_segmentation_pipeline
    from src.gradcam import gradcam_single
    from src.evaluate import load_model

    path = args.image
    if not path or not os.path.exists(path):
        print("wrong path:--image path/to/image.jpg")
        sys.exit(1)

    print(f"\n  : {path}")
    print("=" * 50)

    print("\nPreprocessing...")  
    plot_preprocessing_steps(path)

    print("\nSegmentation...")
    plot_segmentation_pipeline(path)

    print("\nGrad-CAM...")
    model = load_model(args.model_path)
    gradcam_single(model, path)

    print("\nPipeline complete!")



# ===========================================================
# Argument Parser
# ===========================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description=" Pneumonia Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--mode',
        choices=['train_vgg', 'evaluate', 'demo', 'pipeline', 'info'],
        default='info',
        help='select mode: train_vgg | evaluate | demo | pipeline | info'
    )
    parser.add_argument(
        '--model',
        choices=['custom', 'vgg16'],
        default='custom',
        help='select model: custom | vgg16'
    )
    parser.add_argument(
        '--model_path',
        default=None,
        help='model path (optional)'
    )
    parser.add_argument(
        '--image',
        default=None,
        help='image path (for pipeline mode)'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='share Gradio link via internet'
    )

    return parser.parse_args()


# ===========================================================
# Main
# ===========================================================
if __name__ == "__main__":
    args = parse_args()

    if args.model_path is None:
        from config import BEST_MODEL_PATH
        args.model_path = BEST_MODEL_PATH

    print(f"\n Pneumonia Detection System")
    print(f" Mode: {args.mode}\n")


    if args.mode == 'train_vgg':train_vgg16()
    elif args.mode == 'evaluate': run_evaluate(args)
    elif args.mode == 'demo':     run_demo(args)
    elif args.mode == 'pipeline': run_pipeline(args)
  
