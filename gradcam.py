import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_SIZE, GRADCAM_ALPHA, FIGURES_DIR, CLASS_NAMES
from src.preprocessing import preprocess_image




def compute_gradcam(model, img_array, layer_name=None):

    if layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                layer_name = layer.name
                break
        print(f"[Grad-CAM] Target layer auto-selected: {layer_name}")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), float(predictions.numpy()[0][0])


# ===========================================================
# 2. Heatmap Overlay
# ===========================================================

def overlay_heatmap(img_array, heatmap, alpha=GRADCAM_ALPHA):

    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
    img_uint8 = (img_array * 255).astype(np.uint8)

    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

    superimposed = cv2.addWeighted(img_uint8, 1 - alpha, heatmap_colored, alpha, 0)

    return superimposed, heatmap_colored




def gradcam_single(model, image_path, layer_name=None, save=True, idx=0):

    img = preprocess_image(image_path)
    img_array = np.expand_dims(img, axis=0)

    heatmap, prediction = compute_gradcam(model, img_array, layer_name)
    superimposed, heatmap_colored = overlay_heatmap(img, heatmap)

    label = "PNEUMONIA" if prediction >= 0.5 else "NORMAL"
    confidence = prediction if prediction >= 0.5 else 1 - prediction
    title_color = "darkred" if prediction >= 0.5 else "darkgreen"

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle(
        f"Grad-CAM Analysis  |  Prediction: {label}  ({confidence:.1%} confidence)",
        fontsize=14,
        fontweight="bold",
        color=title_color,
    )

    panels = [
        (img,             "Original",        "gray"),
        (heatmap,         "Grad-CAM Heatmap", "jet"),
        (heatmap_colored, "Colorized",        None),
        (superimposed,    "Overlay",          None),
    ]

    for ax, (image, title, cmap) in zip(axes, panels):
        ax.imshow(image, cmap=cmap)
        ax.set_title(title, fontsize=11, pad=8)
        ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    plt.colorbar(sm, ax=axes[1], shrink=0.8, label="Activation Intensity")

    plt.tight_layout()

    if save:
        out_dir = os.path.join(FIGURES_DIR, "gradcam_samples")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"gradcam_{idx}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Grad-CAM] Figure saved: {path}")

    plt.show()
    return heatmap, prediction




def gradcam_batch(model, image_paths, layer_name=None, save=True):

    n = len(image_paths)
    fig, axes = plt.subplots(n, 4, figsize=(18, 5 * n))
    if n == 1:
        axes = [axes]

    fig.suptitle("Grad-CAM — Batch Analysis", fontsize=16, fontweight="bold")

    for i, image_path in enumerate(image_paths):
        img = preprocess_image(image_path)
        img_array = np.expand_dims(img, axis=0)
        heatmap, prediction = compute_gradcam(model, img_array, layer_name)
        superimposed, heatmap_colored = overlay_heatmap(img, heatmap)

        label = "PNEUMONIA" if prediction >= 0.5 else "NORMAL"
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        panels = [
            (img,             f"Original — {label} ({confidence:.1%})", "gray"),
            (heatmap,         "Grad-CAM Heatmap",                        "jet"),
            (heatmap_colored, "Colorized",                               None),
            (superimposed,    "Overlay",                                 None),
        ]

        for j, (image, title, cmap) in enumerate(panels):
            axes[i][j].imshow(image, cmap=cmap)
            axes[i][j].set_title(title, fontsize=10)
            axes[i][j].axis("off")

    plt.tight_layout()

    if save:
        out_dir = os.path.join(FIGURES_DIR, "gradcam_samples")
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, "gradcam_batch.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[Grad-CAM] Batch figure saved: {path}")

    plt.show()




def analyze_with_gradcam(model=None, image_path=None):

    if model is None:
        from src.evaluate import load_model
        model = load_model()

    if image_path is None:
        image_path = "data/raw/test/PNEUMONIA/person1_bacteria_1.jpeg"

    if not os.path.exists(image_path):
        print(f"[Grad-CAM] Image not found: {image_path}")
        return

    gradcam_single(model, image_path)


if __name__ == "__main__":
    analyze_with_gradcam()