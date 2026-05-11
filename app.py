# ============================================================
#  app/app.py — Pneumonia Detection — Gradio Interface
# ============================================================

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
import gradio as gr

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    IMG_SIZE, BEST_MODEL_PATH, THRESHOLD,
    APP_TITLE, APP_DESCRIPTION, APP_PORT, APP_SHARE,
    CLASS_NAMES,
)
from src.preprocessing import apply_clahe, normalize
from src.segmentation import segment_lungs
from src.gradcam import compute_gradcam, overlay_heatmap


# ── Matplotlib global style ──────────────────────────────────
BG      = "#0d1117"
PANEL   = "#161b22"
BORDER  = "#30363d"
TEXT    = "#e6edf3"
SUBTEXT = "#8b949e"
ACCENT  = "#58a6ff"
RED     = "#f85149"
GREEN   = "#3fb950"

plt.rcParams.update({
    "figure.facecolor":  BG,
    "axes.facecolor":    PANEL,
    "axes.edgecolor":    BORDER,
    "axes.labelcolor":   SUBTEXT,
    "xtick.color":       SUBTEXT,
    "ytick.color":       SUBTEXT,
    "text.color":        TEXT,
    "font.family":       "monospace",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

THERMAL = LinearSegmentedColormap.from_list(
    "thermal",
    ["#03071e", "#023e8a", "#0077b6", "#00b4d8",
     "#f7b731", "#f3722c", "#f94144", "#ffffff"],
)


# ===========================================================
# Model Loading
# ===========================================================

print("[App] Loading model...")
try:
    model = tf.keras.models.load_model(BEST_MODEL_PATH)
    print("[App] Model loaded successfully.")
    MODEL_LOADED = True
except Exception as e:
    print(f"[App] Could not load model: {e}")
    print("[App] Running in Demo Mode.")
    model = None
    MODEL_LOADED = False


# ===========================================================
# Preprocessing
# ===========================================================

def preprocess_uploaded(pil_image):
    """
    PIL image  →  (gray uint8,  normalized float32 RGB,  lung_mask uint8)

    Steps:
        1. Grayscale conversion
        2. Resize to IMG_SIZE
        3. CLAHE + Median denoising
        4. Lung segmentation (graceful fallback on failure)
        5. Normalize to [0, 1] float32 RGB for the model
    """
    img_array = np.array(pil_image)

    if img_array.ndim == 3 and img_array.shape[2] == 4:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGBA2GRAY)
    elif img_array.ndim == 3 and img_array.shape[2] == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.astype(np.uint8)

    gray = cv2.resize(gray, IMG_SIZE)
    gray = apply_clahe(gray)

    # Segmentation — graceful fallback
    lung_mask = np.ones(gray.shape, dtype=np.uint8) * 255
    lung_gray = gray.copy()

    try:
        seg  = segment_lungs(gray)
        mask = seg.get("final_mask") or seg.get("roi_mask")
        if mask is not None:
            m = mask.astype(np.uint8)
            lung_mask = (m * 255).astype(np.uint8) if m.max() <= 1 else m

        raw = seg.get("lung_only", gray)
        if raw.ndim == 3:
            lung_gray = cv2.cvtColor(raw.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            lung_gray = raw.astype(np.uint8)

    except Exception as e:
        print(f"[Segmentation] Failed ({e}) — using full image.")

    rgb      = cv2.cvtColor(lung_gray, cv2.COLOR_GRAY2RGB)
    normalized = normalize(rgb).astype(np.float32)
    if normalized.max() > 1.0:
        normalized = normalized / 255.0

    return gray, normalized, lung_mask


# ===========================================================
# Synthetic Grad-CAM for Demo Mode
# ===========================================================

def _synthetic_gradcam(gray):
    """Gaussian blob heatmap centred on the typical lung region."""
    h, w   = gray.shape
    Y, X   = np.ogrid[:h, :w]
    cy, cx = h * 0.55, w * 0.50
    sig    = min(h, w) * 0.28
    blob   = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2 * sig ** 2))
    cy2, cx2 = h * 0.62, w * 0.63
    sig2   = min(h, w) * 0.14
    blob2  = np.exp(-((X - cx2) ** 2 + (Y - cy2) ** 2) / (2 * sig2 ** 2))
    heatmap = np.clip(blob * 0.75 + blob2 * 0.55, 0, 1).astype(np.float32)

    heatmap_rgb  = THERMAL(heatmap)[:, :, :3]
    gray_rgb     = np.stack([gray / 255.0] * 3, axis=-1)
    superimposed = np.clip(0.5 * gray_rgb + 0.5 * heatmap_rgb, 0, 1)
    superimposed = (superimposed * 255).astype(np.uint8)
    return heatmap, superimposed


# ===========================================================
# Figures
# ===========================================================

def build_preprocessing_figure(pil_image, gray, lung_mask):
    """Original | CLAHE grayscale | Lung mask overlay."""
    orig = np.array(pil_image)
    if orig.ndim == 3 and orig.shape[2] == 4:
        orig = cv2.cvtColor(orig, cv2.COLOR_RGBA2RGB)

    # Green tinted mask overlay
    mask_f      = lung_mask.astype(np.float32) / 255.0
    gray_rgb    = np.stack([gray] * 3, axis=-1).astype(np.float32)
    green_layer = np.zeros_like(gray_rgb)
    green_layer[:, :, 1] = 160
    overlay = np.where(
        mask_f[:, :, None] > 0.5,
        0.65 * gray_rgb + 0.35 * green_layer,
        gray_rgb,
    ).astype(np.uint8)

    fig = plt.figure(figsize=(13, 4.5), facecolor=BG)
    fig.suptitle("PREPROCESSING  PIPELINE",
                 color=TEXT, fontsize=10, fontweight="bold", y=1.01)

    panels = [
        (orig,    "ORIGINAL UPLOAD",               None),
        (gray,    "CLAHE + MEDIAN FILTER",          "gray"),
        (overlay, "LUNG SEGMENTATION MASK",         None),
    ]

    for i, (img, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(1, 3, i + 1)
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, color=SUBTEXT, fontsize=8,
                     fontfamily="monospace", pad=6)
        ax.axis("off")

    plt.tight_layout(pad=0.8)
    return fig


def build_gradcam_figure(gray, normalized, prediction):
    """6-panel Grad-CAM explainability figure (works in Demo Mode too)."""
    if MODEL_LOADED:
        try:
            img_input       = np.expand_dims(normalized, axis=0)
            heatmap, _      = compute_gradcam(model, img_input)
            superimposed, _ = overlay_heatmap(normalized, heatmap)
        except Exception as e:
            print(f"[Grad-CAM] {e} — using synthetic heatmap.")
            heatmap, superimposed = _synthetic_gradcam(gray)
    else:
        heatmap, superimposed = _synthetic_gradcam(gray)

    h, w         = gray.shape
    heatmap_full = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
    heatmap_full = np.clip(heatmap_full, 0, 1)

    heatmap_rgb  = THERMAL(heatmap_full)[:, :, :3]
    gray_rgb     = np.stack([gray / 255.0] * 3, axis=-1)
    blend        = np.clip(0.50 * gray_rgb + 0.50 * heatmap_rgb, 0, 1)

    mask        = (heatmap_full > 0.65).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fig = plt.figure(figsize=(15, 9), facecolor=BG)
    gs  = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.38, wspace=0.06,
        left=0.04, right=0.94,
        top=0.87,  bottom=0.07,
    )
    lkw = dict(color=SUBTEXT, fontsize=8, fontfamily="monospace", pad=7)

    # ── Row 0 ────────────────────────────────────────────────
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(gray, cmap="bone")
    ax0.set_title("ORIGINAL", **lkw)
    ax0.axis("off")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(heatmap_full, cmap=THERMAL, vmin=0, vmax=1)
    ax1.set_title("ACTIVATION MAP", **lkw)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.imshow(blend)
    for cnt in contours:
        if cv2.contourArea(cnt) > 120:
            draw = np.zeros_like(gray)
            cv2.drawContours(draw, [cnt], -1, 255, 2)
            ax2.contour(draw, levels=[128],
                        colors=["#ffffff"], linewidths=0.7, alpha=0.55)
    ax2.set_title("OVERLAY + FOCUS REGIONS", **lkw)
    ax2.axis("off")

    # ── Row 1 ────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    vals   = heatmap_full.flatten()
    n_bins = 50
    counts, edges = np.histogram(vals, bins=n_bins, range=(0, 1))
    centers = (edges[:-1] + edges[1:]) / 2
    ax3.bar(centers, counts, width=1 / n_bins,
            color=[THERMAL(c) for c in centers], linewidth=0)
    mean_v = float(np.mean(vals))
    ax3.axvline(mean_v, color=TEXT, lw=0.9, linestyle="--", alpha=0.7)
    ax3.text(mean_v + 0.03, counts.max() * 0.82,
             f"μ={mean_v:.2f}", color=TEXT, fontsize=7)
    ax3.set_title("ACTIVATION DISTRIBUTION", **lkw)
    ax3.set_xlabel("Activation", fontsize=7)
    ax3.set_ylabel("Pixels",     fontsize=7)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(np.linspace(0, 1, 256).reshape(1, -1),
               aspect="auto", cmap=THERMAL, extent=[0, 1, 0, 1])
    ax4.set_yticks([])
    ax4.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax4.set_xticklabels(["0%", "25%", "50%", "75%", "100%"],
                        fontsize=7, color=SUBTEXT)
    ax4.set_title("COLOR SCALE", **lkw)
    ax4.set_xlabel("Low  →  High Activation", fontsize=7)

    ax5 = fig.add_subplot(gs[1, 2])
    ax5.axis("off")
    peak_v   = float(np.max(heatmap_full))
    peak_loc = np.unravel_index(np.argmax(heatmap_full), heatmap_full.shape)
    hi_pct   = float(np.mean(heatmap_full > 0.65)) * 100
    n_focus  = sum(1 for c in contours if cv2.contourArea(c) > 120)

    stats = [
        ("Peak Activation",  f"{peak_v:.3f}"),
        ("Peak Location",    f"({peak_loc[1]}, {peak_loc[0]})"),
        ("Mean Activation",  f"{mean_v:.3f}"),
        ("High-Act Region",  f"{hi_pct:.1f}%"),
        ("Focus Contours",   str(n_focus)),
        ("Mode", "DEMO  [synthetic]" if not MODEL_LOADED else "MODEL"),
    ]
    ax5.text(0.05, 0.96, "STATISTICS", color=SUBTEXT, fontsize=8,
             fontfamily="monospace", transform=ax5.transAxes, va="top")
    for k, (key, val) in enumerate(stats):
        y = 0.80 - k * 0.13
        ax5.text(0.05, y, key,  color="#484f58", fontsize=7,
                 transform=ax5.transAxes, va="top")
        ax5.text(0.97, y, val, color=TEXT, fontsize=8, fontweight="bold",
                 transform=ax5.transAxes, va="top", ha="right")

    mode_tag = "  [DEMO — synthetic heatmap]" if not MODEL_LOADED else ""
    fig.text(0.49, 0.94,
             f"GRAD-CAM  ·  EXPLAINABILITY  ANALYSIS{mode_tag}",
             ha="center", color=TEXT, fontsize=11,
             fontfamily="monospace", fontweight="bold")

    return fig


def build_confidence_figure(prediction):
    """Horizontal confidence bar chart."""
    fig = plt.figure(figsize=(7, 3), facecolor=BG)
    ax  = fig.add_subplot(111)

    probs  = [1 - prediction, prediction]
    colors = [GREEN, RED]

    bars = ax.barh(CLASS_NAMES, probs, color=colors,
                   height=0.4, edgecolor="none")

    for bar, prob in zip(bars, probs):
        ax.text(
            min(bar.get_width() + 0.025, 0.96),
            bar.get_y() + bar.get_height() / 2,
            f"{prob:.1%}",
            va="center", color=TEXT,
            fontsize=10, fontweight="bold",
            fontfamily="monospace",
        )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence", fontsize=9)
    ax.tick_params(labelsize=10)
    ax.set_title("CLASSIFICATION  CONFIDENCE",
                 color=TEXT, fontsize=10,
                 fontfamily="monospace", pad=10)
    ax.xaxis.grid(True, color=BORDER, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    return fig


# ===========================================================
# Main Analysis
# ===========================================================

def analyze_xray(image):
    import traceback
    try:
        if image is None:
            return None, "*Upload an image and click Analyze.*", None, None

        gray, normalized, lung_mask = preprocess_uploaded(image)

        if MODEL_LOADED:
            img_input  = np.expand_dims(normalized, axis=0)
            prediction = float(model.predict(img_input, verbose=0)[0][0])
        else:
            import random
            prediction = random.uniform(0.1, 0.9)

        is_pneumonia = prediction >= THRESHOLD
        confidence   = prediction if is_pneumonia else 1 - prediction
        dot          = "🔴" if is_pneumonia else "🟢"
        label        = "Pneumonia Detected" if is_pneumonia else "Normal"

        result_text = (
            f"### {dot} Diagnosis: {label}\n\n"
            f"**Confidence:** {confidence:.1%}\n\n"
            f"> This system is a decision-support tool only. "
            f"Please consult a qualified radiologist."
        )

        fig_proc    = build_preprocessing_figure(image, gray, lung_mask)
        fig_gradcam = build_gradcam_figure(gray, normalized, prediction)
        fig_bar     = build_confidence_figure(prediction)

        return fig_proc, result_text, fig_gradcam, fig_bar

    except Exception as e:
        traceback.print_exc()
        return None, f"**Error:** `{str(e)}`", None, None


# ===========================================================
# Gradio Interface
# ===========================================================

CSS = f"""
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&display=swap');

*, body, .gradio-container {{
    background-color: {BG} !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: {TEXT} !important;
    box-sizing: border-box;
}}
.gr-button-primary, button.primary {{
    background: {ACCENT} !important;
    color: #0d1117 !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    font-family: 'JetBrains Mono', monospace !important;
}}
.gr-button-primary:hover {{ opacity: 0.82 !important; }}
.gr-box, .gr-panel, .panel, [class*="block"] {{
    background: {PANEL} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}
label, .label-wrap span {{
    color: {SUBTEXT} !important;
    font-size: 11px !important;
    letter-spacing: 0.5px !important;
    text-transform: uppercase !important;
}}
.gr-markdown h3 {{ color: {TEXT} !important; font-size: 1rem !important; }}
.gr-markdown p, .gr-markdown blockquote {{
    color: {SUBTEXT} !important;
    font-size: 13px !important;
}}
.gr-markdown strong {{ color: {TEXT} !important; }}
.gr-plot, [class*="plot"] {{
    background: {BG} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
}}
footer {{ display: none !important; }}
"""


def build_interface():
    with gr.Blocks(title=APP_TITLE, css=CSS) as demo:

        gr.HTML(f"""
        <div style="text-align:center; padding:28px 0 12px;">
            <div style="font-size:10px; letter-spacing:5px; color:{SUBTEXT};
                        margin-bottom:10px; font-family:'JetBrains Mono',monospace;">
                RADIOLOGY · AI · EXPLAINABILITY
            </div>
            <h1 style="font-size:1.9rem; font-weight:700; color:{TEXT};
                       margin:0; letter-spacing:2px;
                       font-family:'JetBrains Mono',monospace;">
                🫁 {APP_TITLE}
            </h1>
            <p style="color:{SUBTEXT}; font-size:13px; margin-top:10px;
                      font-family:'JetBrains Mono',monospace;">
                {APP_DESCRIPTION}
            </p>
        </div>
        """)

        if not MODEL_LOADED:
            gr.HTML(f"""
            <div style="color:#e3b341; font-size:12px; padding:6px 14px;
                        border-left:3px solid #e3b341;
                        background:rgba(227,179,65,0.07); border-radius:4px;
                        font-family:'JetBrains Mono',monospace; margin-bottom:8px;">
                ⚠ DEMO MODE — model not loaded.
                Predictions are random · Grad-CAM shows synthetic heatmap.
            </div>
            """)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                input_image = gr.Image(
                    label="Upload Chest X-Ray",
                    type="pil",
                    height=310,
                )
                analyze_btn = gr.Button(
                    "▶  ANALYZE IMAGE",
                    variant="primary",
                    size="lg",
                )
                gr.Examples(
                    examples=[
                        ["data/raw/test/NORMAL/IM-0001-0001.jpeg"],
                        ["data/raw/test/PNEUMONIA/person1_bacteria_1.jpeg"],
                    ],
                    inputs=input_image,
                    label="Sample Images",
                )

            with gr.Column(scale=2):
                result_text = gr.Markdown(
                    value="*Upload an image and click Analyze.*"
                )
                result_bar = gr.Plot(label="Confidence")

        with gr.Row():
            proc_plot    = gr.Plot(label="Preprocessing Steps")
            gradcam_plot = gr.Plot(label="Grad-CAM — Focus Regions")

        analyze_btn.click(
            fn=analyze_xray,
            inputs=[input_image],
            outputs=[proc_plot, result_text, gradcam_plot, result_bar],
        )

        gr.HTML(f"""
        <div style="text-align:center; padding:18px 0 6px;
                    color:{SUBTEXT}; font-size:10px; letter-spacing:1.5px;
                    font-family:'JetBrains Mono',monospace;">
            FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY · NOT FOR CLINICAL USE
        </div>
        """)

    return demo


# ===========================================================
# Entry Point
# ===========================================================

if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_port=APP_PORT,
        share=APP_SHARE,
        show_error=True,
    )