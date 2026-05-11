import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_curve, auc)
import tensorflow as tf
from tensorflow.keras.models import load_model as tf_load_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from src.transfer_learning import create_vgg_generators      



def get_predictions(model, test_gen, threshold=THRESHOLD):

    print("Generating predictions...")
    test_gen.reset()

    y_prob = model.predict(test_gen, verbose=1).flatten()
    y_pred = (y_prob >= threshold).astype(int)
    y_true = test_gen.classes

    print(f" Predictions done: {len(y_pred)} samples")
    return y_true, y_pred, y_prob



def print_classification_report(y_true, y_pred):
    print("\n" + "="*60)
    print("         CLASSIFICATION REPORT")
    print("="*60)

    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4
    )
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    precision   = tp / (tp + fp)   if (tp + fp) > 0 else 0
    recall      = tp / (tp + fn)   if (tp + fn) > 0 else 0
    f1          = 2 * precision * recall / (precision + recall) \
                  if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp)   if (tn + fp) > 0 else 0

    print(f"Summary:")
    print(f"   Accuracy    : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"   Precision   : {precision:.4f}")
    print(f"   Recall      : {recall:.4f}")
    print(f"   F1 Score    : {f1:.4f}")
    print(f"   Specificity : {specificity:.4f}")
    print(f"\n   TP: {tp}  |  TN: {tn}  |  FP: {fp}  |  FN: {fn}")
    print("="*60)

    return {
        'accuracy': accuracy, 'precision': precision,
        'recall':   recall,   'f1':        f1,
        'specificity': specificity,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }


def plot_confusion_matrix(y_true, y_pred, save=True):
    cm = confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Confusion Matrix Analysis', fontsize=14, fontweight='bold')

    # Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[0], linewidths=0.5, cbar_kws={'shrink': 0.8})
    axes[0].set_title('Confusion Matrix (Counts)',      fontweight='bold')
    axes[0].set_ylabel('True Label',                    fontweight='bold')
    axes[0].set_xlabel('Predicted Label',               fontweight='bold')

    # Percentages
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=axes[1], linewidths=0.5, cbar_kws={'shrink': 0.8})
    axes[1].set_title('Confusion Matrix (Percentages)', fontweight='bold')
    axes[1].set_ylabel('True Label',                    fontweight='bold')
    axes[1].set_xlabel('Predicted Label',               fontweight='bold')

    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, 'confusion_matrix.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f" Saved: {path}")

    plt.show()
    return fig



def plot_roc_curve(y_true, y_prob, save=True):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=1.5,
            linestyle='--', label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.1, color='darkorange')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)',       fontsize=12)
    ax.set_title(f'ROC Curve — AUC = {roc_auc:.4f}',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save:
        path = os.path.join(FIGURES_DIR, 'roc_curve.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f" Saved: {path}")

    plt.show()
    print(f"AUC Score: {roc_auc:.4f}")
    return fig, roc_auc



def save_metrics_to_csv(metrics_dict, filename='metrics_report.csv'):
    path = os.path.join(RESULTS_DIR, filename)
    df   = pd.DataFrame([metrics_dict])
    df.to_csv(path, index=False)
    print(f" Saved: {path}")
    return df



def evaluate_model(model, test_gen=None):

    print("\n" + "="*60)
    print("         FULL MODEL EVALUATION")
    print("="*60)

    if test_gen is None:
        print("Loading VGG test generator...")
        _, test_gen, _,_ = create_vgg_generators()

    y_true, y_pred, y_prob = get_predictions(model, test_gen)

    metrics = print_classification_report(y_true, y_pred)

    plot_confusion_matrix(y_true, y_pred)

    _, auc_score = plot_roc_curve(y_true, y_prob)
    metrics['auc'] = auc_score

    save_metrics_to_csv(metrics)

    print("\n Evaluation complete!")
    return metrics, y_true, y_pred, y_prob


if __name__ == "__main__":
    print("Evaluate module loaded successfully")