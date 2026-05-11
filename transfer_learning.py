import os, sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *


import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_vgg_generators():


    normal_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    pneumonia_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    base = PROC_DIR if os.path.exists(
        os.path.join(PROC_DIR, 'train')) else DATA_DIR

    train_dir = os.path.join(base, 'train')
    test_dir  = os.path.join(base, 'test')
    val_dir   = os.path.join(base, 'val')
    if not os.path.exists(val_dir):
        val_dir = test_dir

    normal_gen = normal_datagen.flow_from_directory(
        train_dir,
        target_size=VGG_IMG_SIZE,
        color_mode='rgb',
        batch_size=BATCH_SIZE // 2,     
        class_mode='binary',
        classes=['NORMAL'],               
        shuffle=True,
        seed=42
    )

    pneumonia_gen = pneumonia_datagen.flow_from_directory(
        train_dir,
        target_size=VGG_IMG_SIZE,
        color_mode='rgb',
        batch_size=BATCH_SIZE // 2,         
        class_mode='binary',
        classes=['PNEUMONIA'],          
        shuffle=True,
        seed=42
    )

    def combined_generator(gen1, gen2):

        normal_count    = gen1.samples       # 1341
        pneumonia_count = gen2.samples       # 3875
        total           = normal_count + pneumonia_count

        weight_normal    = total / (2 * normal_count)     # ~1.94
        weight_pneumonia = total / (2 * pneumonia_count)  # ~0.67

        print(f"   Weight NORMAL    : {weight_normal:.3f}")
        print(f"   Weight PNEUMONIA : {weight_pneumonia:.3f}")

        while True:
            x1, _ = next(gen1)
            y1     = np.zeros(len(x1))                   
            w1     = np.full(len(x1), weight_normal)     

            x2, _ = next(gen2)
            y2     = np.ones(len(x2))                    
            w2     = np.full(len(x2), weight_pneumonia)   

            x = np.concatenate([x1, x2], axis=0)
            y = np.concatenate([y1, y2], axis=0)
            w = np.concatenate([w1, w2], axis=0)

            idx = np.random.permutation(len(x))

            yield x[idx], y[idx], w[idx]                  

    train_gen = combined_generator(normal_gen, pneumonia_gen)

    test_gen = eval_datagen.flow_from_directory(
        test_dir,
        target_size=VGG_IMG_SIZE,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    val_gen = eval_datagen.flow_from_directory(
        val_dir,
        target_size=VGG_IMG_SIZE,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )


    normal_samples    = normal_gen.samples        # 1341
    pneumonia_samples = pneumonia_gen.samples     # 3875
    total_samples     = normal_samples + pneumonia_samples
    steps_per_epoch   = total_samples // BATCH_SIZE

    print(f" VGG Generators ready")
    print(f"   NORMAL samples    : {normal_samples:,}")
    print(f"   PNEUMONIA samples : {pneumonia_samples:,}")
    print(f"   Steps per epoch   : {steps_per_epoch}")
    print(f"   Test              : {test_gen.samples:,}")

    return train_gen, test_gen, val_gen, steps_per_epoch



def build_vgg16_model(trainable_base=False):

    vgg_base = VGG16(
        weights='imagenet',
        include_top=False,
        input_shape=VGG_INPUT_SHAPE
    )
    vgg_base.trainable = trainable_base

    model = models.Sequential([
        vgg_base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ], name='VGG16_Pneumonia')

    return model, vgg_base



def plot_history(history, title="Training History", save=True):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    epochs_r = range(1, len(history.history['accuracy']) + 1)

    axes[0].plot(epochs_r, history.history['accuracy'],
                 'b-o', label='Train', lw=2)
    axes[0].plot(epochs_r, history.history['val_accuracy'],
                 'r-s', label='Val',   lw=2)
    axes[0].set_title('Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim([0, 1.05])

    axes[1].plot(epochs_r, history.history['loss'],
                 'b-o', label='Train', lw=2)
    axes[1].plot(epochs_r, history.history['val_loss'],
                 'r-s', label='Val',   lw=2)
    axes[1].set_title('Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    if save:
        safe_title = title.replace(' ', '_').replace('+', 'plus')
        path = os.path.join(FIGURES_DIR, f'history_{safe_title}.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        print(f" Saved: {path}")

    plt.show()
    return fig



def train_vgg16():


    train_gen, test_gen, val_gen , steps_per_epoch = create_vgg_generators()

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'vgg16_best.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    print("\n" + "="*55)
    print("  Phase 1: Training with FROZEN VGG16")
    print("="*55)

    model, vgg_base = build_vgg16_model(trainable_base=False)

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    model.summary()

    history_p1 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    plot_history(history_p1, "Phase 1 — Frozen VGG16")

    val_acc_p1 = max(history_p1.history['val_accuracy'])
    print(f"\n Phase 1 Best Val Accuracy: {val_acc_p1:.4f}")

   
    print("\n" + "="*55)
    print("  Phase 2: Fine-tuning last 4 VGG16 layers")
    print("="*55)

    vgg_base.trainable = True
    for layer in vgg_base.layers[:-4]:
        layer.trainable = False

    trainable_count = sum(1 for l in vgg_base.layers if l.trainable)
    print(f"   Trainable VGG layers: {trainable_count}")

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    early_stop_ft = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint_ft = ModelCheckpoint(
        filepath=os.path.join(MODEL_DIR, 'vgg16_finetuned.h5'),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    history_p2 = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        epochs=10,
        callbacks=[early_stop_ft, checkpoint_ft],
        verbose=1
    )

    plot_history(history_p2, "Phase 2 — Fine-tuning")

    val_acc_p2 = max(history_p2.history['val_accuracy'])
    print(f"\n Phase 2 Best Val Accuracy: {val_acc_p2:.4f}")

    print("\n" + "="*55)
    print("  FINAL RESULTS")
    print("="*55)
    print(f"   Phase 1 Best Accuracy: {val_acc_p1:.4f} ({val_acc_p1*100:.1f}%)")
    print(f"   Phase 2 Best Accuracy: {val_acc_p2:.4f} ({val_acc_p2*100:.1f}%)")
    improvement = (val_acc_p2 - val_acc_p1) * 100
    print(f"   Improvement:           {improvement:+.1f}%")
    print("="*55)

    return model, history_p1, history_p2, test_gen



if __name__ == "__main__":
    model, h1, h2, test_gen = train_vgg16()