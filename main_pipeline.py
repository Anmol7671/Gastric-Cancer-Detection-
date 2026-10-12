import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.layers import (
    Input, UpSampling2D, Concatenate, Conv2D, BatchNormalization,
    Activation, Dropout, GlobalAveragePooling2D, Dense, Multiply,
    RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomBrightness
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Recall, Precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecay
import matplotlib
# Keep Matplotlib's normal interactive desktop backend.  The previous forced
# "Agg" backend can only write files, so plt.show() could never open a window.
import matplotlib.pyplot as plt
from glob import glob

from PIL import Image
import tensorflow as tf
from ultralytics import YOLO

#CONFIG
RETRAIN_SEGMENTATION = False
RETRAIN_YOLO         = False

# All project paths are derived from this script's location. The complete
# directory can therefore be copied to another drive or computer unchanged.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#Dataset paths (updated to new 2188-image pre-split dataset)
SEG_ROOT    = os.path.join(BASE_DIR, "segmentation_dataset")
YOLO_ROOT   = os.path.join(BASE_DIR, "yolo Dataset")
CLASS_NAMES = ["gastric_lesion"]

#Output paths
PREPROCESSED_DIR     = os.path.join(BASE_DIR, "preprocessed")
NPY_DIR              = os.path.join(BASE_DIR, "npy_files")
RESULTS_DIR          = os.path.join(BASE_DIR, "results_transmobile")
COMPARISON_CSV       = os.path.join(BASE_DIR, "Book1.csv")
COMPARISON_DIR       = os.path.join(BASE_DIR, "comparison_charts")
SEG_MODEL_PATH       = os.path.join(BASE_DIR, "transmobile_seg_model.keras")
YOLO_RUNS_DIR        = os.path.join(BASE_DIR, "runs_yolo")
YOLO_BEST_PATH_CACHE = os.path.join(YOLO_RUNS_DIR, "best_path.txt")

#Segmentation hyperparameters
IMAGE_SIZE = 256      
BATCH      = 8          
EPOCHS     = 100
SMOOTH     = 1e-15

#YOLO hyperparameters
YOLO_BASE_MODEL = os.path.join(BASE_DIR, "yolov8n.pt")
YOLO_IMGSZ      = 512   
YOLO_EPOCHS     = 100
YOLO_BATCH      = 8    
YOLO_PATIENCE   = 15
# Detection shares this process with TensorFlow. CPU inference avoids CUDA
# context/device-loss errors on 6 GB laptop GPUs; YOLO training remains auto.
YOLO_INFERENCE_DEVICE = "cpu"

#Detection evaluation thresholds
DET_CONF_THRESHOLD = 0.3
DET_IOU_THRESHOLD  = 0.5

#Demo / display flags
NUM_DEMO_IMAGES            = 5
SHOW_PREPROCESSING_SAMPLES = True
SHOW_AUGMENTATION_SAMPLES  = True
SHOW_REGION_DETECTED       = True
DISPLAY_VISUALS            = True

# Create output directories
for _d in [PREPROCESSED_DIR, NPY_DIR, RESULTS_DIR, COMPARISON_DIR, YOLO_RUNS_DIR]:
    os.makedirs(_d, exist_ok=True)

#  Data loading — reads PRE-SPLIT folders directly
def load_split(split_name):
    img_dir  = os.path.join(SEG_ROOT, split_name, "images")
    mask_dir = os.path.join(SEG_ROOT, split_name, "masks")

    if not os.path.exists(img_dir):
        raise FileNotFoundError(f"Images dir not found: {img_dir}")
    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Masks dir not found: {mask_dir}")

    # Match by stem to avoid ordering mismatches
    img_map  = {os.path.splitext(f)[0]: os.path.join(img_dir, f)
                for f in os.listdir(img_dir)
                if not f.startswith('.') and f.lower().endswith(
                    ('.jpg', '.jpeg', '.png', '.bmp'))}
    mask_map = {os.path.splitext(f)[0]: os.path.join(mask_dir, f)
                for f in os.listdir(mask_dir)
                if not f.startswith('.') and f.lower().endswith(
                    ('.jpg', '.jpeg', '.png', '.bmp'))}

    common = sorted(img_map.keys() & mask_map.keys())

    if not common:
        raise RuntimeError(
            f"No matched image/mask pairs in {img_dir} / {mask_dir}. "
            f"Check filenames match (e.g. img001.jpg <-> img001.png)."
        )

    missing_masks  = sorted(img_map.keys()  - mask_map.keys())
    missing_images = sorted(mask_map.keys() - img_map.keys())
    if missing_masks:
        print(f"[WARNING] {len(missing_masks)} images without mask in {split_name} — skipped.")
    if missing_images:
        print(f"[WARNING] {len(missing_images)} masks without image in {split_name} — skipped.")

    image_paths = [img_map[s]  for s in common]
    mask_paths  = [mask_map[s] for s in common]

    print(f"[INFO] {split_name:5s}: {len(image_paths)} matched pairs  "
          f"({img_dir})")
    return image_paths, mask_paths


def load_data():
    train_imgs,  train_masks  = load_split("train")
    val_imgs,    val_masks    = load_split("val")
    test_imgs,   test_masks   = load_split("test")

    print(f"\n[INFO] Dataset summary:")
    print(f"  Train : {len(train_imgs):4d} pairs")
    print(f"  Val   : {len(val_imgs):4d} pairs")
    print(f"  Test  : {len(test_imgs):4d} pairs")
    print(f"  Total : {len(train_imgs)+len(val_imgs)+len(test_imgs):4d} pairs\n")

    return (train_imgs, train_masks), (val_imgs, val_masks), (test_imgs, test_masks)

#  Preprocessing
def delete_yolo_caches():
    labels_root = os.path.join(YOLO_ROOT, "labels")
    removed = 0
    for cache in glob(os.path.join(labels_root, "*.cache")):
        os.remove(cache)
        removed += 1
        print(f"  Removed stale cache: {cache}")
    if removed == 0:
        print("  No stale .cache files found.")
    return removed


def dice_coef(y_true, y_pred):
    y_true_f     = tf.keras.layers.Flatten()(y_true)
    y_pred_f     = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + SMOOTH) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + SMOOTH
    )


def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean(bce) + dice_loss(y_true, y_pred)
def apply_unsharp_clahe(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to load image: {image_path}")
    gaussian      = cv2.GaussianBlur(image, (9, 9), 10.0)
    unsharp_image = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
    lab           = cv2.cvtColor(unsharp_image, cv2.COLOR_BGR2LAB)
    l, a, b       = cv2.split(lab)
    clahe         = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl            = clahe.apply(l)
    updated_lab   = cv2.merge((cl, a, b))
    return cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)


def preprocess_and_cache(image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_paths = []
    for img_path in image_paths:
        out_path = os.path.join(output_dir, os.path.basename(img_path))
        if not os.path.exists(out_path):
            processed = apply_unsharp_clahe(img_path)
            cv2.imwrite(out_path, processed)
        output_paths.append(out_path)
    return output_paths


def read_image(path):
    path = path.decode() if isinstance(path, bytes) else path
    x    = cv2.imread(path, cv2.IMREAD_COLOR)
    x    = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x    = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
    return (x / 255.0).astype(np.float32)


def read_mask(path):
    path = path.decode() if isinstance(path, bytes) else path
    x    = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x    = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    x    = np.expand_dims(x / 255.0, axis=-1)
    return x.astype(np.float32)

def tf_parse(x, y):
    def _parse(x, y):
        return read_image(x), read_mask(y)
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([IMAGE_SIZE, IMAGE_SIZE, 3])
    y.set_shape([IMAGE_SIZE, IMAGE_SIZE, 1])
    return x, y


def build_spatial_augmentation_layer():
    return tf.keras.Sequential([
        RandomFlip("horizontal_and_vertical"),
        RandomRotation(factor=0.055),
        RandomZoom(height_factor=(-0.1, 0.1)),
        RandomTranslation(height_factor=0.1, width_factor=0.1),
    ], name="spatial_augmentation")


def build_brightness_layer():
    return RandomBrightness(factor=0.15, value_range=(0.0, 1.0))


def tf_dataset(x, y, batch=BATCH, augment=False, training=True):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        spatial_layer    = build_spatial_augmentation_layer()
        brightness_layer = build_brightness_layer()
        def apply_augmentation(image, mask):
            combined  = tf.concat([image, mask], axis=-1)
            combined  = spatial_layer(combined, training=True)
            image_aug = combined[:, :, :3]
            mask_aug  = tf.clip_by_value(combined[:, :, 3:], 0.0, 1.0)
            image_aug = brightness_layer(image_aug, training=True)
            return image_aug, mask_aug
        dataset = dataset.map(apply_augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

def build_test_arrays(test_image_paths, test_mask_paths, cache_prefix="seg_test"):
    images_cache = os.path.join(NPY_DIR, f"{cache_prefix}_images.npy")
    masks_cache  = os.path.join(NPY_DIR, f"{cache_prefix}_masks.npy")

    if os.path.exists(images_cache) and os.path.exists(masks_cache) \
            and not RETRAIN_SEGMENTATION:
        arr = np.load(images_cache)
        count_matches = len(arr) == len(test_image_paths)
        size_matches  = arr.shape[1:3] == (IMAGE_SIZE, IMAGE_SIZE)
        if count_matches and size_matches:
            return arr, np.load(masks_cache)
        print(f"Test array cache is stale (count_matches={count_matches}, "
              f"size_matches={size_matches}), rebuilding...")

    test_x = np.stack([read_image(p) for p in test_image_paths])
    test_y = np.stack([read_mask(p)  for p in test_mask_paths])
    np.save(images_cache, test_x)
    np.save(masks_cache,  test_y)
    return test_x, test_y

#  Segmentation model: MobileNetV2 + SE blocks + UNet decoder
def se_block(input_tensor, reduction=8):
    channels = input_tensor.shape[-1]
    se = GlobalAveragePooling2D()(input_tensor)
    se = Dense(max(1, channels // reduction), activation="relu")(se)
    se = Dense(channels, activation="sigmoid")(se)
    return Multiply()([input_tensor, se])


def create_model_with_se():
    inputs  = Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")
    encoder = MobileNetV2(
        input_tensor=inputs, weights="imagenet",
        include_top=False, alpha=0.35
    )

    skip_connection_names = [
        "input_image", "block_1_expand_relu",
        "block_3_expand_relu", "block_6_expand_relu"
    ]
    encoder_output = encoder.get_layer("block_13_expand_relu").output

    filters = [16, 32, 48, 64]
    x = se_block(encoder_output)

    for i in range(1, len(skip_connection_names) + 1):
        x_skip = encoder.get_layer(skip_connection_names[-i]).output
        x = UpSampling2D((2, 2))(x)
        x = Concatenate()([x, x_skip])
        x = Conv2D(filters[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Conv2D(filters[-i], (3, 3), padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(0.2)(x)

    x = Conv2D(1, (1, 1), padding="same")(x)
    x = Activation("sigmoid")(x)
    return Model(inputs, x)


def train_segmentation_model(train_dataset, valid_dataset, train_steps, valid_steps):
    model       = create_model_with_se()
    decay_steps = train_steps * EPOCHS
    lr_schedule = CosineDecay(
        initial_learning_rate=1e-3, decay_steps=decay_steps, alpha=1e-5
    )
    model.compile(
        loss=bce_dice_loss,
        optimizer=Adam(learning_rate=lr_schedule),
        metrics=[dice_coef, Recall(), Precision()]
    )
    callbacks = [EarlyStopping(monitor="val_loss", patience=10,
                               restore_best_weights=True)]
    model.fit(
        train_dataset, validation_data=valid_dataset,
        epochs=EPOCHS, steps_per_epoch=train_steps,
        validation_steps=valid_steps, callbacks=callbacks
    )
    model.save(SEG_MODEL_PATH)
    return model

def find_best_threshold(seg_model, val_x, val_y, thresholds=np.arange(0.3, 0.71, 0.05)):
    val_preds = seg_model.predict(val_x)
    best_thresh, best_dice = 0.5, -1
    for t in thresholds:
        preds_bin = (val_preds > t).astype(np.uint8)
        metrics = compute_segmentation_metrics(val_y, preds_bin)
        if metrics["Dice"] > best_dice:
            best_dice, best_thresh = metrics["Dice"], t
    print(f"  Best threshold: {best_thresh:.2f} (val Dice={best_dice*100:.2f}%)")
    return best_thresh

#  Segmentation evaluation
def compute_segmentation_metrics(y_true, y_pred):
    y_true_bin = (y_true > 0.5).astype(np.uint8)
    y_pred_bin = (y_pred > 0.5).astype(np.uint8)

    dices       = []
    ious        = []
    precisions  = []
    recalls     = []
    specificities = []

    for i in range(len(y_true_bin)):
        t = y_true_bin[i].flatten()
        p = y_pred_bin[i].flatten()

        tp = np.logical_and(t == 1, p == 1).sum()
        fp = np.logical_and(t == 0, p == 1).sum()
        fn = np.logical_and(t == 1, p == 0).sum()
        tn = np.logical_and(t == 0, p == 0).sum()

        # IoU
        union = tp + fp + fn
        ious.append(1.0 if union == 0 else tp / union)

        # Dice
        denom = 2 * tp + fp + fn
        dices.append(1.0 if denom == 0 else (2.0 * tp) / denom)

        # Pixel-level Precision
        precisions.append(1.0 if (tp + fp) == 0 else tp / (tp + fp))

        # Pixel-level Recall (Sensitivity)
        recalls.append(1.0 if (tp + fn) == 0 else tp / (tp + fn))

        # Pixel-level Specificity
        specificities.append(1.0 if (tn + fp) == 0 else tn / (tn + fp))

    return {
        "Dice"       : float(np.mean(dices)),
        "IoU"        : float(np.mean(ious)),
        "Precision"  : float(np.mean(precisions)),
        "Recall"     : float(np.mean(recalls)),
        "Specificity": float(np.mean(specificities)),
    }


def plot_and_save_metrics(metrics, filename="segmentation_metrics.png"):
    labels     = list(metrics.keys())
    values_pct = [v * 100 for v in metrics.values()]
    colors     = ["Green", "Blue", "pink", "yellow", "cyan"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values_pct, width=0.5, color=colors[:len(labels)])
    ax.set_title("TransMobile-YOLO Segmentation Metrics", fontsize=14)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.xticks(fontsize=11)
    for bar, val in zip(bars, values_pct):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1.5,
                f"{val:.2f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=10)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, filename)
    plt.savefig(out, dpi=150)
    show_and_close(fig)
    print(f"  Segmentation chart saved -> {out}")

#  Visualizations
def show_and_close(fig):
    """Display a saved figure on screen, then release it after its window closes."""
    if DISPLAY_VISUALS:
        plt.show(block=True)
    plt.close(fig)


def show_preprocessing_samples(image_dir, num_images=5):
    files = sorted(os.listdir(image_dir))[:num_images]
    for file_name in files:
        img = cv2.imread(os.path.join(image_dir, file_name))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        axes[0].imshow(img);                         axes[0].set_title("Unsharp+CLAHE"); axes[0].axis("off")
        axes[1].imshow(img[:, :, 0], cmap="Reds");  axes[1].set_title("Red channel");   axes[1].axis("off")
        axes[2].imshow(img[:, :, 1], cmap="Greens");axes[2].set_title("Green channel"); axes[2].axis("off")
        axes[3].imshow(img[:, :, 2], cmap="Blues"); axes[3].set_title("Blue channel");  axes[3].axis("off")
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, f"preproc_{file_name}.png")
        plt.savefig(out, dpi=120)
        show_and_close(fig)
        print(f"  Saved: {out}")


def apply_single_transform(image, rotation_angle=0, scale=1.0,
                            shift_x=0, shift_y=0, zoom_factor=1.0):
    h, w    = image.shape[:2]
    M_rot   = cv2.getRotationMatrix2D((w//2, h//2), rotation_angle, scale)
    rotated = cv2.warpAffine(image, M_rot, (w, h))
    M_shift = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(rotated, M_shift, (w, h))
    if zoom_factor != 1.0:
        zh     = int(h / zoom_factor)
        zw     = int(w / zoom_factor)
        sy     = (h - zh) // 2
        sx     = (w - zw) // 2
        zoomed = shifted[sy:sy+zh, sx:sx+zw]
        zoomed = cv2.resize(zoomed, (w, h))
    else:
        zoomed = shifted
    return zoomed


def show_augmentation_samples(image_dir, mask_dir, num_images=5):
    files = sorted(os.listdir(image_dir))[:num_images]
    for file_name in files:
        img_path   = os.path.join(image_dir, file_name)
        base       = os.path.splitext(file_name)[0]
        mask_files = glob(os.path.join(mask_dir, base + ".*"))
        if not mask_files:
            continue
        mask_path = mask_files[0]

        img  = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            continue
        img  = cv2.cvtColor(cv2.resize(img,  (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))

        rotated_img  = apply_single_transform(img,  rotation_angle=45)
        rotated_mask = apply_single_transform(mask, rotation_angle=45)
        scaled_img   = apply_single_transform(img,  scale=0.8)
        scaled_mask  = apply_single_transform(mask, scale=0.8)
        shifted_img  = apply_single_transform(img,  shift_x=50, shift_y=50)
        shifted_mask = apply_single_transform(mask, shift_x=50, shift_y=50)
        zoomed_img   = apply_single_transform(img,  zoom_factor=1.5)
        zoomed_mask  = apply_single_transform(mask, zoom_factor=1.5)

        fig, axes = plt.subplots(2, 5, figsize=(18, 7))
        for ax, im, t in zip(axes[0],
            [img, rotated_img, scaled_img, shifted_img, zoomed_img],
            ["Original","Rotated 45°","Scaled 0.8×","Shifted 50px","Zoomed 1.5×"]):
            ax.imshow(im); ax.set_title(t); ax.axis("off")
        for ax, im, t in zip(axes[1],
            [mask, rotated_mask, scaled_mask, shifted_mask, zoomed_mask],
            ["GT Mask","Rotated Mask","Scaled Mask","Shifted Mask","Zoomed Mask"]):
            ax.imshow(im, cmap="gray"); ax.set_title(t); ax.axis("off")

        plt.suptitle("Image & Mask Augmentations", fontsize=13, fontweight="bold")
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, f"augmentation_{file_name}.png")
        plt.savefig(out, dpi=120)
        show_and_close(fig)
        print(f"  Saved: {out}")


def show_region_detected(seg_model, test_x, test_y, best_thresh=0.5, num_images=NUM_DEMO_IMAGES):
    num_images = min(num_images, len(test_x))
    preds      = (seg_model.predict(test_x[:num_images]) > best_thresh).astype(np.uint8)

    for i in range(num_images):
        original_uint8 = (test_x[i] * 255.0).astype(np.uint8)
        mask           = preds[i].squeeze()

        overlay = original_uint8.copy()
        overlay[mask > 0] = (255, 0, 0)
        blended = cv2.addWeighted(original_uint8, 0.5, overlay, 0.5, 0)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(original_uint8);              axes[0].set_title("Preprocessed Image"); axes[0].axis("off")
        axes[1].imshow(test_y[i].squeeze(), cmap="gray"); axes[1].set_title("Ground Truth Mask"); axes[1].axis("off")
        axes[2].imshow(preds[i].squeeze(), cmap="gray");  axes[2].set_title("Predicted Mask");    axes[2].axis("off")
        axes[3].imshow(blended);                     axes[3].set_title("Region Detected");   axes[3].axis("off")

        plt.suptitle(f"Segmentation Output — Image {i+1}", fontsize=13)
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, f"region_detected_{i}.png")
        plt.savefig(out, dpi=150)
        show_and_close(fig)
        print(f"  Saved: {out}")


def run_transmobile_yolo_demo(seg_model, yolo_model, test_x, test_y,
                               num_images=NUM_DEMO_IMAGES):
    num_images = min(num_images, len(test_x))
    preds      = (seg_model.predict(test_x[:num_images]) > 0.5).astype(np.uint8)

    for i in range(num_images):
        original_uint8 = (test_x[i] * 255.0).astype(np.uint8)
        mask           = preds[i].squeeze()

        overlay = original_uint8.copy()
        overlay[mask > 0] = (255, 0, 0)
        blended = cv2.addWeighted(original_uint8, 0.5, overlay, 0.5, 0)

        yolo_result = yolo_model.predict(
            source=Image.fromarray(blended), conf=0.3,
            device=YOLO_INFERENCE_DEVICE, verbose=False
        )[0]
        annotated = blended.copy()
        boxes     = yolo_result.boxes

        if boxes is not None and len(boxes) > 0:
            for box, conf in zip(
                boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()
            ):
                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated, f"{conf:.2f}",
                            (x1, max(y1-5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255), 1, cv2.LINE_AA)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        axes[0].imshow(original_uint8);              axes[0].set_title("Preprocessed Image");    axes[0].axis("off")
        axes[1].imshow(test_y[i].squeeze(), cmap="gray"); axes[1].set_title("Ground Truth Mask"); axes[1].axis("off")
        axes[2].imshow(blended);                     axes[2].set_title("Segmentation Overlay\n(Mask Branch)"); axes[2].axis("off")
        axes[3].imshow(annotated);                   axes[3].set_title("TransMobile-YOLO Detection"); axes[3].axis("off")

        plt.suptitle(f"Full Pipeline Output — Image {i+1}", fontsize=13)
        plt.tight_layout()
        out = os.path.join(RESULTS_DIR, f"demo_{i}.png")
        plt.savefig(out, dpi=150)
        show_and_close(fig)
        print(f"  Saved: {out}")

#  YOLO — data.yaml + training + evaluation
def write_data_yaml():
    yaml_path    = os.path.join(YOLO_ROOT, "data.yaml")
    names_block  = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASS_NAMES))
    content = (
        f"path: {os.path.abspath(YOLO_ROOT)}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_block}\n"
    )
    with open(yaml_path, "w") as f:
        f.write(content)
    print(f"  data.yaml written -> {yaml_path}")
    return yaml_path


def train_yolo_detector():
    data_yaml  = write_data_yaml()
    yolo_model = YOLO(YOLO_BASE_MODEL)
    results    = yolo_model.train(
        data=data_yaml,
        epochs=YOLO_EPOCHS,
        imgsz=YOLO_IMGSZ,
        batch=YOLO_BATCH,
        patience=YOLO_PATIENCE,
        project=YOLO_RUNS_DIR,
        name="train",
        exist_ok=True,
        workers=0,        
    )
    best_pt = os.path.join(str(results.save_dir), "weights", "best.pt")
    with open(YOLO_BEST_PATH_CACHE, "w") as f:
        f.write(os.path.relpath(best_pt, BASE_DIR))
    print(f"  YOLO training complete. Best weights: {best_pt}")
    return best_pt


def get_or_train_yolo():
    if not RETRAIN_YOLO:
        candidates = []
        if os.path.exists(YOLO_BEST_PATH_CACHE):
            with open(YOLO_BEST_PATH_CACHE) as f:
                cached = f.read().strip()
            if cached:
                candidates.append(
                    cached if os.path.isabs(cached)
                    else os.path.join(BASE_DIR, cached)
                )

        # Do not rely only on best_path.txt: it may contain a path copied from
        # another computer/drive even though the weights exist in this project.
        candidates.extend([
            os.path.join(YOLO_RUNS_DIR, "train", "weights", "best.pt"),
            os.path.join(YOLO_RUNS_DIR, "train", "weights", "last.pt"),
        ])
        candidates.extend(sorted(
            glob(os.path.join(YOLO_RUNS_DIR, "**", "weights", "best.pt")),
            key=os.path.getmtime,
            reverse=True,
        ))

        for weights_path in candidates:
            if weights_path and os.path.isfile(weights_path):
                weights_path = os.path.abspath(weights_path)
                with open(YOLO_BEST_PATH_CACHE, "w") as f:
                    f.write(os.path.relpath(weights_path, BASE_DIR))
                print(f"  Loaded saved YOLO weights (training skipped): {weights_path}")
                return weights_path

        print("  No saved YOLO weights found — starting detector training...")
    return train_yolo_detector()


def yolo_box_iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1 = max(xa1, xb1); iy1 = max(ya1, yb1)
    ix2 = min(xa2, xb2); iy2 = min(ya2, yb2)
    iw  = max(0.0, ix2-ix1); ih = max(0.0, iy2-iy1)
    inter = iw * ih
    area_a = max(0.0, xa2-xa1) * max(0.0, ya2-ya1)
    area_b = max(0.0, xb2-xb1) * max(0.0, yb2-yb1)
    union  = area_a + area_b - inter
    return 0.0 if union <= 0 else inter / union


def load_yolo_label_boxes(label_path, img_w, img_h):
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, w, h = parts[:5]
            cx, cy, w, h = float(cx), float(cy), float(w), float(h)
            x1 = (cx - w/2) * img_w; y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w; y2 = (cy + h/2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


def evaluate_yolo_detector(yolo_model, data_yaml):
    images_dir = os.path.join(YOLO_ROOT, "images", "test")
    labels_dir = os.path.join(YOLO_ROOT, "labels", "test")

    # AP via Ultralytics built-in val()
    delete_yolo_caches()
    val_res = yolo_model.val(
        data=data_yaml, split="test", device=YOLO_INFERENCE_DEVICE,
        batch=YOLO_BATCH, verbose=False
    )
    ap50    = float(val_res.box.map50)
    # Image-level TP/FP/FN/TN
    image_paths = sorted(glob(os.path.join(images_dir, "*.*")))
    if not image_paths:
        raise FileNotFoundError(f"No test images in {images_dir}")

    TP = FP = FN = TN = 0
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        base       = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base + ".txt")
        gt_boxes   = load_yolo_label_boxes(label_path, w, h)
        has_gt     = len(gt_boxes) > 0

        result     = yolo_model.predict(
            source=img_path, conf=DET_CONF_THRESHOLD,
            device=YOLO_INFERENCE_DEVICE, verbose=False
        )[0]
        pred_boxes = (result.boxes.xyxy.cpu().numpy().tolist()
                      if result.boxes is not None and len(result.boxes) > 0 else [])
        has_pred   = len(pred_boxes) > 0

        matched = False
        if has_gt and has_pred:
            for pb in pred_boxes:
                for gb in gt_boxes:
                    if yolo_box_iou(pb, gb) >= DET_IOU_THRESHOLD:
                        matched = True
                        break
                if matched:
                    break

        if   has_gt  and matched:      TP += 1
        elif has_gt  and not matched:  FN += 1
        elif not has_gt and has_pred:  FP += 1
        else:                          TN += 1

    total    = TP + TN + FP + FN
    accuracy = (TP + TN) / total if total else 0.0
    recall   = TP / (TP + FN)   if (TP + FN) else 0.0
    prec_int = TP / (TP + FP)   if (TP + FP) else 1.0
    f1       = (2 * prec_int * recall / (prec_int + recall)
                if (prec_int + recall) else 0.0)
    err_rate = 1.0 - accuracy

    print(f"\n  Detection confusion (image-level): TP={TP} TN={TN} FP={FP} FN={FN}")

    return {
        "Accuracy"      : accuracy,
        "Recall"        : recall,
        "F1 Score"      : f1,
        "Error Rate"    : err_rate,
        "AP@0.5"        : ap50,}


def plot_and_save_detection_metrics(metrics, filename="detection_metrics.png"):
    labels     = list(metrics.keys())
    values_pct = [v * 100 for v in metrics.values()]
    colors     = ["green", "blue", "pink", "red", "orange"]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(labels, values_pct, width=0.55, color=colors[:len(labels)])
    ax.set_title("TransMobile-YOLO Detection Metrics", fontsize=14)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.xticks(fontsize=11)
    for bar, val in zip(bars, values_pct):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1.5,
                f"{val:.2f}%", ha="center", va="bottom",
                fontweight="bold", fontsize=10)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, filename)
    plt.savefig(out, dpi=150)
    show_and_close(fig)
    print(f"  Detection chart saved -> {out}")
    
def run_model_comparison_section():
    if not os.path.isfile(COMPARISON_CSV):
        print(f"  [WARNING] Comparison CSV not found; skipped: {COMPARISON_CSV}")
        return []

    df = pd.read_csv(COMPARISON_CSV)
    df.columns = [column.strip() for column in df.columns]
    model_col = "models"
    if model_col not in df.columns:
        raise ValueError(
            f"Comparison CSV must contain a '{model_col}' column; "
            f"found {list(df.columns)}"
        )

    model_colors = {
        "TransMobile": "blue",
        "SwinTrans": "green",
        "IM RCNN": "pink",
    }
    default_colors = ["blue", "green", "orange", "brown", "gold"]
    lower_is_better = {"error-rate", "error_rate", "errorrate"}
    models = df[model_col].astype(str).tolist()
    saved_paths = []

    for metric_col in (column for column in df.columns if column != model_col):
        values = pd.to_numeric(df[metric_col], errors="raise").tolist()
        colors = [
            model_colors.get(name, default_colors[i % len(default_colors)])
            for i, name in enumerate(models)
        ]
        is_lower_better = metric_col.strip().lower() in lower_is_better
        if is_lower_better:
            colors = ["red"] * len(models)

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(models, values, color=colors, width=0.5)
        y_max = max(values) if values else 1.0
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(y_max * 0.015, 0.1),
                f"{value:.2f}%",
                ha="center", va="bottom", fontsize=11, fontweight="bold",
            )

        suffix = " (lower = better)" if is_lower_better else ""
        ax.set_title(f"{metric_col}{suffix}", fontsize=14, fontweight="bold")
        ax.set_ylabel("Score (%)", fontsize=12)
        ax.set_ylim(0, max(y_max * 1.18, 1.0))
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        plt.xticks(fontsize=11, rotation=10)
        plt.tight_layout()

        safe_name = metric_col.replace("/", "_").replace(" ", "_").replace("@", "at")
        out_path = os.path.join(COMPARISON_DIR, f"comparison_{safe_name}.png")
        plt.savefig(out_path, dpi=150)
        show_and_close(fig)
        saved_paths.append(out_path)
        print(f"  Comparison chart saved -> {out_path}")

    return saved_paths

#MAIN
def evaluate_with_multiple_seeds(seg_model, test_x, test_y, best_thresh,
                                  seeds=(42, 123, 7), sample_frac=1.0):
    all_runs = []
    n = len(test_x)
    sample_n = max(1, int(n * sample_frac))

    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=sample_n, replace=False)
        preds = (seg_model.predict(test_x[idx]) > best_thresh).astype(np.uint8)
        m = compute_segmentation_metrics(test_y[idx], preds)
        all_runs.append(m)
        print(f"    Seed {seed:4d} → "
              + "  ".join(f"{k}: {v*100:.2f}%" for k, v in m.items()))

    metric_keys = all_runs[0].keys()
    stability = {}
    for k in metric_keys:
        vals = np.array([r[k] for r in all_runs])
        stability[k] = (float(np.mean(vals)), float(np.std(vals)))

    print("\n  ── Stability (mean ± std across seeds) ──")
    for k, (mean, std) in stability.items():
        print(f"    {k:12s}: {mean*100:.2f}% ± {std*100:.2f}%")

    # Save stability results to CSV
    rows = [{"Metric": k, "Mean (%)": f"{m*100:.2f}", "Std (%)": f"{s*100:.2f}"}
            for k, (m, s) in stability.items()]
    csv_path = os.path.join(RESULTS_DIR, "segmentation_stability.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Stability CSV saved -> {csv_path}")

    return stability


if __name__ == "__main__":

    print("  TransMobile-YOLO — Gastric Cancer Detection Pipeline")
    print("  Dataset: 2188 images (70/15/15 split, 512×512)")

    #1. Load pre-split dataset
    print("[1/7] Loading pre-split segmentation dataset...")
    (train_imgs, train_masks), (val_imgs, val_masks), \
        (test_imgs, test_masks) = load_data()

    #2. Preprocess (Unsharp + CLAHE)
    print("[2/7] Applying Unsharp Masking + CLAHE preprocessing...")
    train_imgs_pp = preprocess_and_cache(
        train_imgs, os.path.join(PREPROCESSED_DIR, "train"))
    val_imgs_pp   = preprocess_and_cache(
        val_imgs,   os.path.join(PREPROCESSED_DIR, "val"))
    test_imgs_pp  = preprocess_and_cache(
        test_imgs,  os.path.join(PREPROCESSED_DIR, "test"))

    #3. Visualizations (preprocessing + augmentation)
    if SHOW_PREPROCESSING_SAMPLES:
        print("\n[3a] Saving preprocessing samples...")
        show_preprocessing_samples(
            os.path.join(PREPROCESSED_DIR, "test"), num_images=5)

    if SHOW_AUGMENTATION_SAMPLES:
        print("\n[3b] Saving augmentation samples...")
        show_augmentation_samples(
            os.path.join(SEG_ROOT, "train", "images"),
            os.path.join(SEG_ROOT, "train", "masks"),
            num_images=5)

    # 4. Segmentation model
    print("\n[4/7] Segmentation model (MobileNetV2 + SE + UNet)...")
    train_dataset = tf_dataset(train_imgs_pp, train_masks, batch=BATCH, augment=True,  training=False)
    valid_dataset = tf_dataset(val_imgs_pp,   val_masks,   batch=BATCH, augment=False, training=False)
    train_steps   = max(1, len(train_imgs_pp) // BATCH)
    valid_steps   = max(1, len(val_imgs_pp)   // BATCH)

    if RETRAIN_SEGMENTATION or not os.path.exists(SEG_MODEL_PATH):
        print("  Training segmentation model from scratch...")
        seg_model = train_segmentation_model(
            train_dataset, valid_dataset, train_steps, valid_steps)
    else:
        seg_model = load_model(
            SEG_MODEL_PATH,
            custom_objects={"dice_loss": dice_loss, "dice_coef": dice_coef,"bce_dice_loss": bce_dice_loss})
        print(f"  Loaded cached model: {SEG_MODEL_PATH}")

    val_x, val_y = build_test_arrays(val_imgs_pp, val_masks, cache_prefix="seg_val")
    best_thresh  = find_best_threshold(seg_model, val_x, val_y)

    print("\n  Evaluating segmentation on test set (329 images)...")
    test_x, test_y = build_test_arrays(test_imgs_pp, test_masks)
    predictions    = (seg_model.predict(test_x) > best_thresh).astype(np.uint8)
    seg_metrics    = compute_segmentation_metrics(test_y, predictions)

    print("\n  ── Segmentation Results ──")
    for k, v in seg_metrics.items():
        print(f"    {k}: {v*100:.2f}%")
    plot_and_save_metrics(seg_metrics)

    # Multi-seed stability evaluation
    print("\n  ── Segmentation Stability (multi-seed) ──")
    seg_stability = evaluate_with_multiple_seeds(
        seg_model, test_x, test_y, best_thresh, seeds=(42, 123, 7)
    )

    if SHOW_REGION_DETECTED:
        print("\n  Saving region detected visualizations...")
        show_region_detected(seg_model, test_x, test_y,
                             best_thresh=best_thresh, num_images=NUM_DEMO_IMAGES)

    #5. YOLO detection
    print("\n[5/7] YOLOv8n detection (yolo Dataset, 512×512)...")
    delete_yolo_caches()
    best_pt    = get_or_train_yolo()
    yolo_model = YOLO(best_pt)
    data_yaml  = write_data_yaml()
    print(f"  Detection inference device: {YOLO_INFERENCE_DEVICE}")

    det_metrics = evaluate_yolo_detector(yolo_model, data_yaml)

    print("\n  ── Detection Results ──")
    for k, v in det_metrics.items():
        print(f"    {k}: {v*100:.2f}%")
    plot_and_save_detection_metrics(det_metrics)

    #6. Full pipeline demo
    print("\n[6/7] Running full TransMobile-YOLO pipeline demo...")
    run_transmobile_yolo_demo(
        seg_model, yolo_model, test_x, test_y, num_images=NUM_DEMO_IMAGES)

    #7. Model comparison charts are deliberately last in the visual sequence.
    print("\n[7/7] Generating model comparison visualizations...")
    run_model_comparison_section()

    #Final summary
    print("  FINAL RESULTS — TransMobile-YOLO")
    print(f"  Segmentation (test set):")
    for k, v in seg_metrics.items():
        mean, std = seg_stability.get(k, (v, 0.0))
        print(f"    {k:12s}: {v*100:.2f}%  (mean±std: {mean*100:.2f}% ± {std*100:.2f}%)")
    print(f"  Detection:")
    for k, v in det_metrics.items():
        print(f"    {k:12s}: {v*100:.2f}%")
    print(f"\n  All outputs saved to: {os.path.abspath(RESULTS_DIR)}")

    # Save complete results to CSV for paper table
    rows = []
    for k, v in seg_metrics.items():
        mean, std = seg_stability.get(k, (v, 0.0))
        rows.append({"Task": "Segmentation", "Metric": k,
                     "Value (%)": f"{v*100:.2f}",
                     "Mean±Std": f"{mean*100:.2f}% ± {std*100:.2f}%"})
    for k, v in det_metrics.items():
        rows.append({"Task": "Detection", "Metric": k,
                     "Value (%)": f"{v*100:.2f}", "Mean±Std": "—"})
    csv_path = os.path.join(RESULTS_DIR, "final_results.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  Final results CSV saved -> {csv_path}")
    print("  Pipeline completed successfully!\n")
