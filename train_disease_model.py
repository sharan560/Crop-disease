import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def class_directories(root: Path, ignore_names: Sequence[str] = ()) -> List[Path]:
    ignore_set = set(ignore_names)
    dirs: List[Path] = []
    for item in sorted(root.iterdir()):
        if not item.is_dir() or item.name in ignore_set:
            continue
        has_image = any(
            Path(file_name).suffix.lower() in IMAGE_EXTENSIONS
            for _, _, files in os.walk(item)
            for file_name in files
        )
        if has_image:
            dirs.append(item)
    return dirs


def detect_dataset_root(base_dir: Path) -> Tuple[Path, str]:
    nested = base_dir / "PlantVillage"
    outer_classes = class_directories(base_dir, ignore_names=("PlantVillage",))
    nested_classes = class_directories(nested) if nested.exists() else []

    outer_names = {p.name for p in outer_classes}
    nested_names = {p.name for p in nested_classes}

    if nested_classes and outer_names and nested_names == outer_names:
        return nested, (
            "Detected duplicated wrapper folder. Using nested dataset root "
            f"{nested} to avoid double-counting images."
        )

    if outer_classes:
        return base_dir, f"Using dataset root: {base_dir}"

    if nested_classes:
        return nested, f"Using nested dataset root: {nested}"

    raise ValueError(f"No class directories with images found under {base_dir}")


def collect_samples(dataset_root: Path) -> Tuple[List[str], List[int], List[str]]:
    class_dirs = class_directories(dataset_root)
    if not class_dirs:
        raise ValueError(f"No valid class folders found in {dataset_root}")

    class_names = [p.name for p in class_dirs]
    label_map: Dict[str, int] = {name: idx for idx, name in enumerate(class_names)}

    image_paths: List[str] = []
    labels: List[int] = []

    for class_dir in class_dirs:
        label = label_map[class_dir.name]
        for root, _, files in os.walk(class_dir):
            for file_name in files:
                if Path(file_name).suffix.lower() in IMAGE_EXTENSIONS:
                    image_paths.append(str(Path(root) / file_name))
                    labels.append(label)

    return image_paths, labels, class_names


def load_and_preprocess(path: tf.Tensor, label: tf.Tensor, image_size: int) -> Tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size])
    image = preprocess_input(tf.cast(image, tf.float32))
    return image, label


def make_dataset(paths: Sequence[str], labels: Sequence[int], image_size: int, batch_size: int, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((list(paths), list(labels)))
    if training:
        ds = ds.shuffle(buffer_size=min(len(paths), 10000), reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, y: load_and_preprocess(p, y, image_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(
    num_classes: int,
    image_size: int,
    dropout_rate: float,
    base_trainable: bool = False,
) -> Tuple[tf.keras.Model, tf.keras.Model]:
    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.2),
            layers.RandomContrast(0.15),
            layers.RandomBrightness(0.1),
        ],
        name="augmentation",
    )

    inputs = layers.Input(shape=(image_size, image_size, 3), name="input_image")
    x = augmentation(inputs)

    base_model = MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = base_trainable

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="plant_disease_classifier")
    return model, base_model


def make_callbacks(checkpoint_path: Path, early_patience: int) -> List[tf.keras.callbacks.Callback]:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=max(1, early_patience - 1),
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def merge_histories(*histories: tf.keras.callbacks.History) -> Dict[str, List[float]]:
    merged: Dict[str, List[float]] = {}
    for history in histories:
        for metric, values in history.history.items():
            merged.setdefault(metric, []).extend(float(v) for v in values)
    return merged


def evaluate_detailed(model: tf.keras.Model, test_ds: tf.data.Dataset, class_names: Sequence[str], output_dir: Path) -> None:
    y_true: List[int] = []
    y_pred: List[int] = []

    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_pred.extend(int(p) for p in preds)
        y_true.extend(int(v) for v in labels.numpy())

    report_dict = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    report_path = output_dir / "classification_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)

    cm_path = output_dir / "confusion_matrix.csv"
    np.savetxt(cm_path, cm, fmt="%d", delimiter=",")

    macro_f1 = report_dict.get("macro avg", {}).get("f1-score", 0.0)
    weighted_f1 = report_dict.get("weighted avg", {}).get("f1-score", 0.0)
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"Saved detailed report: {report_path}")
    print(f"Saved confusion matrix: {cm_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a plant disease prediction model.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=r"c:\Users\asus\Downloads\archive (1)\PlantVillage",
        help="Path to dataset root. Script auto-detects nested duplicate root if present.",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Directory for saved model and metadata.")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Legacy alias for --head-epochs.")
    parser.add_argument("--head-epochs", type=int, default=8, help="Epochs with frozen backbone.")
    parser.add_argument("--fine-tune-epochs", type=int, default=12, help="Fine-tuning epochs with partially unfrozen backbone.")
    parser.add_argument("--fine-tune-at", type=int, default=100, help="Unfreeze MobileNetV2 layers from this index onward.")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Learning rate for frozen-backbone stage.")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5, help="Learning rate for fine-tuning stage.")
    parser.add_argument("--dropout", type=float, default=0.35, help="Dropout before classifier head.")
    parser.add_argument("--use-class-weights", action="store_true", help="Use balanced class weights during training.")
    parser.add_argument("--early-patience", type=int, default=4, help="Early-stopping patience in epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--test-size", type=float, default=0.15, help="Fraction for test split.")
    parser.add_argument("--val-size", type=float, default=0.15, help="Fraction for validation split.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epochs is not None:
        args.head_epochs = args.epochs

    tf.keras.utils.set_random_seed(args.seed)

    base_dir = Path(args.data_dir)
    if not base_dir.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {base_dir}")

    dataset_root, dataset_message = detect_dataset_root(base_dir)
    print(dataset_message)

    image_paths, labels, class_names = collect_samples(dataset_root)
    print(f"Detected {len(class_names)} classes and {len(image_paths)} images.")

    # Split to train/val/test with stratification for balanced class distribution.
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths,
        labels,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels,
    )

    val_ratio_of_remaining = args.val_size / (1.0 - args.test_size)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=val_ratio_of_remaining,
        random_state=args.seed,
        stratify=train_labels,
    )

    print(
        "Split sizes -> "
        f"train: {len(train_paths)}, "
        f"val: {len(val_paths)}, "
        f"test: {len(test_paths)}"
    )

    train_ds = make_dataset(train_paths, train_labels, args.image_size, args.batch_size, training=True)
    val_ds = make_dataset(val_paths, val_labels, args.image_size, args.batch_size, training=False)
    test_ds = make_dataset(test_paths, test_labels, args.image_size, args.batch_size, training=False)

    class_weights = None
    if args.use_class_weights:
        classes_array = np.array(sorted(set(train_labels)))
        class_weights_values = compute_class_weight(class_weight="balanced", classes=classes_array, y=np.array(train_labels))
        class_weights = {int(c): float(w) for c, w in zip(classes_array, class_weights_values)}
        print(f"Computed class weights for {len(class_weights)} classes.")
    else:
        print("Class weights disabled (best overall accuracy mode).")

    model, base_model = build_model(
        num_classes=len(class_names),
        image_size=args.image_size,
        dropout_rate=args.dropout,
        base_trainable=False,
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.head_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_head_model_path = output_dir / "best_head_model.keras"
    head_callbacks = make_callbacks(best_head_model_path, args.early_patience)

    print(f"Stage 1/2: Training classifier head for {args.head_epochs} epochs.")
    head_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        callbacks=head_callbacks,
        class_weight=class_weights,
    )

    print("Stage 2/2: Fine-tuning top backbone layers.")
    base_model.trainable = True
    for layer in base_model.layers[: args.fine_tune_at]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.fine_tune_lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    best_model_path = output_dir / "best_model.keras"
    fine_tune_callbacks = make_callbacks(best_model_path, args.early_patience)
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        callbacks=fine_tune_callbacks,
        class_weight=class_weights,
    )

    full_history = merge_histories(head_history, fine_tune_history)

    final_model_path = output_dir / "plant_disease_model.keras"
    model.save(final_model_path)

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    labels_path = output_dir / "class_names.json"
    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    history_path = output_dir / "train_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(full_history, f, indent=2)

    evaluate_detailed(model, test_ds, class_names, output_dir)

    print("Saved files:")
    print(f"- {final_model_path}")
    print(f"- {best_head_model_path}")
    print(f"- {best_model_path}")
    print(f"- {labels_path}")
    print(f"- {history_path}")


if __name__ == "__main__":
    main()
