import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict plant disease from one image.")
    parser.add_argument("--image", type=str, required=True, help="Path to image file.")
    parser.add_argument(
        "--model",
        type=str,
        default="artifacts/plant_disease_model.keras",
        help="Path to trained model file.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="artifacts/class_names.json",
        help="Path to class names JSON.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Input image size.")
    return parser.parse_args()


def load_image_for_model(image_path: Path, image_size: int) -> np.ndarray:
    image_bytes = tf.io.read_file(str(image_path))
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [image_size, image_size])
    image = preprocess_input(tf.cast(image, tf.float32))
    image = tf.expand_dims(image, axis=0)
    return image.numpy()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    labels_path = Path(args.labels)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    with labels_path.open("r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(model_path)
    image_batch = load_image_for_model(image_path, args.image_size)

    probs = model.predict(image_batch, verbose=0)[0]
    best_idx = int(np.argmax(probs))

    print(f"Predicted disease class: {class_names[best_idx]}")
    print(f"Confidence: {probs[best_idx] * 100:.2f}%")

    # Print top-3 classes for better interpretability.
    top_indices = np.argsort(probs)[-3:][::-1]
    print("Top-3 predictions:")
    for idx in top_indices:
        print(f"- {class_names[int(idx)]}: {probs[int(idx)] * 100:.2f}%")


if __name__ == "__main__":
    main()
