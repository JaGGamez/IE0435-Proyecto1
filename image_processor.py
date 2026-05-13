import os
import numpy as np
from PIL import Image

IMAGES_DIR = "Images"
TARGET_SIZE = (128, 128)
# Pixels brighter than this threshold are considered background (white → 1)
# Pixels at or below are considered object/rice (dark → 0)
THRESHOLD = 150


def process_image(image_path: str) -> np.ndarray:
    """Load an image, convert to grayscale 128x128, and binarize to 0/1 array."""
    with Image.open(image_path) as img:
        img = img.convert("L")              # grayscale
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        pixels = np.array(img, dtype=np.uint8)

    # 1 = white background, 0 = object (rice)
    binary = np.where(pixels > THRESHOLD, 1, 0).astype(np.uint8)
    return binary


def process_folder(folder_path: str) -> dict[str, np.ndarray]:
    """Recursively process all images in folder_path and return a label→arrays dict."""
    results = {}
    supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    for root, _, files in os.walk(folder_path):
        for filename in files:
            if os.path.splitext(filename)[1].lower() not in supported:
                continue

            file_path = os.path.join(root, filename)
            # Use path relative to IMAGES_DIR as the key
            rel_key = os.path.relpath(file_path, folder_path)

            try:
                binary_array = process_image(file_path)
                results[rel_key] = binary_array
            except Exception as exc:
                print(f"  [SKIP] {rel_key}: {exc}")

    return results


def main():
    if not os.path.isdir(IMAGES_DIR):
        raise FileNotFoundError(f"Folder '{IMAGES_DIR}' not found in the current directory.")

    print(f"Processing images in '{IMAGES_DIR}' -> grayscale 128x128 -> binary array\n")
    images = process_folder(IMAGES_DIR)

    for rel_path, arr in images.items():
        white_pixels = int(np.sum(arr == 1))
        object_pixels = int(np.sum(arr == 0))
        print(f"{rel_path}")
        print(f"  shape : {arr.shape}  |  white(1): {white_pixels}  object(0): {object_pixels}")
        print(f"  array preview (first row):\n  {arr[0]}\n")

    print(f"Total images processed: {len(images)}")

    # Save all arrays to a single .npz file for later use
    output_file = "binary_images.npz"
    np.savez_compressed(output_file, **{k.replace(os.sep, "_"): v for k, v in images.items()})
    print(f"\nAll binary arrays saved to '{output_file}'")


if __name__ == "__main__":
    main()
