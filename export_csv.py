import numpy as np
import csv
import os

NPZ_FILE = "binary_images.npz"
OUTPUT_CSV = "dataset.csv"
POSITIVE_FOLDER = "Positivo"   # label = 1
TARGET_SIZE = (128, 128)
N_PIXELS = TARGET_SIZE[0] * TARGET_SIZE[1]  # 16384


def get_label(key: str) -> int:
    """Return 1 if image comes from the Positivo folder, 0 otherwise."""
    return 1 if key.startswith(POSITIVE_FOLDER) else 0


def main():
    if not os.path.isfile(NPZ_FILE):
        print(f"'{NPZ_FILE}' not found. Run image_processor.py first.")
        return

    data = np.load(NPZ_FILE)
    keys = sorted(data.files)

    # Column headers: pixel_0, pixel_1, ..., pixel_16383, label
    headers = [f"pixel_{i}" for i in range(N_PIXELS)] + ["label"]

    positives = sum(1 for k in keys if get_label(k) == 1)
    negatives = len(keys) - positives
    print(f"Images found: {len(keys)}  (Positivo=1: {positives}, Negativo=0: {negatives})")
    print(f"Row vector size: {N_PIXELS} pixels + 1 label = {N_PIXELS + 1} columns")
    print(f"Writing '{OUTPUT_CSV}'...")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for key in keys:
            arr = data[key]
            row_vector = arr.flatten().tolist()
            label = get_label(key)
            writer.writerow(row_vector + [label])

    size_kb = os.path.getsize(OUTPUT_CSV) / 1024
    print(f"Done. '{OUTPUT_CSV}' — {len(keys)} rows x {N_PIXELS + 1} columns ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
