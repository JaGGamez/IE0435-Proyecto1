import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
from PIL import Image

IMAGES_DIR = "Images"
TARGET_SIZE = (128, 128)
SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def collect_images(folder: str) -> list[str]:
    paths = []
    for root, _, files in os.walk(folder):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in SUPPORTED:
                paths.append(os.path.join(root, f))
    return sorted(paths)


def load_grayscale(path: str) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("L").resize(TARGET_SIZE, Image.LANCZOS)
        return np.array(img, dtype=np.uint8)


def binarize(gray: np.ndarray, threshold: int) -> np.ndarray:
    return np.where(gray > threshold, 1, 0).astype(np.uint8)


def main():
    paths = collect_images(IMAGES_DIR)
    if not paths:
        print("No images found in", IMAGES_DIR)
        return

    print(f"Found {len(paths)} images.")
    for i, p in enumerate(paths):
        print(f"  [{i}] {os.path.relpath(p, IMAGES_DIR)}")

    choice = input("\nEnter image number to start from (or press Enter for 0): ").strip()
    start = int(choice) if choice.isdigit() and int(choice) < len(paths) else 0

    # Mutable state shared across callbacks
    state = {"index": start}

    # --- Build figure ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.subplots_adjust(bottom=0.22)

    ax_orig, ax_gray, ax_bin = axes

    ax_orig.axis("off")
    ax_gray.axis("off")
    ax_bin.axis("off")

    img_orig = ax_orig.imshow(np.zeros(TARGET_SIZE), cmap="gray", vmin=0, vmax=255)
    ax_orig.set_title("Grayscale (0-255)")

    img_gray = ax_gray.imshow(np.zeros(TARGET_SIZE), cmap="gray", vmin=0, vmax=255)
    ax_gray.set_title("Intensity map")
    plt.colorbar(img_gray, ax=ax_gray, fraction=0.046, pad=0.04)

    img_bin = ax_bin.imshow(np.zeros(TARGET_SIZE), cmap="gray", vmin=0, vmax=1)
    title_bin = ax_bin.set_title("")

    suptitle = fig.suptitle("", fontsize=11)

    # --- Slider ---
    ax_slider = fig.add_axes([0.2, 0.1, 0.6, 0.04])
    slider = widgets.Slider(
        ax_slider, "Threshold", valmin=0, valmax=255,
        valinit=128, valstep=1, color="steelblue"
    )

    # --- Navigation buttons ---
    ax_prev = fig.add_axes([0.3, 0.03, 0.12, 0.05])
    ax_next = fig.add_axes([0.58, 0.03, 0.12, 0.05])
    btn_prev = widgets.Button(ax_prev, "< Previous")
    btn_next = widgets.Button(ax_next, "Next >")

    # Counter label between buttons
    ax_counter = fig.add_axes([0.43, 0.03, 0.14, 0.05])
    ax_counter.axis("off")
    counter_text = ax_counter.text(
        0.5, 0.5, "", ha="center", va="center", fontsize=10, transform=ax_counter.transAxes
    )

    fig.text(
        0.5, 0.005,
        "Drag the slider to adjust the threshold, then set THRESHOLD in image_processor.py.",
        ha="center", fontsize=9, color="gray"
    )

    def refresh():
        idx = state["index"]
        gray = load_grayscale(paths[idx])
        t = int(slider.val)
        binary = binarize(gray, t)

        img_orig.set_data(gray)
        img_gray.set_data(gray)
        img_bin.set_data(binary)

        label = os.path.relpath(paths[idx], IMAGES_DIR)
        suptitle.set_text(f"[{idx}] {label}")
        title_bin.set_text(
            f"Binary (threshold={t})\n"
            f"white(1): {int(np.sum(binary==1))}  object(0): {int(np.sum(binary==0))}"
        )
        counter_text.set_text(f"{idx + 1} / {len(paths)}")
        fig.canvas.draw_idle()

    def on_slider(_val):
        refresh()

    def on_next(_event):
        state["index"] = (state["index"] + 1) % len(paths)
        refresh()

    def on_prev(_event):
        state["index"] = (state["index"] - 1) % len(paths)
        refresh()

    slider.on_changed(on_slider)
    btn_next.on_clicked(on_next)
    btn_prev.on_clicked(on_prev)

    refresh()
    plt.show()


if __name__ == "__main__":
    main()
