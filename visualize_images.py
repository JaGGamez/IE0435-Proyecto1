import numpy as np
import os

NPZ_FILE = "binary_images.npz"


def print_matrix(key: str, arr: np.ndarray):
    print(f"\n{'='*60}")
    print(f"Image: {key}  |  shape: {arr.shape}")
    print(f"  white pixels (1): {int(np.sum(arr == 1))}  |  object pixels (0): {int(np.sum(arr == 0))}")
    print(f"{'='*60}")
    # Print each row as a compact string of 0s and 1s for readability
    for row in arr:
        print("".join(str(v) for v in row))


def save_matrices_to_txt(data: np.lib.npyio.NpzFile, output_file: str = "matrices.txt"):
    with open(output_file, "w") as f:
        for key in sorted(data.files):
            arr = data[key]
            f.write(f"{'='*60}\n")
            f.write(f"Image: {key}  |  shape: {arr.shape}\n")
            f.write(f"  white pixels (1): {int(np.sum(arr == 1))}  |  object pixels (0): {int(np.sum(arr == 0))}\n")
            f.write(f"{'='*60}\n")
            for row in arr:
                f.write("".join(str(v) for v in row) + "\n")
            f.write("\n")
    print(f"All matrices saved to '{output_file}'")


def main():
    if not os.path.isfile(NPZ_FILE):
        print(f"'{NPZ_FILE}' not found. Run image_processor.py first.")
        return

    data = np.load(NPZ_FILE)
    keys = sorted(data.files)
    print(f"Loaded {len(keys)} images.\n")
    print("Available images:")
    for i, k in enumerate(keys):
        print(f"  [{i}] {k}")

    print("\nOptions:")
    print("  Enter an image number to print its matrix")
    print("  Enter 'all' to print all matrices")
    print("  Enter 'save' to save all matrices to matrices.txt")
    print("  Enter 'q' to quit")

    while True:
        choice = input("\n> ").strip().lower()

        if choice == "q":
            break
        elif choice == "all":
            for key in keys:
                print_matrix(key, data[key])
        elif choice == "save":
            save_matrices_to_txt(data)
        elif choice.isdigit() and int(choice) < len(keys):
            key = keys[int(choice)]
            print_matrix(key, data[key])
        else:
            print("Invalid input. Try a number, 'all', 'save', or 'q'.")


if __name__ == "__main__":
    main()
