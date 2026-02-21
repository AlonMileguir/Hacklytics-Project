"""
Download medical imaging datasets from MedMNIST and encode into VectorAI DB.

Three datasets are downloaded by default (no account required):

  pneumonia  — Chest X-rays:        Normal vs Pneumonia          (624 images)
  path       — Colon histology:     9 cancer tissue types       (7180 images, capped)
  derma      — Skin lesions:        7 dermatology categories    (2010 images, capped)

Images are saved to  ./data/<dataset>/<label>/  then encoded into the vector DB.

Usage:
  # Download all 3 datasets and encode into VectorAI DB (recommended)
  python app/download_dataset.py --encode

  # Download only (skip encoding)
  python app/download_dataset.py

  # One specific dataset
  python app/download_dataset.py --dataset pneumonia --encode

  # Cap images per class (default 300)
  python app/download_dataset.py --limit 100 --encode

  # Use training split for more images
  python app/download_dataset.py --split train --encode
"""

import sys
import argparse
from pathlib import Path

from PIL import Image
import numpy as np

# Add project root so we can import app.medical_image_encoder
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------
# Each entry: (medmnist_class_name, modality_tag, label_dict)
DATASETS = {
    "pneumonia": dict(
        class_name="PneumoniaMNIST",
        modality="xray",
        description="Chest X-rays — Normal vs Pneumonia",
        labels={0: "Normal", 1: "Pneumonia"},
    ),
    "path": dict(
        class_name="PathMNIST",
        modality="pathology",
        description="Colon cancer histology — 9 tissue types",
        labels={
            0: "Adipose", 1: "Background", 2: "Debris", 3: "Lymphocytes",
            4: "Mucus", 5: "Muscle", 6: "Normal_colon", 7: "Cancer_stroma",
            8: "Colorectal_cancer",
        },
    ),
    "derma": dict(
        class_name="DermaMNIST",
        modality="dermatology",
        description="Skin lesions — 7 dermatology categories",
        labels={
            0: "Melanocytic_nevi", 1: "Melanoma", 2: "Benign_keratosis",
            3: "Basal_cell_carcinoma", 4: "Actinic_keratoses",
            5: "Vascular_lesions", 6: "Dermatofibroma",
        },
    ),
}


def check_medmnist():
    try:
        import medmnist  # noqa: F401
        return True
    except ImportError:
        return False


def download_and_save(
    dataset_key: str,
    split: str = "test",
    limit: int = 300,
    size: int = 128,
) -> tuple[Path, int, dict]:
    """
    Download a MedMNIST dataset and save images as PNGs organized by label.

    Returns (dataset_dir, total_saved, label_counts)
    """
    import medmnist

    cfg = DATASETS[dataset_key]
    DataClass = getattr(medmnist, cfg["class_name"])

    print(f"\n{'='*60}")
    print(f"Downloading: {cfg['description']}")
    print(f"  Split : {split}  |  Image size: {size}x{size}  |  Cap: {limit}/class")
    print(f"{'='*60}")

    try:
        dataset = DataClass(split=split, download=True, size=size)
    except TypeError:
        # Older medmnist versions don't support size param
        dataset = DataClass(split=split, download=True)

    out_dir = DATA_DIR / dataset_key
    label_counts: dict[str, int] = {}
    total_saved = 0

    for idx in range(len(dataset)):
        img_data, label_data = dataset[idx]

        # Label may be a numpy array or int
        label_idx = int(label_data.flat[0]) if hasattr(label_data, "flat") else int(label_data)
        label_name = cfg["labels"].get(label_idx, f"class_{label_idx}")

        # Enforce per-class cap
        if label_counts.get(label_name, 0) >= limit:
            continue

        # Convert to PIL
        if isinstance(img_data, np.ndarray):
            arr = img_data
            if arr.ndim == 2:
                arr = arr[:, :, np.newaxis]
            if arr.shape[2] == 1:
                arr = np.concatenate([arr, arr, arr], axis=2)
            img = Image.fromarray(arr.astype(np.uint8))
        else:
            img = img_data.convert("RGB")

        # Save
        label_dir = out_dir / label_name
        label_dir.mkdir(parents=True, exist_ok=True)
        count = label_counts.get(label_name, 0)
        img.save(label_dir / f"img_{count:05d}.png")

        label_counts[label_name] = count + 1
        total_saved += 1

    print(f"\nSaved {total_saved} images to {out_dir}")
    for label, n in sorted(label_counts.items()):
        print(f"  {label:<30} {n:>4} images")

    return out_dir, total_saved, label_counts


def encode_dataset(
    dataset_key: str,
    dataset_dir: Path,
    label_counts: dict,
    server: str,
    start_id: int,
) -> int:
    """Encode all saved images into VectorAI DB with modality + label metadata."""
    from app.medical_image_encoder import MedicalImageEncoder, setup_collection, COLLECTION_NAME
    from cortex import CortexClient

    cfg = DATASETS[dataset_key]
    modality = cfg["modality"]

    print(f"\nEncoding '{dataset_key}' into VectorAI DB (server: {server})…")

    encoder = MedicalImageEncoder()

    current_id = start_id
    total_encoded = 0

    with CortexClient(server) as client:
        setup_collection(client)

        for label_name in sorted(label_counts.keys()):
            label_dir = dataset_dir / label_name
            image_files = sorted(label_dir.glob("*.png"))

            if not image_files:
                continue

            print(f"\n  [{label_name}] — {len(image_files)} images")

            ids, vectors, payloads = [], [], []

            for img_path in image_files:
                try:
                    vector = encoder.encode_image(str(img_path))
                    payload = {
                        "filename": img_path.name,
                        "path": str(img_path.resolve()),
                        "format": ".png",
                        "modality": modality,
                        "label": label_name,
                        "dataset": dataset_key,
                    }
                    ids.append(current_id)
                    vectors.append(vector.tolist())
                    payloads.append(payload)
                    current_id += 1
                except Exception as e:
                    print(f"    Warning: {img_path.name} failed — {e}")

            if ids:
                client.batch_upsert(COLLECTION_NAME, ids, vectors, payloads)
                total_encoded += len(ids)
                print(f"    Stored {len(ids)} vectors")

    return total_encoded


def print_manual_encode_commands(datasets_done: list[str], server: str):
    """Print CLI commands the user can run manually if they skipped --encode."""
    print("\n" + "="*60)
    print("To encode the downloaded images into VectorAI DB, run:")
    print("="*60)

    start_id = 0
    for key in datasets_done:
        cfg = DATASETS[key]
        dataset_dir = DATA_DIR / key
        for label_dir in sorted(dataset_dir.iterdir()):
            if label_dir.is_dir():
                print(
                    f"\npython app/medical_image_encoder.py encode "
                    f'"{label_dir}" '
                    f"--modality {cfg['modality']} "
                    f"--start-id {start_id} "
                    f"--server {server}"
                )
                # Rough estimate: 300 images max per class
                start_id += 300

    print(f"\n# Then launch the web app:")
    print(f"streamlit run app/web_app.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download MedMNIST datasets and optionally encode into VectorAI DB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        choices=list(DATASETS.keys()) + ["all"],
        default="all",
        help="Which dataset to download (default: all)",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val", "train"],
        default="test",
        help="MedMNIST split to use (default: test — smallest download)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=300,
        help="Max images per class label (default: 300)",
    )
    parser.add_argument(
        "--size",
        type=int,
        choices=[28, 64, 128, 224],
        default=128,
        help="Image size in pixels (default: 128)",
    )
    parser.add_argument(
        "--encode",
        action="store_true",
        help="Encode and store images in VectorAI DB after downloading",
    )
    parser.add_argument(
        "--server",
        default="localhost:50051",
        help="VectorAI DB server address (default: localhost:50051)",
    )

    args = parser.parse_args()

    # Check medmnist
    if not check_medmnist():
        print("medmnist is not installed. Installing now...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "medmnist"])

    keys = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]

    print(f"\nMedical Imaging Dataset Downloader")
    print(f"Datasets : {', '.join(keys)}")
    print(f"Split    : {args.split}")
    print(f"Limit    : {args.limit} images/class")
    print(f"Img size : {args.size}x{args.size} px")
    print(f"Output   : {DATA_DIR}/")

    if args.encode:
        # Quick DB connectivity check
        try:
            from cortex import CortexClient
            with CortexClient(args.server) as client:
                client.health_check()
            print(f"DB       : connected ({args.server})")
        except Exception as e:
            print(f"\nError: Cannot connect to VectorAI DB at {args.server}")
            print(f"       {e}")
            print(f"       Start it with: docker compose up -d")
            sys.exit(1)

    # Download and (optionally) encode each dataset
    current_id = 0
    datasets_done = []

    for key in keys:
        dataset_dir, total_saved, label_counts = download_and_save(
            key, split=args.split, limit=args.limit, size=args.size
        )
        datasets_done.append(key)

        if args.encode and total_saved > 0:
            encoded = encode_dataset(
                key, dataset_dir, label_counts, args.server, start_id=current_id
            )
            current_id += encoded
            print(f"\nEncoded {encoded} images from '{key}'")

    # Summary
    print(f"\n{'='*60}")
    print(f"Done!")
    print(f"{'='*60}")

    if args.encode:
        try:
            from app.medical_image_encoder import COLLECTION_NAME
            from cortex import CortexClient
            with CortexClient(args.server) as client:
                total = client.count(COLLECTION_NAME)
            print(f"Total images in VectorAI DB: {total}")
            print(f"\nLaunch the web app:")
            print(f"  streamlit run app/web_app.py")
        except Exception:
            pass
    else:
        print_manual_encode_commands(datasets_done, args.server)


if __name__ == "__main__":
    main()
