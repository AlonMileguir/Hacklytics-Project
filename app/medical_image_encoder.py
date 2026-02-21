"""
Medical Image Encoder for Actian VectorAI DB

Encodes medical images (X-rays, CT, MRI, pathology slides, etc.) into
512-dimensional vectors using BiomedCLIP and stores them for similarity search.

Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
  - Fine-tuned on 15M biomedical image-text pairs from PubMed
  - Handles mixed medical modalities
  - 512-dimensional COSINE embeddings

Usage:
    # Encode a directory of images
    python app/medical_image_encoder.py encode ./my_images/

    # Search for similar images
    python app/medical_image_encoder.py search ./query_image.jpg --top-k 5

Install dependencies first:
    pip install -r app/requirements_medical.txt
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional, Callable

# --- Dependency checks ---
try:
    import open_clip
    import torch
    from PIL import Image
except ImportError:
    print("Missing dependencies. Run:")
    print("  pip install -r app/requirements_medical.txt")
    sys.exit(1)

# Optional DICOM support (install with: pip install pydicom)
try:
    import pydicom
    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False

from cortex import CortexClient, DistanceMetric
from cortex.filters import Filter

# --- Constants ---
COLLECTION_NAME = "medical_images"
DIMENSION = 512  # BiomedCLIP image encoder output
MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
if DICOM_AVAILABLE:
    SUPPORTED_FORMATS.add(".dcm")


class MedicalImageEncoder:
    """
    Encodes medical images into 512-d vectors using BiomedCLIP.

    BiomedCLIP is trained on biomedical image-text pairs, making it far more
    effective than general-purpose CLIP for medical similarity search.

    Example:
        encoder = MedicalImageEncoder()
        with CortexClient("localhost:50051") as client:
            setup_collection(client)
            encoder.encode_directory(client, "./xrays/")
            results = encoder.search_similar(client, "./query.jpg", top_k=5)
    """

    def __init__(self, db_address: str = "localhost:50051"):
        self.db_address = db_address
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.preprocess = None
        self._load_model()

    def _load_model(self):
        print(f"Loading BiomedCLIP on {self.device}...")
        print("(First run downloads ~350MB from HuggingFace — cached after that)")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL_ID)
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"BiomedCLIP loaded. Embedding dimension: {DIMENSION}")

    def _load_image(self, image_path: str) -> Image.Image:
        """Load an image, with special handling for DICOM files."""
        path = Path(image_path)
        ext = path.suffix.lower()

        if ext == ".dcm":
            if not DICOM_AVAILABLE:
                raise ImportError(
                    "pydicom is required for DICOM files. Install with: pip install pydicom"
                )
            ds = pydicom.dcmread(str(path))
            arr = ds.pixel_array.astype(np.float32)
            # Normalize pixel values to 0-255
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
            arr = arr.astype(np.uint8)
            # Convert grayscale slices to RGB (BiomedCLIP expects 3 channels)
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            elif arr.ndim == 3 and arr.shape[0] in (1, 3):
                # Channel-first format → channel-last
                arr = arr.transpose(1, 2, 0)
                if arr.shape[2] == 1:
                    arr = np.concatenate([arr, arr, arr], axis=-1)
            return Image.fromarray(arr)
        else:
            return Image.open(image_path).convert("RGB")

    def encode_image(self, image_path: str) -> np.ndarray:
        """
        Generate a normalized 512-d embedding for a single image.

        Args:
            image_path: Path to the image file (.jpg, .png, .dcm, etc.)

        Returns:
            numpy array of shape (512,), L2-normalized for COSINE similarity
        """
        img = self._load_image(image_path)
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(tensor)
            # L2-normalize so COSINE distance = dot product (required for COSINE metric)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()[0]

    def store_image(
        self,
        client: CortexClient,
        image_id: int,
        image_path: str,
        metadata: Optional[dict] = None,
    ):
        """
        Encode a single image and store it in the database.

        Args:
            client: Active CortexClient instance
            image_id: Unique integer ID for this vector
            image_path: Path to the image file
            metadata: Optional dict of fields to store alongside the vector
                      e.g. {"modality": "xray", "label": "pneumonia", "patient_id": "P001"}
        """
        vector = self.encode_image(image_path)
        payload = {
            "filename": Path(image_path).name,
            "path": str(Path(image_path).resolve()),
            "format": Path(image_path).suffix.lower(),
        }
        if metadata:
            payload.update(metadata)

        client.upsert(COLLECTION_NAME, id=image_id, vector=vector.tolist(), payload=payload)
        print(f"  Stored: {Path(image_path).name} (id={image_id})")

    def encode_directory(
        self,
        client: CortexClient,
        directory: str,
        start_id: int = 0,
        metadata_fn: Optional[Callable[[str], dict]] = None,
        recursive: bool = True,
    ) -> int:
        """
        Encode all supported images in a directory and batch-store them.

        Args:
            client: Active CortexClient instance
            directory: Path to the directory containing medical images
            start_id: Starting integer ID for the first image
            metadata_fn: Optional function(image_path: str) -> dict
                         Use this to attach labels, modalities, patient IDs, etc.
                         Example: lambda p: {"modality": "xray", "label": "normal"}
            recursive: If True, also scan subdirectories

        Returns:
            Number of images successfully encoded and stored

        Example with metadata:
            def get_metadata(path):
                # Derive modality from parent folder name
                return {"modality": Path(path).parent.name}

            encoder.encode_directory(client, "./images/", metadata_fn=get_metadata)
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        glob_fn = directory.rglob if recursive else directory.glob
        image_paths = sorted(
            p for p in glob_fn("*") if p.suffix.lower() in SUPPORTED_FORMATS
        )

        if not image_paths:
            print(f"No supported images found in {directory}")
            print(f"Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}")
            return 0

        print(f"\nFound {len(image_paths)} images in {directory}")

        ids, vectors, payloads = [], [], []
        failed = 0

        for i, path in enumerate(image_paths):
            print(f"  [{i + 1}/{len(image_paths)}] Encoding {path.name}...", end=" ")
            try:
                vector = self.encode_image(str(path))
                payload = {
                    "filename": path.name,
                    "path": str(path.resolve()),
                    "format": path.suffix.lower(),
                }
                if metadata_fn:
                    payload.update(metadata_fn(str(path)))

                ids.append(start_id + i)
                vectors.append(vector.tolist())
                payloads.append(payload)
                print("OK")
            except Exception as e:
                print(f"FAILED ({e})")
                failed += 1

        if ids:
            print(f"\nStoring {len(ids)} embeddings in VectorAI DB...")
            client.batch_upsert(COLLECTION_NAME, ids, vectors, payloads)
            print(f"Done. ({failed} failed)")

        return len(ids)

    def search_similar(
        self,
        client: CortexClient,
        query_image_path: str,
        top_k: int = 5,
        filter: Optional[Filter] = None,
    ) -> list:
        """
        Find the most similar images to a query image.

        Args:
            client: Active CortexClient instance
            query_image_path: Path to the query image
            top_k: Number of results to return
            filter: Optional Filter DSL expression to narrow results
                    e.g. Filter().must(Field("modality").eq("xray"))

        Returns:
            List of SearchResult objects with .id, .score, .payload

        Example:
            results = encoder.search_similar(client, "./scan.jpg", top_k=5)
            for r in results:
                print(r.score, r.payload["filename"])
        """
        print(f"Encoding query: {Path(query_image_path).name}")
        query_vector = self.encode_image(query_image_path)

        results = client.search(
            COLLECTION_NAME,
            query=query_vector.tolist(),
            top_k=top_k,
            filter=filter,
            with_payload=True,
        )
        return results


def setup_collection(client: CortexClient):
    """
    Create the medical_images collection if it doesn't already exist.
    Uses COSINE distance (best for normalized BiomedCLIP embeddings).
    Higher hnsw_ef_search improves accuracy at a small speed cost.
    """
    created = client.get_or_create_collection(
        name=COLLECTION_NAME,
        dimension=DIMENSION,
        distance_metric=DistanceMetric.COSINE,
        hnsw_m=16,
        hnsw_ef_construct=200,
        hnsw_ef_search=100,
    )
    if created:
        print(f"Created collection '{COLLECTION_NAME}' (dim={DIMENSION}, COSINE)")
    else:
        count = client.count(COLLECTION_NAME)
        print(f"Using existing collection '{COLLECTION_NAME}' ({count} images stored)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_encode(args):
    encoder = MedicalImageEncoder(args.server)

    # If --modality is provided, tag every image with it
    metadata_fn = None
    if args.modality:
        metadata_fn = lambda _: {"modality": args.modality}

    with CortexClient(args.server) as client:
        setup_collection(client)
        n = encoder.encode_directory(
            client,
            args.directory,
            start_id=args.start_id,
            metadata_fn=metadata_fn,
        )
        total = client.count(COLLECTION_NAME)
        print(f"\nEncoded {n} images. Total in DB: {total}")


def _cli_search(args):
    encoder = MedicalImageEncoder(args.server)

    filter_obj = None
    if args.modality:
        from cortex.filters import Filter, Field
        filter_obj = Filter().must(Field("modality").eq(args.modality))

    with CortexClient(args.server) as client:
        results = encoder.search_similar(
            client, args.image, top_k=args.top_k, filter=filter_obj
        )
        print(f"\nTop {len(results)} similar images:")
        print("-" * 60)
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.payload.get('filename', 'unknown')}  (score: {r.score:.4f})")
            if r.payload.get("modality"):
                print(f"     Modality : {r.payload['modality']}")
            if r.payload.get("label"):
                print(f"     Label    : {r.payload['label']}")
            print(f"     Path     : {r.payload.get('path', 'n/a')}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Medical Image Encoder — encode images into VectorAI DB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Encode all images in a folder
  python app/medical_image_encoder.py encode ./data/xrays/ --modality xray

  # Encode without modality tag
  python app/medical_image_encoder.py encode ./data/images/

  # Search for similar images
  python app/medical_image_encoder.py search ./query.jpg --top-k 5

  # Search within a specific modality
  python app/medical_image_encoder.py search ./query.jpg --modality xray
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # encode
    enc_p = subparsers.add_parser("encode", help="Encode images from a directory")
    enc_p.add_argument("directory", help="Directory containing medical images")
    enc_p.add_argument("--server", default="localhost:50051", help="VectorAI DB address")
    enc_p.add_argument("--start-id", type=int, default=0, help="Starting vector ID")
    enc_p.add_argument("--modality", default=None, help="Tag all images with this modality (e.g. xray, mri, ct)")

    # search
    srch_p = subparsers.add_parser("search", help="Find similar images")
    srch_p.add_argument("image", help="Query image path")
    srch_p.add_argument("--top-k", type=int, default=5, help="Number of results")
    srch_p.add_argument("--server", default="localhost:50051", help="VectorAI DB address")
    srch_p.add_argument("--modality", default=None, help="Filter results by modality")

    args = parser.parse_args()

    if args.command == "encode":
        _cli_encode(args)
    elif args.command == "search":
        _cli_search(args)
