"""
MultiCaRe Dataset Setup — One-time script to download and process real medical cases.

Downloads real clinical cases from PubMed Central via the MultiCaRe Dataset (Zenodo),
uses Gemini to extract structured clinical fields from the raw case text, and saves
everything to data/multicare_cases.json for use in the web app.

Source: https://github.com/mauro-nievoff/MultiCaRe_Dataset
License: CC0 (public domain)

Usage:
    python app/setup_multicare.py --api-key YOUR_GEMINI_KEY
    python app/setup_multicare.py --api-key YOUR_GEMINI_KEY --specialty dermatology
    python app/setup_multicare.py --api-key YOUR_GEMINI_KEY --max-cases 50

⚠️  First run downloads ~2GB from Zenodo — takes 5–15 minutes.
    Subsequent runs reuse the cached download.
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

MULTICARE_DIR    = str(ROOT / "data" / "multicare_raw")
OUTPUT_JSON      = ROOT / "data" / "multicare_cases.json"
DATA_DIR         = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Specialty filter presets — maps friendly name → MultiCaRe image labels
# ---------------------------------------------------------------------------
SPECIALTY_FILTERS = {
    "dermatology": {
        "label": ["dermatology"],
        "description": "Skin lesions, rashes, and dermatological conditions",
    },
    "radiology": {
        "label": ["radiology", "chest"],
        "description": "Chest X-rays and general radiological imaging",
    },
    "pathology": {
        "label": ["pathology", "histology"],
        "description": "Histopathology and microscopy slides",
    },
    "mri": {
        "label": ["mri", "brain"],
        "description": "MRI scans (brain, spine, musculoskeletal)",
    },
    "ct": {
        "label": ["ct", "abdomen"],
        "description": "CT scans (abdominal, thoracic)",
    },
    "mixed": {
        "label": ["dermatology", "radiology", "pathology", "mri", "ct", "chest"],
        "description": "Mixed modalities (dermatology, radiology, pathology, MRI, CT)",
    },
}


# ---------------------------------------------------------------------------
# Step 1 — Download MultiCaRe dataset
# ---------------------------------------------------------------------------

def download_multicare(specialty: str, dataset_name: str) -> object:
    """Download and filter the MultiCaRe dataset. Returns mdc object."""
    try:
        from multiversity.multicare_dataset import MedicalDatasetCreator
    except ImportError:
        print("ERROR: multiversity not installed. Run: pip install multiversity ipython")
        sys.exit(1)

    cfg = SPECIALTY_FILTERS[specialty]
    print(f"\n{'='*60}")
    print(f"MultiCaRe Dataset Download")
    print(f"{'='*60}")
    print(f"Specialty : {specialty}  ({cfg['description']})")
    print(f"Directory : {MULTICARE_DIR}")
    print(f"⚠️  First run downloads ~2GB from Zenodo (~5–15 min).")
    print(f"{'='*60}\n")

    mdc = MedicalDatasetCreator(directory=MULTICARE_DIR)

    # Check if this dataset name already exists
    dataset_path = Path(MULTICARE_DIR) / dataset_name
    if dataset_path.exists():
        print(f"Dataset '{dataset_name}' already exists — skipping download.")
        # Reload from saved files
        import pandas as pd
        mdc.dataset_name = dataset_name
        mdc.case_df = pd.read_csv(dataset_path / "cases.csv")
        # Reconstruct file paths for images
        image_meta_path = dataset_path / "image_metadata.json"
        if image_meta_path.exists():
            mdc.filtered_image_metadata_df = pd.read_json(image_meta_path)
            mdc.filtered_image_metadata_df["file_path"] = mdc.filtered_image_metadata_df["file"].apply(
                lambda x: str(dataset_path / "images" / x[:4] / x[:5] / x)
            )
        else:
            mdc.filtered_image_metadata_df = pd.DataFrame()
        return mdc

    filters = [
        {
            "field": "label",
            "string_list": cfg["label"],
            "operator": "any",
        }
    ]

    mdc.create_dataset(
        dataset_name=dataset_name,
        filter_list=filters,
        dataset_type="multimodal",
    )

    return mdc


# ---------------------------------------------------------------------------
# Step 2 — Extract structured clinical info with Gemini
# ---------------------------------------------------------------------------

def extract_case_structure(api_key: str, case_text: str, captions: list[str]) -> dict:
    """
    Use Gemini to parse raw clinical case text into structured fields.
    Returns a dict with all fields needed for the simulation.
    """
    from google import genai

    client = genai.Client(api_key=api_key)

    captions_str = "\n".join(f"• {c}" for c in captions[:4]) if captions else "Not available"

    prompt = f"""You are extracting structured clinical information from a published medical case report
for use in a medical student education platform.

CLINICAL CASE TEXT:
{case_text[:4000]}

IMAGE CAPTIONS:
{captions_str}

Extract and return a JSON object with EXACTLY these fields (all required):
{{
  "chief_complaint": "One concise sentence: why did the patient present? (e.g. '3-day history of fever and cough')",
  "patient_history": "2-3 sentences covering relevant PMH, medications, allergies, social/family history",
  "presenting_symptoms": ["symptom 1", "symptom 2", "symptom 3"],
  "exam_findings": "Key physical examination findings. If not described, write 'Not described in this case.'",
  "lab_results": "Relevant laboratory values. If not described, write 'Not described in this case.'",
  "imaging_description": "What the imaging shows, based on captions and text. Be specific.",
  "diagnosis": "The final diagnosis from the case report (be specific — include type/subtype if available)",
  "treatment": "Treatment provided or recommended in the case",
  "specialty": "Medical specialty (e.g. Dermatology, Cardiology, Neurology, Gastroenterology)",
  "difficulty": "Beginner or Intermediate or Advanced",
  "key_learning_points": ["educational point 1", "educational point 2", "educational point 3"],
  "case_summary": "One sentence description for the case library card (e.g. '45yo male with chest pain and ST-elevation — STEMI')"
}}

Rules:
- Return ONLY the JSON object — no markdown, no explanation, no code block
- All fields must be present even if the information is limited
- Be medically accurate
- If age/sex is mentioned in the text, make sure your summary reflects it"""

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text = response.text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip().rstrip("```").strip()
        return json.loads(text)
    except (json.JSONDecodeError, Exception) as e:
        return {
            "chief_complaint": "See case text",
            "patient_history": case_text[:300],
            "presenting_symptoms": [],
            "exam_findings": "Not described in this case.",
            "lab_results": "Not described in this case.",
            "imaging_description": captions_str,
            "diagnosis": "Unknown — see case text",
            "treatment": "See case text",
            "specialty": "General Medicine",
            "difficulty": "Intermediate",
            "key_learning_points": [],
            "case_summary": "Medical case from PubMed Central",
        }


# ---------------------------------------------------------------------------
# Step 3 — Build case objects
# ---------------------------------------------------------------------------

def build_cases(api_key: str, mdc, max_cases: int) -> list[dict]:
    """Convert MultiCaRe rows into our case format using Gemini extraction."""
    import pandas as pd

    case_df = mdc.case_df
    image_df = mdc.filtered_image_metadata_df

    print(f"\nAvailable cases: {len(case_df)}  |  Processing up to {max_cases}")

    # Sample evenly if there are more cases than max_cases
    if len(case_df) > max_cases:
        case_df = case_df.sample(n=max_cases, random_state=42).reset_index(drop=True)

    processed = []

    for idx, row in case_df.iterrows():
        case_id  = str(row["case_id"])
        pmcid    = str(row["pmcid"])
        gender   = str(row.get("gender", "unknown")).lower()
        age      = row.get("age", None)
        case_text = str(row["case_text"])

        # Find images for this case
        if "patient_id" in image_df.columns:
            case_imgs = image_df[image_df["patient_id"] == case_id]
        else:
            case_imgs = pd.DataFrame()

        captions    = case_imgs["caption"].tolist() if "caption" in case_imgs.columns else []
        file_paths  = case_imgs["file_path"].tolist() if "file_path" in case_imgs.columns else []
        raw_labels  = (
            case_imgs["gt_labels_for_semisupervised_classification"].explode().dropna().unique().tolist()
            if "gt_labels_for_semisupervised_classification" in case_imgs.columns
            else []
        )

        # Top-level label (first meaningful one)
        imaging_type = raw_labels[0] if raw_labels else "Medical imaging"

        # Find a real image path that exists on disk
        image_path = None
        for fp in file_paths:
            if Path(fp).exists():
                image_path = fp
                break

        print(f"  [{idx + 1}/{len(case_df)}] {case_id} — extracting with Gemini...", end=" ", flush=True)

        extracted = extract_case_structure(api_key, case_text, captions)
        time.sleep(0.3)  # light rate-limit buffer

        case = {
            "id":        case_id,
            "title":     extracted["case_summary"][:80],
            "specialty": extracted["specialty"],
            "difficulty": extracted["difficulty"],
            "patient": {
                "name":       "Patient",   # anonymised — real name not in dataset
                "age":        age if age is not None else "Unknown",
                "sex":        gender,
                "occupation": "Not specified",
            },
            "chief_complaint":    extracted["chief_complaint"],
            "history":            extracted["patient_history"],
            "presenting_symptoms": extracted["presenting_symptoms"],
            "exam_findings":      extracted["exam_findings"],
            "labs":               extracted["lab_results"],
            "imaging_type":       imaging_type,
            "imaging_label":      imaging_type,
            "imaging_description": extracted["imaging_description"],
            "imaging_path":       image_path,          # actual file path (may be None)
            "diagnosis":          extracted["diagnosis"],
            "treatment":          extracted["treatment"],
            "key_learning_points": extracted["key_learning_points"],
            "description":        extracted["case_summary"],
            "clinical_text":      case_text,           # full raw text for Gemini simulation
            "source":             "PubMed Central (MultiCaRe)",
            "pmcid":              pmcid,
            "image_captions":     captions[:3],
        }

        processed.append(case)
        print("✓")

    return processed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and process MultiCaRe medical cases for MedCase app",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--api-key", required=True, help="Gemini API key")
    parser.add_argument(
        "--specialty",
        choices=list(SPECIALTY_FILTERS.keys()),
        default="mixed",
        help="Type of cases to download (default: mixed)",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=40,
        help="Maximum number of cases to process (default: 40)",
    )
    parser.add_argument(
        "--dataset-name",
        default="medcase_subset",
        help="Name for the MultiCaRe subset folder (default: medcase_subset)",
    )
    args = parser.parse_args()

    print("\n🏥 MedCase — MultiCaRe Dataset Setup")
    print("=" * 60)

    # Step 1: Download
    mdc = download_multicare(args.specialty, args.dataset_name)

    # Step 2 + 3: Extract + build
    print(f"\nUsing Gemini to extract structured fields from case texts...")
    cases = build_cases(args.api_key, mdc, max_cases=args.max_cases)

    # Save
    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"✅  Done! {len(cases)} cases saved to {OUTPUT_JSON}")
    print(f"{'='*60}")
    print(f"\nLaunch the web app:")
    print(f"  streamlit run app/web_app.py")


if __name__ == "__main__":
    main()
