"""
Case Manager — stores and retrieves medical cases in VectorAI DB.

Case sources (in priority order):
  1. data/multicare_cases.json  — Real cases from PubMed Central (MultiCaRe)
     Run  python app/setup_multicare.py --api-key KEY  to generate this.
  2. app/cases_data.py          — Hardcoded demo cases (always available as fallback)

Cases are embedded using Gemini text-embedding-004 (768d) for semantic search.

Collections:
  medical_cases  (768d, COSINE) — case text embeddings
"""

import sys
import json
import random
from pathlib import Path
from typing import Optional
from xml.parsers.expat import model

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from cortex import CortexClient, DistanceMetric
from app.cases_data import CASES as HARDCODED_CASES

CASES_COLLECTION    = "medical_cases"
CASES_DIMENSION     = 768        # Gemini text-embedding-004
MULTICARE_JSON      = ROOT / "data" / "multicare_cases.json"


# ---------------------------------------------------------------------------
# Case source — MultiCaRe preferred, hardcoded fallback
# ---------------------------------------------------------------------------

def load_multicare_cases() -> list[dict]:
    """Load processed MultiCaRe cases from JSON if available."""
    if MULTICARE_JSON.exists():
        with open(MULTICARE_JSON, encoding="utf-8") as f:
            return json.load(f)
    return []


def get_all_cases() -> list[dict]:
    """Return MultiCaRe cases if available, otherwise hardcoded demo cases."""
    multicare = load_multicare_cases()
    return multicare if multicare else HARDCODED_CASES


def is_multicare_available() -> bool:
    return MULTICARE_JSON.exists() and MULTICARE_JSON.stat().st_size > 100


def get_case_by_id(case_id: str) -> Optional[dict]:
    """Find a case by its id field across all sources."""
    for case in get_all_cases():
        if case["id"] == case_id:
            return case
    return None


def filter_cases(
    specialty: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> list[dict]:
    """In-memory filter by specialty and/or difficulty."""
    results = get_all_cases()
    if specialty and specialty != "All":
        results = [c for c in results if c.get("specialty", "") == specialty]
    if difficulty and difficulty != "All":
        results = [c for c in results if c.get("difficulty", "") == difficulty]
    return results


def get_random_case(
    specialty: Optional[str] = None,
    difficulty: Optional[str] = None,
) -> dict:
    pool = filter_cases(specialty, difficulty)
    return random.choice(pool) if pool else random.choice(get_all_cases())


def get_unique_specialties() -> list[str]:
    return sorted(set(c.get("specialty", "Unknown") for c in get_all_cases()))


def get_unique_difficulties() -> list[str]:
    order = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    diffs = set(c.get("difficulty", "Intermediate") for c in get_all_cases())
    return sorted(diffs, key=lambda x: order.get(x, 9))


# ---------------------------------------------------------------------------
# Image path resolution
# ---------------------------------------------------------------------------

_MULTICARE_IMG_ROOT = ROOT / "data" / "multicare_raw" / "medcase_subset" / "images"


def get_case_image_path(case: dict) -> Optional[Path]:
    """
    Resolve the image for a case.

    Priority:
      1. case["imaging_path"]  — direct file path
      2. multicare_raw images  — search for PMC-ID-prefixed files (.webp/.jpg/.png)
      3. MedMNIST download dir — data/<imaging_dataset>/<imaging_label>/
         (hardcoded demo cases)
    """
    # 1. Direct path
    direct = case.get("imaging_path")
    if direct:
        p = Path(direct)
        if p.exists():
            return p

    # 2. MultiCaRe raw images: filename starts with the PMC article ID
    pmc_id = case["id"].split("_")[0]  # e.g. "PMC11148442"
    if _MULTICARE_IMG_ROOT.exists():
        matches = sorted(_MULTICARE_IMG_ROOT.rglob(f"{pmc_id}_*"))
        if matches:
            return matches[0]

    # 3. MedMNIST folder (demo cases)
    dataset = case.get("imaging_dataset")
    label   = case.get("imaging_label")
    if dataset and label:
        label_dir = ROOT / "data" / dataset / label
        if label_dir.exists():
            images = sorted(label_dir.glob("*.png"))
            if images:
                try:
                    idx = int(case["id"].split("_")[1]) - 1
                except (IndexError, ValueError):
                    idx = 0
                return images[idx % len(images)]

    return None


# ---------------------------------------------------------------------------
# VectorAI DB — case indexing for semantic search
# ---------------------------------------------------------------------------

def setup_cases_collection(client: CortexClient) -> bool:
    """Create the medical_cases collection if it doesn't exist."""
    return client.get_or_create_collection(
        name=CASES_COLLECTION,
        dimension=CASES_DIMENSION,
        distance_metric=DistanceMetric.COSINE,
    )


def is_cases_indexed(client: CortexClient) -> bool:
    if not client.has_collection(CASES_COLLECTION):
        return False
    return client.count(CASES_COLLECTION) >= len(get_all_cases())


def _embed_text(api_key: str, text: str) -> list[float]:
    from google import genai
    try:
        client = genai.Client(api_key=api_key)
        result = client.models.embed_content(model="gemini-embedding-001", contents=text)
        return list(result.embeddings[0].values)
    except Exception as e:
        print(f"Embedding with model gemini-embedding-001 failed: {e}")
        raise RuntimeError("No Gemini embedding model available — check your API key")


def _case_payload(case: dict) -> dict:
    """Serialisable payload for VectorAI DB (no large text blobs)."""
    return {
        "case_id":     case["id"],
        "title":       case["title"],
        "specialty":   case.get("specialty", ""),
        "difficulty":  case.get("difficulty", ""),
        "chief_complaint": case.get("chief_complaint", ""),
        "patient_age": case["patient"].get("age", ""),
        "patient_sex": case["patient"].get("sex", ""),
        "imaging_type": case.get("imaging_type", ""),
        "diagnosis":   case.get("diagnosis", ""),
        "description": case.get("description", ""),
        "source":      case.get("source", "demo"),
        "clinical_text": case.get("clinical_text", ""),
    }


def index_all_cases(api_key: str, db_server: str = "localhost:50051"):
    """
    Embed all cases with Gemini and store in VectorAI DB.
    Safe to call multiple times — skips if count already matches.
    """
    cases = get_all_cases()

    with CortexClient(db_server) as client:
        setup_cases_collection(client)
        
        if is_cases_indexed(client):
            print(f"Cases already indexed ({client.count(CASES_COLLECTION)} in DB).")
            return

        print(f"Indexing {len(cases)} cases into VectorAI DB…")

        for i, case in enumerate(cases):
            embed_text = case.get("clinical_text")
            vec = _embed_text(api_key, embed_text)
            payload = _case_payload(case)
            client.upsert(CASES_COLLECTION, id=i, vector=vec, payload=payload)

        client.flush(CASES_COLLECTION)
        print(f"Indexed {len(cases)} cases.")


def index_case_images(db_server: str = "localhost:50051") -> int:
    """
    Encode all case images with BiomedCLIP and store in the medical_images VectorAI collection.
    Stores case_id in metadata so image search results can be mapped back to cases.
    Safe to re-run — re-indexes all available images.

    Returns number of images successfully indexed.
    Requires: pip install open_clip_torch torch pillow
    """
    try:
        from app.medical_image_encoder import MedicalImageEncoder, setup_collection
        from cortex import CortexClient
    except ImportError as e:
        print(f"BiomedCLIP not available: {e}")
        return 0

    cases   = get_all_cases()
    encoder = MedicalImageEncoder(db_address=db_server)

    total = 0
    with CortexClient(db_server) as client:
        setup_collection(client)
        for i, case in enumerate(cases):
            img_path = get_case_image_path(case)
            if not img_path or not img_path.exists():
                continue
            try:
                encoder.store_image(
                    client,
                    image_id=i,
                    image_path=str(img_path),
                    metadata={"case_id": case["id"]},
                )
                total += 1
            except Exception as e:
                print(f"  Failed {case['id']}: {e}")

    print(f"Indexed {total} case images with BiomedCLIP.")
    return total


def search_cases(
    api_key: str,
    query: str,
    top_k: int = 9,
    db_server: str = "localhost:50051",
) -> list[dict]:
    """
    Semantic search using Gemini embeddings + VectorAI DB.
    Falls back to simple keyword filter if DB unavailable.
    """
    try:
        query_vec = _embed_text(api_key, query)
        with CortexClient(db_server) as client:
            if not client.has_collection(CASES_COLLECTION):
                raise RuntimeError("Collection not found")
            results = client.search(
                CASES_COLLECTION,
                query=query_vec,
                top_k=top_k,
                with_payload=True,
            )

        matched = []
        for r in results:
            case = get_case_by_id(r.payload["case_id"])
            if case:
                c = dict(case)
                c["_score"] = round(r.score * 100, 1)
                matched.append(c)
        matched.sort(key=lambda c: c["_score"], reverse=True)
        return matched
    except Exception as e:
        print(f"Error in search_cases: {e}")
        # Keyword fallback — score by number of fields that match the query
        q = query.lower()
        scored = []
        for c in get_all_cases():
            hits = sum([
                q in c.get("title", "").lower(),
                q in c.get("specialty", "").lower(),
                q in c.get("chief_complaint", "").lower(),
                q in c.get("description", "").lower(),
            ])
            if hits:
                case = dict(c)
                case["_score"] = round(hits / 4 * 100, 1)
                scored.append(case)
        scored.sort(key=lambda c: c["_score"], reverse=True)
        if scored:
            return scored[:top_k]
        # No keyword match — return first top_k cases with a low score
        return [{**c, "_score": 10.0} for c in get_all_cases()[:top_k]]
