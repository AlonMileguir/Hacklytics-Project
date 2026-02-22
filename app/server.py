"""
MedCase — FastAPI backend for the clinical education platform.

Run with:
    uvicorn app.server:app --reload --port 8000
"""

import sys
import os
import random
import threading
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from cortex import Filter
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.case_manager import (
    CASES_COLLECTION,
    get_all_cases,
    get_case_by_id,
    get_case_image_path,
    filter_cases,
    search_cases,
    is_cases_indexed,
    index_all_cases,
    index_case_images,
)
from app.clinical_sim import (
    get_opening_message,
    send_message,
    detect_intent,
    is_case_complete,
)

API_KEY      = os.environ.get("GEMINI_API_KEY", "")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
STATIC_DIR   = ROOT / "app" / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI()


def _background_index():
    """Index cases into VectorAI DB on startup (skipped if already indexed)."""
    if not API_KEY:
        return
    try:
        from cortex import CortexClient
        with CortexClient("localhost:50051") as c:
            if is_cases_indexed(c):
                print("[index] Cases already indexed — skipping.")
                return
    except Exception as e:
        print("[index] VectorAI DB not reachable — skipping case indexing.")
        return
    print("[index] Indexing cases into VectorAI DB…")
    try:
        index_all_cases(API_KEY)
        print("[index] Case indexing complete.")
    except Exception as e:
        print(f"[index] Case indexing failed: {e}")


threading.Thread(target=_background_index, daemon=True).start()


# ---------------------------------------------------------------------------
# In-memory sessions
# ---------------------------------------------------------------------------

_sessions: dict[str, dict] = {}

_MALE_NAMES   = ["James", "Robert", "Michael", "William", "David", "Richard", "Joseph",
                 "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Mark", "Donald",
                 "Steven", "Paul", "Andrew", "Kenneth", "Joshua", "George"]
_FEMALE_NAMES = ["Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Susan", "Jessica",
                 "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra",
                 "Ashley", "Emily", "Dorothy", "Kimberly", "Carol", "Michelle"]
_LAST_NAMES   = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
                 "Davis", "Wilson", "Moore", "Taylor", "Anderson", "Thomas", "Jackson",
                 "White", "Harris", "Martin", "Thompson", "Young", "Robinson"]


def _assign_random_name(case: dict) -> dict:
    """Return a shallow copy of case with a randomly generated patient name."""
    sex = (case.get("patient") or {}).get("sex", "").lower()
    first = random.choice(_FEMALE_NAMES if "f" in sex else _MALE_NAMES)
    last  = random.choice(_LAST_NAMES)
    case  = {**case, "patient": {**case.get("patient", {}), "name": f"{first} {last}"}}
    return case


_biomed_encoder = None


def _get_biomed_encoder():
    """Lazy-load BiomedCLIP encoder (downloads ~350 MB on first run, then cached)."""
    global _biomed_encoder
    if _biomed_encoder is None:
        from app.medical_image_encoder import MedicalImageEncoder
        _biomed_encoder = MedicalImageEncoder()
    return _biomed_encoder


def _new_session() -> dict:
    return {
        "case":     None,
        "history":  [],
        "revealed": set(),
        "complete": False,
    }


def _get(sid: str) -> dict:
    if sid not in _sessions:
        _sessions[sid] = _new_session()
    return _sessions[sid]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _case_meta(case: dict) -> dict:
    return {
        "id":              case["id"],
        "title":           case.get("title", ""),
        "specialty":       case.get("specialty", ""),
        "difficulty":      case.get("difficulty", ""),
        "patient":         case.get("patient", {}),
        "chief_complaint": case.get("chief_complaint", ""),
        "history":         case.get("history", ""),
        "imaging_type":    case.get("imaging_type", ""),
    }


_GENERIC_IMAGING = {"medical imaging", "not available", "imaging", "", "none"}


def _image_is_usable(case: dict) -> bool:
    """
    Return True only when the case has a real image file AND an imaging type
    specific enough that the file is likely to match what the nurse describes.
    MultiCaRe cases with generic 'Medical imaging' type have arbitrary PMC
    article figures that may be histology slides, diagrams, etc. — not the
    modality the nurse describes — so we suppress them.
    """
    imaging_type = case.get("imaging_type", "").strip().lower()
    if imaging_type in _GENERIC_IMAGING:
        return False
    img_path = get_case_image_path(case)
    return bool(img_path and img_path.exists())


def _revealed_data(case: dict, revealed: set) -> dict:
    data = {}
    if "exam" in revealed:
        data["exam"] = case.get("exam_findings", "")
    if "labs" in revealed:
        data["labs"] = case.get("labs", "")
    if "imaging" in revealed:
        data["imaging"] = {
            "type":        case.get("imaging_type", "Imaging"),
            "description": case.get("imaging_description", ""),
            "has_image":   _image_is_usable(case),
        }
    return data


def _find_case(query: str) -> dict:
    """Find best matching case using Gemini selection, VectorAI, or fallback."""
    # 1. Try VectorAI semantic search
    try:
        from cortex import CortexClient
        with CortexClient("localhost:50051") as c:
            if is_cases_indexed(c):
                cases = search_cases(API_KEY, query, top_k=1, db_server="localhost:50051")
                if cases:
                    return cases[0]
    except Exception:
        print("Vector search failed, falling back to Gemini ranking")
        pass

    # 2. Use Gemini to intelligently pick the best case
    if API_KEY:
        try:
            return _gemini_pick_case(query)
        except Exception:
            pass

    return random.choice(get_all_cases())


def _gemini_pick_case(query: str) -> dict:
    """Ask Gemini to select the best matching case from all available cases."""
    import re
    from google import genai

    all_cases = get_all_cases()

    # Build a compact one-line summary per case
    summaries = []
    for i, c in enumerate(all_cases):
        p = c.get("patient", {})
        summaries.append(
            f"{i}: {c.get('title', '')} | sex={p.get('sex', '?')} age={p.get('age', '?')} "
            f"| {c.get('specialty', '')} | {c.get('chief_complaint', '')}"
        )

    prompt = (
        f'A medical student wants to practice this type of case: "{query}"\n\n'
        f"Available cases (index | title | sex | age | specialty | chief complaint):\n"
        + "\n".join(summaries)
        + "\n\nReply with ONLY the index number of the single best matching case. "
        "Carefully match sex, age range, and medical condition to what was requested."
    )

    client   = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
    match    = re.search(r"\d+", response.text.strip())
    if match:
        idx = int(match.group())
        if 0 <= idx < len(all_cases):
            return all_cases[idx]

    return random.choice(all_cases)


def _vector_search_cases(query: str, top_k: int = 10) -> list[dict]:
    """Semantic search via VectorAI DB (Gemini embeddings) with keyword fallback."""
    return search_cases(API_KEY, query, top_k=top_k, db_server="localhost:50051")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/auth-required")
def auth_required():
    return {"required": bool(APP_PASSWORD)}


class LoginReq(BaseModel):
    password: str


@app.post("/api/login")
def login(req: LoginReq):
    if not APP_PASSWORD:
        return {"ok": True}
    if req.password == APP_PASSWORD:
        return {"ok": True}
    raise HTTPException(status_code=401, detail="Invalid password")


@app.get("/api/image/{case_id}")
def case_image(case_id: str):
    case = get_case_by_id(case_id)
    if not case:
        raise HTTPException(status_code=404)
    img_path = get_case_image_path(case)
    if not img_path or not img_path.exists():
        raise HTTPException(status_code=404)
    return FileResponse(str(img_path))


class ChatReq(BaseModel):
    session_id: str
    message: str


@app.post("/api/chat")
def chat(req: ChatReq):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    sess = _get(req.session_id)

    # ── No active case → treat message as a search query ──────────────────
    if sess["case"] is None:
        case = _assign_random_name(_find_case(req.message))
        sess["case"]     = case
        sess["history"]  = []
        sess["revealed"] = set()
        sess["complete"] = False

        opening = get_opening_message(API_KEY, case)
        sess["history"].append({"role": "gemini", "content": opening})

        return {
            "response":       opening,
            "case":           _case_meta(case),
            "revealed":       {},
            "newly_revealed": {},
            "complete":       False,
            "image_url":      None,
        }

    # ── Active simulation ──────────────────────────────────────────────────
    case          = sess["case"]
    prev_revealed = set(sess["revealed"])
    intents       = detect_intent(req.message)
    new_revealed  = sess["revealed"] | (intents & {"exam", "labs", "imaging"})

    reply = send_message(API_KEY, case, sess["history"], req.message, new_revealed)

    sess["revealed"] = new_revealed
    sess["history"].append({"role": "student", "content": req.message})
    sess["history"].append({"role": "gemini",  "content": reply})

    complete        = is_case_complete(sess["history"])
    sess["complete"] = complete

    # Clear the case when complete, to allow starting a new case on next message
    if complete:
        sess["case"] = None

    delta           = new_revealed - prev_revealed
    newly_revealed  = _revealed_data(case, delta)

    image_url = None
    if "imaging" in delta and _image_is_usable(case):
        image_url = f"/api/image/{case['id']}"

    return {
        "response":       reply,
        "case":           _case_meta(case),
        "revealed":       _revealed_data(case, new_revealed),
        "newly_revealed": newly_revealed,
        "complete":       complete,
        "image_url":      image_url,
    }


class SessionReq(BaseModel):
    session_id: str


@app.post("/api/panel-update")
def panel_update(req: SessionReq):
    """
    Separate agent: reads the chat history and extracts clinical info for the
    Case Overview panel. Does NOT look at the dataset — only the conversation.
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    sess = _get(req.session_id)
    if not sess["history"]:
        return {"panel": {}}

    from google import genai

    # Build a readable transcript
    lines = []
    for msg in sess["history"]:
        speaker = "Nurse" if msg["role"] == "gemini" else "Doctor"
        lines.append(f"{speaker}: {msg['content']}")
    transcript = "\n\n".join(lines)

    prompt = f"""You are reading a conversation between a nurse and a doctor during a clinical case.
Extract ONLY information that has been explicitly stated in the conversation — do not invent anything.

CONVERSATION:
{transcript}

Return a JSON object with exactly these fields (use null for anything not yet mentioned):
{{
  "patient_name": "...",
  "age": "...",
  "sex": "...",
  "chief_complaint": "...",
  "symptoms": ["symptom 1", "symptom 2"],
  "vital_signs": {{
    "blood_pressure": "...",
    "heart_rate": "...",
    "temperature": "...",
    "oxygen_saturation": "...",
    "respiratory_rate": "..."
  }},
  "history": "...",
  "tests": [
    {{"name": "test name", "result": "full result text"}}
  ],
  "imaging": [
    {{"type": "imaging type", "findings": "findings text"}}
  ]
}}

Rules:
- Return ONLY the JSON object, no markdown or explanation
- Only include tests and imaging that have actually been reported in the conversation
- vital_signs fields should be null if not mentioned
- symptoms should be an empty list if none mentioned yet"""

    client   = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)

    import json, re
    text = response.text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)

    try:
        panel = json.loads(text)
    except json.JSONDecodeError:
        panel = {}

    return {"panel": panel}


@app.post("/api/new-case")
def new_case(req: SessionReq):
    _sessions[req.session_id] = _new_session()
    return {"ok": True}


@app.post("/api/similar-case")
def similar_case(req: SessionReq):
    """Reset session and auto-start a case from the same specialty."""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    old_case  = _get(req.session_id).get("case")
    specialty = old_case.get("specialty") if old_case else None
    old_id    = old_case.get("id") if old_case else None

    _sessions[req.session_id] = _new_session()
    sess = _sessions[req.session_id]

    if specialty:
        pool = [c for c in filter_cases(specialty=specialty) if c["id"] != old_id]
    else:
        pool = [c for c in get_all_cases() if c["id"] != old_id]

    case = _assign_random_name(random.choice(pool) if pool else random.choice(get_all_cases()))

    sess["case"]    = case
    sess["history"] = []

    opening = get_opening_message(API_KEY, case)
    sess["history"].append({"role": "gemini", "content": opening})

    return {
        "response":       opening,
        "case":           _case_meta(case),
        "revealed":       {},
        "newly_revealed": {},
        "complete":       False,
        "image_url":      None,
    }


class SearchReq(BaseModel):
    query: str


@app.post("/api/search-cases")
def search_cases_api(req: SearchReq):
    """Return top 10 cases ranked by relevance to a text query.

    Strategy:
      1. VectorAI DB semantic search (Gemini embeddings) — if cases are indexed
      2. Gemini LLM ranking — fallback
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    results = _vector_search_cases(req.query, 10)
    return {
        "cases": [
            {**_case_meta(c), "score": c.get("_score", 50)}
            for c in results
        ]
    }


@app.post("/api/search-by-image")
async def search_by_image_api(file: UploadFile = File(...)):
    """
    Accept a medical image and return top 8 matching cases.

    Strategy (in order):
      1. BiomedCLIP vector search — if images are indexed in VectorAI DB
      2. Gemini vision description → text ranking (fallback)
    """
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    import tempfile
    image_bytes  = await file.read()
    content_type = file.content_type or "image/jpeg"
    suffix       = Path(file.filename or "img.jpg").suffix or ".jpg"

    # ── 1. Try BiomedCLIP vector search ───────────────────────────────────
    try:
        from app.medical_image_encoder import COLLECTION_NAME
        from cortex import CortexClient

        with CortexClient("localhost:50051") as _c:
            img_count = _c.count(COLLECTION_NAME) if _c.has_collection(COLLECTION_NAME) else 0

        if img_count > 0:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(image_bytes)
                tmp_path = Path(tmp.name)

            try:
                encoder = _get_biomed_encoder()
                with CortexClient("localhost:50051") as client:
                    results = encoder.search_similar(client, str(tmp_path), top_k=8)
            finally:
                tmp_path.unlink(missing_ok=True)

            cases, seen = [], set()
            for r in results:
                case_id = r.payload.get("case_id")
                if case_id and case_id not in seen:
                    case = get_case_by_id(case_id)
                    if case:
                        seen.add(case_id)
                        c = dict(case)
                        c["_score"] = round(r.score * 100, 1)
                        cases.append(c)
            cases.sort(key=lambda c: c["_score"], reverse=True)

            if cases:
                return {
                    "description": "BiomedCLIP vector similarity search",
                    "cases": [{**_case_meta(c), "score": c["_score"]} for c in cases],
                }
    except Exception:
        pass  # fall through to Gemini vision

    # ── 2. Fallback: Gemini vision description → Gemini ranking ───────────
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=API_KEY)
    vision_resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            types.Part(inline_data=types.Blob(mime_type=content_type, data=image_bytes)),
            types.Part(text=(
                "Describe this medical image in clinical terms for case matching. "
                "Include: imaging modality, body region, key findings, likely diagnosis. "
                "Be specific and concise (2-3 sentences)."
            )),
        ],
    )
    description = vision_resp.text
    results     = _vector_search_cases(description)

    return {
        "description": description,
        "cases": [
            {**_case_meta(c), "score": c.get("_score", 50)}
            for c in results
        ],
    }


@app.post("/api/index-images")
def index_images_api():
    """
    Index all case images with BiomedCLIP into VectorAI DB.
    Run once (or after adding new cases) to enable vector image search.
    Requires open_clip_torch, torch, pillow.
    """
    try:
        n = index_case_images()
        return {"ok": True, "indexed": n}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class StartCaseReq(BaseModel):
    session_id: str
    case_id:    str


@app.post("/api/start-case")
def start_case(req: StartCaseReq):
    """Start a simulation with a specific case by ID (used from the case finder)."""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    case = get_case_by_id(req.case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    case = _assign_random_name(case)
    sess = _get(req.session_id)
    sess["case"]     = case
    sess["history"]  = []
    sess["revealed"] = set()
    sess["complete"] = False

    opening = get_opening_message(API_KEY, case)
    sess["history"].append({"role": "gemini", "content": opening})

    return {
        "response":       opening,
        "case":           _case_meta(case),
        "revealed":       {},
        "newly_revealed": {},
        "complete":       False,
        "image_url":      None,
    }


# Static files (must be last)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
