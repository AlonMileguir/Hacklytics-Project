"""
Medical Image Similarity Search — Streamlit Web App

Three tabs:
  🔍 Search     — Upload a query image, view top-K similar results
  📥 Add Images — Upload medical images to encode and store in VectorAI DB
  🗂  Browse     — Paginate through all stored images

Run with:
    streamlit run app/web_app.py
"""

import sys
import os
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

# Add project root to sys.path so we can import app.medical_image_encoder
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from app.medical_image_encoder import (
    MedicalImageEncoder,
    setup_collection,
    COLLECTION_NAME,
)
from cortex import CortexClient
from cortex.filters import Filter, Field

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Uploaded images are saved here permanently so search results can show them
UPLOADS_DIR = ROOT / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

PAGE_SIZE = 12  # images per page in Browse tab

MODALITIES = ["xray", "mri", "ct", "pathology", "ultrasound", "other"]

st.set_page_config(
    page_title="Medical Image Search",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading BiomedCLIP model (first run ~30 s)…")
def get_encoder() -> MedicalImageEncoder:
    """Load BiomedCLIP once and reuse across all reruns."""
    return MedicalImageEncoder()


def try_connect(server: str) -> tuple[bool, str]:
    try:
        with CortexClient(server) as client:
            version, _ = client.health_check()
        return True, version
    except Exception as e:
        return False, str(e)


def get_collection_count(server: str) -> int:
    try:
        with CortexClient(server) as client:
            if client.has_collection(COLLECTION_NAME):
                return client.count(COLLECTION_NAME)
    except Exception:
        pass
    return 0


def image_card(payload: dict, score: float | None = None):
    """Render an image thumbnail with score badge and metadata tags."""
    img_path = payload.get("path", "")
    if img_path and Path(img_path).exists():
        st.image(img_path, use_container_width=True)
    else:
        st.markdown(
            "<div style='background:#262730;height:130px;border-radius:8px;"
            "display:flex;align-items:center;justify-content:center;"
            "color:#888;font-size:2rem'>📷</div>",
            unsafe_allow_html=True,
        )

    if score is not None:
        pct = score * 100
        color = "green" if pct >= 85 else "orange" if pct >= 70 else "red"
        st.markdown(f"**Match:** :{color}[{pct:.1f}%]")

    st.caption(payload.get("filename", "unknown"))

    tags = []
    if payload.get("modality"):
        tags.append(f"🏷 {payload['modality']}")
    if payload.get("label"):
        tags.append(f"📋 {payload['label']}")
    if tags:
        st.caption("  ·  ".join(tags))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🏥 Medical Image Search")
    st.caption("BiomedCLIP  ·  Actian VectorAI DB")
    st.divider()

    server = st.text_input("VectorAI DB server", value="localhost:50051")

    ok, version_or_err = try_connect(server)
    if ok:
        st.success("Connected")
        st.caption(f"Server: {version_or_err}")
        db_count = get_collection_count(server)
        st.metric("Images in database", db_count)
    else:
        st.error("Not connected")
        st.caption(version_or_err)
        st.code("docker compose up -d", language="bash")
        db_count = 0

    st.divider()
    st.markdown("**Search settings**")
    top_k = st.slider("Results (top-k)", min_value=1, max_value=20, value=5)
    filter_modality = st.selectbox("Filter by modality", ["All"] + MODALITIES)

    st.divider()
    st.markdown(
        "<small>Model: [BiomedCLIP](https://huggingface.co/microsoft/"
        "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)  ·  512-d COSINE</small>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_search, tab_add, tab_browse = st.tabs(["🔍 Search", "📥 Add Images", "🗂 Browse"])

# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_search:
    st.header("Find Similar Medical Images")
    st.caption(
        "Upload any medical image — BiomedCLIP encodes it in real time and retrieves "
        "the most visually similar images from the database."
    )

    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        query_file = st.file_uploader(
            "Query image",
            type=["jpg", "jpeg", "png", "tiff", "bmp"],
            label_visibility="collapsed",
        )
        if query_file:
            st.image(
                Image.open(query_file).convert("RGB"),
                caption="Query image",
                use_container_width=True,
            )
            search_btn = st.button(
                "🔍 Search",
                type="primary",
                use_container_width=True,
                disabled=not ok,
            )
            if not ok:
                st.warning("Connect the database first.")
        else:
            st.info("⬆️ Upload a medical image to search")
            search_btn = False

    with col_right:
        if query_file and search_btn:
            if db_count == 0:
                st.warning(
                    "The database is empty. "
                    "Add images first using the **Add Images** tab."
                )
            else:
                suffix = Path(query_file.name).suffix or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(query_file.getbuffer())
                    tmp_path = tmp.name

                try:
                    with st.spinner("Encoding and searching…"):
                        encoder = get_encoder()
                        filter_obj = (
                            Filter().must(Field("modality").eq(filter_modality))
                            if filter_modality != "All"
                            else None
                        )
                        with CortexClient(server) as client:
                            results = encoder.search_similar(
                                client,
                                tmp_path,
                                top_k=top_k,
                                filter=filter_obj,
                            )

                    if not results:
                        st.info(
                            "No results found. "
                            "Try changing the modality filter or add more images."
                        )
                    else:
                        st.subheader(f"Top {len(results)} matches")
                        grid = st.columns(3)
                        for i, r in enumerate(results):
                            with grid[i % 3]:
                                image_card(r.payload or {}, score=r.score)

                except Exception as e:
                    st.error(f"Search failed: {e}")
                finally:
                    os.unlink(tmp_path)

        elif query_file and not search_btn:
            st.info("Click **Search** to find similar images.")

# ═══════════════════════════════════════════════════════════════════════════════
# ADD IMAGES TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_add:
    st.header("Add Images to Database")
    st.caption(
        "Upload medical images to encode with BiomedCLIP and store in VectorAI DB. "
        "Images are saved locally so they can be shown in search results."
    )

    if not ok:
        st.error("Connect to VectorAI DB to add images.")
    else:
        col_form, col_preview = st.columns([1, 1], gap="large")

        with col_form:
            files = st.file_uploader(
                "Select images",
                type=["jpg", "jpeg", "png", "tiff", "bmp"],
                accept_multiple_files=True,
                label_visibility="collapsed",
            )

            modality_sel = st.selectbox("Modality", ["(none)"] + MODALITIES)
            label_input = st.text_input(
                "Label / Diagnosis (optional)",
                placeholder="e.g. pneumonia, normal, fracture",
            )

            with st.expander("Extra metadata fields"):
                c1, c2 = st.columns(2)
                ek1 = c1.text_input("Key", key="ek1", placeholder="patient_id")
                ev1 = c2.text_input("Value", key="ev1", placeholder="P001")
                ek2 = c1.text_input("Key", key="ek2", placeholder="hospital")
                ev2 = c2.text_input("Value", key="ev2", placeholder="City Medical")

            encode_btn = st.button(
                f"⚡ Encode & Store {len(files)} image(s)"
                if files
                else "⚡ Encode & Store",
                type="primary",
                use_container_width=True,
                disabled=not files,
            )

        with col_preview:
            if files:
                st.caption(f"{len(files)} image(s) selected")
                pcols = st.columns(min(3, len(files)))
                for i, f in enumerate(files[:6]):
                    with pcols[i % 3]:
                        st.image(
                            Image.open(f).convert("RGB"),
                            caption=f.name,
                            use_container_width=True,
                        )
                if len(files) > 6:
                    st.caption(f"… and {len(files) - 6} more")
            else:
                st.info("⬆️ Select images to preview them here")

        if encode_btn and files:
            # Build shared metadata
            metadata: dict = {}
            if modality_sel != "(none)":
                metadata["modality"] = modality_sel
            if label_input:
                metadata["label"] = label_input
            if ek1 and ev1:
                metadata[ek1] = ev1
            if ek2 and ev2:
                metadata[ek2] = ev2

            encoder = get_encoder()

            # Ensure collection exists and get base ID
            with CortexClient(server) as client:
                setup_collection(client)
                base_id = client.count(COLLECTION_NAME)

            progress_bar = st.progress(0.0, text="Starting…")
            success_count = 0
            failures: list[str] = []

            for i, file in enumerate(files):
                progress_bar.progress(i / len(files), text=f"Encoding {file.name}…")

                # Save to uploads/ permanently (so search results can show the image)
                dest = UPLOADS_DIR / file.name
                if dest.exists():
                    dest = UPLOADS_DIR / f"{dest.stem}_{base_id + i}{dest.suffix}"
                dest.write_bytes(file.getbuffer())

                try:
                    vector = encoder.encode_image(str(dest))
                    payload = {
                        "filename": file.name,
                        "path": str(dest),
                        "format": dest.suffix.lower(),
                        **metadata,
                    }
                    with CortexClient(server) as client:
                        client.upsert(
                            COLLECTION_NAME,
                            id=base_id + i,
                            vector=vector.tolist(),
                            payload=payload,
                        )
                    success_count += 1
                except Exception as e:
                    failures.append(f"{file.name}: {e}")

            progress_bar.progress(1.0, text="Done!")

            if success_count:
                st.success(f"✅ Stored {success_count} image(s) in the database!")
            for msg in failures:
                st.warning(f"⚠️ Failed — {msg}")

            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# BROWSE TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_browse:
    st.header("Browse Database")

    if not ok:
        st.error("Connect to VectorAI DB to browse.")
    elif db_count == 0:
        st.info("Database is empty. Add images using the **Add Images** tab.")
    else:
        # Pagination state
        if "browse_page" not in st.session_state:
            st.session_state.browse_page = 1

        total_pages = max(1, (db_count + PAGE_SIZE - 1) // PAGE_SIZE)
        # Clamp in case images were deleted
        st.session_state.browse_page = min(st.session_state.browse_page, total_pages)
        page = st.session_state.browse_page

        # cursor = offset per CRTX-232
        offset = (page - 1) * PAGE_SIZE

        st.caption(f"Page {page} of {total_pages}  ·  {db_count} total images")

        with CortexClient(server) as client:
            records, _ = client.scroll(
                COLLECTION_NAME,
                limit=PAGE_SIZE,
                cursor=offset if offset > 0 else None,
                with_payload=True,
            )

        if records:
            grid = st.columns(4)
            for i, rec in enumerate(records):
                with grid[i % 4]:
                    image_card(rec.payload or {})
        else:
            st.info("No records found on this page.")

        # Pagination controls
        st.divider()
        nav_l, nav_c, nav_r = st.columns([1, 3, 1])

        with nav_l:
            if st.button("← Prev", disabled=(page <= 1), use_container_width=True):
                st.session_state.browse_page -= 1
                st.rerun()

        with nav_c:
            st.markdown(
                f"<p style='text-align:center;padding-top:6px'>"
                f"Page <b>{page}</b> of <b>{total_pages}</b></p>",
                unsafe_allow_html=True,
            )

        with nav_r:
            if st.button(
                "Next →", disabled=(page >= total_pages), use_container_width=True
            ):
                st.session_state.browse_page += 1
                st.rerun()
