"""
MedCase — Clinical Education Platform
======================================
Medical students play the attending physician.
Gemini plays the patient + clinical environment and evaluates reasoning.

Run with:
    streamlit run app/web_app.py
"""

import sys
import os
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Load .env if present (e.g. GEMINI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

from app.case_manager import (
    get_all_cases,
    filter_cases,
    get_case_by_id,
    get_case_image_path,
    get_unique_specialties,
    get_unique_difficulties,
    index_all_cases,
    is_cases_indexed,
    search_cases,
    is_multicare_available,
)
from app.clinical_sim import (
    get_opening_message,
    send_message,
    get_hint,
    detect_intent,
    is_case_complete,
)
from cortex import CortexClient

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="MedCase — Clinical Education",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

defaults = {
    "active_case": None,        # dict: the current case
    "chat_history": [],         # list of {role, content}
    "revealed": set(),          # {"exam", "labs", "imaging"}
    "case_complete": False,
    "view": "library",          # "library" | "simulation"
    "search_query": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DIFFICULTY_COLOR = {"Beginner": "🟢", "Intermediate": "🟡", "Advanced": "🔴"}
DIFFICULTY_BADGE = {"Beginner": "green", "Intermediate": "orange", "Advanced": "red"}


def try_db_connect(server: str):
    try:
        with CortexClient(server) as c:
            v, _ = c.health_check()
        return True, v
    except Exception as e:
        return False, str(e)


def start_case(case: dict, api_key: str):
    """Initialise session state for a new case and generate the opening message."""
    st.session_state.active_case = case
    st.session_state.chat_history = []
    st.session_state.revealed = set()
    st.session_state.case_complete = False
    st.session_state.view = "simulation"

    with st.spinner("Starting clinical encounter…"):
        opening = get_opening_message(api_key, case)

    st.session_state.chat_history = [{"role": "gemini", "content": opening}]


def reset_to_library():
    st.session_state.active_case = None
    st.session_state.chat_history = []
    st.session_state.revealed = set()
    st.session_state.case_complete = False
    st.session_state.view = "library"


def render_chat_message(role: str, content: str):
    if role == "student":
        with st.chat_message("user", avatar="👨‍⚕️"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤒"):
            st.markdown(content)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🩺 MedCase")
    st.caption("Clinical Education Platform")
    st.divider()

    # API Key — loaded from .env, not shown to user
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("GEMINI_API_KEY not set. Add it to your .env file.")

    st.divider()

    # Data source
    st.markdown("**Case Library**")
    if is_multicare_available():
        n = len(get_all_cases())
        st.success(f"✅ MultiCaRe dataset ({n} cases)")
        st.caption("Real PubMed Central case reports")
    else:
        st.info("📦 Demo cases (9 cases)")
        st.caption(
            "Run `python app/setup_multicare.py --api-key KEY` "
            "to load real MultiCaRe cases."
        )

    st.divider()

    # DB connection
    st.markdown("**VectorAI DB**")
    db_server = st.text_input("Server", value="localhost:50051", label_visibility="collapsed")
    db_ok, db_info = try_db_connect(db_server)
    if db_ok:
        st.success(f"Connected")
        st.caption(db_info)

        # Index cases button
        try:
            with CortexClient(db_server) as c:
                already = is_cases_indexed(c)
        except Exception:
            already = False

        if not already and api_key:
            if st.button("📥 Index cases into DB", use_container_width=True):
                with st.spinner("Indexing cases with Gemini embeddings…"):
                    index_all_cases(api_key, db_server)
                st.success("Cases indexed! Semantic search now available.")
                st.rerun()
        elif already:
            st.caption("✅ Cases indexed — semantic search active")
    else:
        st.error("DB not connected")
        st.caption("Start: `docker compose up -d`")

    st.divider()

    # Navigation
    if st.session_state.view == "simulation" and st.session_state.active_case:
        case = st.session_state.active_case
        st.markdown(f"**Current Case**")
        st.markdown(f"_{case['title']}_")
        st.caption(
            f"{DIFFICULTY_COLOR[case['difficulty']]} {case['difficulty']}  ·  {case['specialty']}"
        )
        st.divider()

        # Revealed status
        st.markdown("**Investigations**")
        for item, label in [("exam", "Physical Exam"), ("labs", "Lab Results"), ("imaging", "Imaging")]:
            if item in st.session_state.revealed:
                st.markdown(f"✅ {label}")
            else:
                st.markdown(f"⬜ {label}")

        st.divider()
        if st.button("← Back to Case Library", use_container_width=True):
            reset_to_library()
            st.rerun()


# ---------------------------------------------------------------------------
# LIBRARY VIEW
# ---------------------------------------------------------------------------

if st.session_state.view == "library":
    st.title("🏥 MedCase — Clinical Education Platform")
    st.markdown(
        "Practice clinical reasoning by taking on the role of the attending physician. "
        "Gemini plays your patient and evaluates your diagnosis and treatment plan."
    )

    # Search bar
    col_search, col_btn = st.columns([4, 1])
    with col_search:
        search_q = st.text_input(
            "Search cases",
            placeholder="e.g. 'chest pain', 'dermatology', 'beginner pneumonia'…",
            label_visibility="collapsed",
        )
    with col_btn:
        do_search = st.button("🔍 Search", use_container_width=True)

    # Filters
    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        specialty_filter = st.selectbox(
            "Specialty", ["All"] + get_unique_specialties()
        )
    with f_col2:
        difficulty_filter = st.selectbox(
            "Difficulty", ["All"] + get_unique_difficulties()
        )
    with f_col3:
        st.markdown("")  # spacer

    st.divider()

    # Resolve which cases to show
    if do_search and search_q and api_key:
        with st.spinner("Searching cases…"):
            cases = search_cases(api_key, search_q, top_k=9, db_server=db_server)
        st.caption(f"Showing {len(cases)} results for _\"{search_q}\"_")
    else:
        cases = filter_cases(specialty_filter, difficulty_filter)
        st.caption(f"{len(cases)} case(s) available")

    if not cases:
        st.info("No cases match your filters.")
    else:
        # 3-column card grid
        cols = st.columns(3)
        for i, case in enumerate(cases):
            with cols[i % 3]:
                img_path = get_case_image_path(case)

                with st.container(border=True):
                    if img_path and img_path.exists():
                        st.image(str(img_path), use_container_width=True)
                    else:
                        st.markdown(
                            "<div style='background:#1e2130;height:100px;border-radius:6px;"
                            "display:flex;align-items:center;justify-content:center;"
                            "color:#888;font-size:1.5rem'>🏥</div>",
                            unsafe_allow_html=True,
                        )

                    st.markdown(f"**{case['title']}**")
                    diff = case["difficulty"]
                    st.markdown(
                        f":{DIFFICULTY_BADGE[diff]}[{DIFFICULTY_COLOR[diff]} {diff}]  ·  "
                        f"_{case['specialty']}_"
                    )
                    st.caption(
                        f"👤 {case['patient']['age']}yo {case['patient']['sex']}  ·  "
                        f"_{case['chief_complaint']}_"
                    )

                    if st.button(
                        "Start Case →",
                        key=f"start_{case['id']}",
                        use_container_width=True,
                        type="primary",
                        disabled=not api_key,
                    ):
                        start_case(case, api_key)
                        st.rerun()


# ---------------------------------------------------------------------------
# SIMULATION VIEW
# ---------------------------------------------------------------------------

elif st.session_state.view == "simulation":
    case = st.session_state.active_case
    if not case:
        reset_to_library()
        st.rerun()

    # ── Header ──────────────────────────────────────────────────────────────
    h_col1, h_col2 = st.columns([3, 1])
    with h_col1:
        st.markdown(f"## 🩺 {case['title']}")
        diff = case["difficulty"]
        st.markdown(
            f":{DIFFICULTY_BADGE[diff]}[{DIFFICULTY_COLOR[diff]} {diff}]  ·  "
            f"_{case['specialty']}_  ·  "
            f"**{case['patient']['age']}yo {case['patient']['sex']}**"
        )
    with h_col2:
        if st.session_state.case_complete:
            st.success("Case Complete ✅")

    st.divider()

    # ── Two-column layout: chat | info panel ────────────────────────────────
    chat_col, info_col = st.columns([2, 1], gap="large")

    # ── INFO PANEL (right) ───────────────────────────────────────────────────
    with info_col:
        st.markdown("### 📋 Patient Chart")

        with st.expander("👤 Patient Info", expanded=True):
            p = case["patient"]
            st.markdown(f"**Name:** {p['name']}")
            st.markdown(f"**Age/Sex:** {p['age']}yo {p['sex']}")
            if p.get("occupation"):
                st.markdown(f"**Occupation:** {p['occupation']}")
            st.markdown(f"**CC:** _{case['chief_complaint']}_")
            st.markdown(f"**History:** {case['history']}")

        # Physical Exam — reveal when unlocked
        with st.expander(
            "🩺 Physical Examination" + (" ✅" if "exam" in st.session_state.revealed else " 🔒"),
            expanded="exam" in st.session_state.revealed,
        ):
            if "exam" in st.session_state.revealed:
                st.markdown(case["exam_findings"])
            else:
                st.caption("Perform a physical examination in the chat to reveal findings.")

        # Lab Results
        with st.expander(
            "🧪 Lab Results" + (" ✅" if "labs" in st.session_state.revealed else " 🔒"),
            expanded="labs" in st.session_state.revealed,
        ):
            if "labs" in st.session_state.revealed:
                st.markdown(case["labs"])
            else:
                st.caption("Order laboratory investigations in the chat to reveal results.")

        # Imaging
        with st.expander(
            f"🖼 {case['imaging_type']}" + (" ✅" if "imaging" in st.session_state.revealed else " 🔒"),
            expanded="imaging" in st.session_state.revealed,
        ):
            if "imaging" in st.session_state.revealed:
                img_path = get_case_image_path(case)
                if img_path and img_path.exists():
                    st.image(str(img_path), caption=case["imaging_type"], use_container_width=True)
                else:
                    st.info("Image not available for this case.")
                st.markdown(f"**Report:** _{case['imaging_description']}_")
            else:
                st.caption(f"Order {case['imaging_type']} in the chat to reveal the image.")

        # Quick-action buttons
        st.markdown("### ⚡ Quick Actions")
        st.caption("Click to send a standard clinical request:")

        qa_cols = st.columns(2)
        actions = [
            ("🩺 Examine Patient", "I'd like to perform a physical examination."),
            ("🧪 Order Labs", "Please order a full blood count, metabolic panel, and relevant labs."),
            (f"🖼 Order {case['imaging_type'].split()[0]}", f"I'd like to order a {case['imaging_type']}."),
            ("💡 Give me a hint", "__HINT__"),
        ]
        for j, (label, msg) in enumerate(actions):
            with qa_cols[j % 2]:
                if st.button(label, use_container_width=True, key=f"qa_{j}"):
                    if msg == "__HINT__":
                        with st.spinner("Getting a hint…"):
                            hint = get_hint(
                                api_key, case, st.session_state.chat_history,
                                st.session_state.revealed
                            )
                        st.session_state.chat_history.append(
                            {"role": "student", "content": "Can you give me a hint?"}
                        )
                        st.session_state.chat_history.append(
                            {"role": "gemini", "content": hint}
                        )
                    else:
                        # Process like a normal chat message
                        intents = detect_intent(msg)
                        new_revealed = st.session_state.revealed | (
                            intents & {"exam", "labs", "imaging"}
                        )
                        with st.spinner("…"):
                            reply = send_message(
                                api_key, case,
                                st.session_state.chat_history,
                                msg,
                                new_revealed,
                            )
                        st.session_state.revealed = new_revealed
                        st.session_state.chat_history.append({"role": "student", "content": msg})
                        st.session_state.chat_history.append({"role": "gemini", "content": reply})

                        if is_case_complete(st.session_state.chat_history):
                            st.session_state.case_complete = True
                    st.rerun()

    # ── CHAT PANEL (left) ────────────────────────────────────────────────────
    with chat_col:
        st.markdown("### 💬 Clinical Encounter")
        st.caption(
            "You are the attending physician. Talk to your patient, order investigations, "
            "then state your diagnosis and treatment plan."
        )

        # Render full chat history
        chat_container = st.container(height=520)
        with chat_container:
            for msg in st.session_state.chat_history:
                render_chat_message(msg["role"], msg["content"])

        # Input
        if not st.session_state.case_complete:
            user_input = st.chat_input(
                "Speak to your patient or order investigations…",
                disabled=not api_key,
            )
            if user_input:
                # Detect intent and update revealed set
                intents = detect_intent(user_input)
                new_revealed = st.session_state.revealed | (intents & {"exam", "labs", "imaging"})

                with st.spinner("…"):
                    reply = send_message(
                        api_key,
                        case,
                        st.session_state.chat_history,
                        user_input,
                        new_revealed,
                    )

                st.session_state.revealed = new_revealed
                st.session_state.chat_history.append(
                    {"role": "student", "content": user_input}
                )
                st.session_state.chat_history.append(
                    {"role": "gemini", "content": reply}
                )

                if is_case_complete(st.session_state.chat_history):
                    st.session_state.case_complete = True

                st.rerun()

        else:
            # Case complete — show end screen
            st.success("🎉 Case complete! Review the feedback above.")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🔄 Try Another Case", type="primary", use_container_width=True):
                    reset_to_library()
                    st.rerun()
            with c2:
                if st.button("🔁 Redo This Case", use_container_width=True):
                    start_case(case, api_key)
                    st.rerun()
