"""
Clinical Simulation Engine — powered by Gemini.

The student plays the attending physician.
Gemini plays the patient AND the clinical environment (responds to exam requests,
lab orders, imaging queries) then evaluates the student's diagnosis and treatment.

Flow:
  1. Gemini opens with the patient's chief complaint
  2. Student asks questions / orders investigations (free text)
  3. Special triggers reveal structured info:
       "perform exam" / "examine"     → exam findings
       "order labs" / "blood work"    → lab results
       "view imaging" / "order x-ray" → imaging description + image shown
       "my diagnosis is" / "diagnose" → Gemini evaluates diagnosis
       "my treatment" / "I would treat"→ Gemini evaluates treatment + gives summary
  4. After diagnosis + treatment evaluated → case complete with feedback
"""

import re

# Keywords that trigger structured reveals
EXAM_TRIGGERS = re.compile(
    r"\b(perform|do|conduct|physical|examine|examination|exam|auscult|percuss|palpat|vital)\b",
    re.IGNORECASE,
)
LAB_TRIGGERS = re.compile(
    r"\b(lab|blood|CBC|FBC|CRP|culture|test|panel|result|urine|stool|serum|check|order)\b",
    re.IGNORECASE,
)
IMAGING_TRIGGERS = re.compile(
    r"\b(imaging|x.ray|xray|radiograph|CT|MRI|scan|ultrasound|dermoscopy|biopsy|histol|pathol|view|image)\b",
    re.IGNORECASE,
)
DIAGNOSIS_TRIGGERS = re.compile(
    r"\b(diagnos|impression|assessment|think it is|most likely|differential|I believe|suspect|conclude)\b",
    re.IGNORECASE,
)
TREATMENT_TRIGGERS = re.compile(
    r"\b(treat|manage|prescribe|give|start|initiate|therapy|medication|plan|admit|refer|surgery)\b",
    re.IGNORECASE,
)


def detect_intent(message: str) -> set[str]:
    """Detect what kind of action the student is requesting."""
    intents = set()
    if EXAM_TRIGGERS.search(message):
        intents.add("exam")
    if LAB_TRIGGERS.search(message):
        intents.add("labs")
    if IMAGING_TRIGGERS.search(message):
        intents.add("imaging")
    if DIAGNOSIS_TRIGGERS.search(message):
        intents.add("diagnosis")
    if TREATMENT_TRIGGERS.search(message):
        intents.add("treatment")
    return intents


def build_system_prompt(case: dict, revealed: set) -> str:
    """
    Build the Gemini system prompt for the current state of the simulation.

    revealed: set containing any of {"exam", "labs", "imaging"}
    """
    exam_section = (
        f"PHYSICAL EXAMINATION (already revealed to student):\n{case['exam_findings']}"
        if "exam" in revealed
        else "Physical Examination: NOT YET PERFORMED — reveal only when student examines the patient."
    )

    labs_section = (
        f"LABORATORY RESULTS (already revealed to student):\n{case['labs']}"
        if "labs" in revealed
        else "Laboratory Results: NOT YET ORDERED — reveal only when student orders investigations."
    )

    imaging_section = (
        f"IMAGING FINDINGS (image is displayed to student separately):\n"
        f"Modality: {case['imaging_type']}\n{case['imaging_description']}"
        if "imaging" in revealed
        else f"Imaging: NOT YET ORDERED — reveal the imaging description ({case['imaging_type']}) only when "
             f"the student orders or requests imaging."
    )

    symptoms_list = "\n".join(f"  • {s}" for s in case.get("presenting_symptoms", []))

    # If this is a real MultiCaRe case, include the original text for richer context
    source_context = ""
    if case.get("clinical_text"):
        # Truncate to avoid overloading the context window
        truncated = case["clinical_text"][:3000]
        source_context = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ORIGINAL CASE TEXT (from PubMed Central — for your reference)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{truncated}
[...text may be truncated]
"""

    return f"""You are running a clinical education simulation for medical students.

THE STUDENT IS THE ATTENDING PHYSICIAN. You play two roles:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLE 1 — THE PATIENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Name: {case['patient']['name']}
Age: {case['patient']['age']} years old, {case['patient']['sex']}
Occupation: {case['patient'].get('occupation', 'not specified')}
Chief Complaint: {case['chief_complaint']}
History (you know this — share naturally when asked): {case['history']}
Symptoms the patient experiences:
{symptoms_list}

Speak naturally as this patient using LAY LANGUAGE, not medical terminology.
Be realistic: worried, a little nervous, gives information when asked but doesn't volunteer everything.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ROLE 2 — THE CLINICAL ENVIRONMENT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{exam_section}

{labs_section}

{imaging_section}

When the student performs/requests these investigations, provide the results in a clinical format
(as a nurse or lab system would report them). Do not give results before they are requested.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CORRECT ANSWERS (NEVER REVEAL DIRECTLY)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Correct Diagnosis: {case['diagnosis']}
Correct Treatment: {case['treatment']}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EVALUATION RULES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
When the student states a DIAGNOSIS:
  - Compare to the correct diagnosis
  - If correct: congratulate and explain why the clinical picture supports it
  - If partially correct: acknowledge what's right, guide them toward the full picture
  - If wrong: gently redirect with educational hints, do not just give the answer
  - Use Socratic questioning to help them think through the logic

When the student proposes a TREATMENT plan:
  - Evaluate each element against the correct treatment
  - Point out what is correct, what is missing, what is inappropriate
  - Explain the evidence base briefly
  - After treatment discussion → give a COMPREHENSIVE CASE SUMMARY with key learning points

{source_context}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BEHAVIORAL GUIDELINES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Keep responses concise (2–4 sentences) unless giving evaluation or the case summary
• Never volunteer the diagnosis or treatment — the student must earn it through clinical reasoning
• Be encouraging but rigorous — this is a learning environment, mistakes are expected
• Use ✅ for correct elements, ⚠️ for partially correct, ❌ for incorrect when evaluating
• After full case completion, use 📚 KEY LEARNING POINTS as a section header"""


def get_opening_message(api_key: str, case: dict) -> str:
    """
    Generate the opening patient presentation.
    Called once when a case is first started.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    system_prompt = build_system_prompt(case, revealed=set())

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Please introduce yourself as the patient. State your name, age, and chief complaint "
            "in 2–3 natural sentences, as a real patient would to a doctor."
        ),
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        ),
    )
    return response.text


def send_message(
    api_key: str,
    case: dict,
    history: list[dict],
    student_message: str,
    revealed: set,
) -> str:
    """
    Send the student's message and get Gemini's response.

    history: list of {"role": "student"|"gemini", "content": str}
    revealed: set of already-revealed items {"exam", "labs", "imaging"}
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    system_prompt = build_system_prompt(case, revealed)

    # Build message contents from history
    contents = []
    for msg in history:
        role = "user" if msg["role"] == "student" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])],
            )
        )
    # Add the new student message
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=student_message)],
        )
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
        ),
    )
    return response.text


def get_hint(api_key: str, case: dict, history: list[dict], revealed: set) -> str:
    """
    Ask Gemini for a Socratic hint without giving away the answer.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    system_prompt = build_system_prompt(case, revealed)

    contents = []
    for msg in history:
        role = "user" if msg["role"] == "student" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=msg["content"])])
        )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part(text=
                "I'm not sure what to do next. Can you give me a clinical hint "
                "without telling me the answer?"
            )],
        )
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=contents,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    return response.text


def is_case_complete(history: list[dict]) -> bool:
    """
    Heuristic: case is complete when both a diagnosis and treatment
    have been discussed (look for key learning points section in responses).
    """
    for msg in history:
        if msg["role"] == "gemini" and "KEY LEARNING POINTS" in msg["content"]:
            return True
    return False
