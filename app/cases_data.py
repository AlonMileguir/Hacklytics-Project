"""
Medical case database.

Each case links to a MedMNIST image label that gets downloaded by download_dataset.py.
  imaging_dataset: "pneumonia" | "path" | "derma"
  imaging_label:   folder name inside data/<dataset>/

Cases are loaded by case_manager.py and stored in VectorAI DB.
"""

CASES = [
    # ─── PULMONOLOGY ──────────────────────────────────────────────────────────
    {
        "id": "case_001",
        "title": "Productive Cough and Fever",
        "specialty": "Pulmonology",
        "difficulty": "Beginner",
        "patient": {
            "name": "James Alvarez",
            "age": 45,
            "sex": "male",
            "occupation": "school teacher",
        },
        "chief_complaint": "Productive cough and fever for 3 days",
        "history": (
            "No significant past medical history. Non-smoker. No recent travel or sick contacts "
            "outside the home — wife had a mild cold last week. No known allergies. Takes no medications."
        ),
        "presenting_symptoms": [
            "productive cough with yellow-green sputum",
            "fever 38.9°C",
            "right-sided chest pain worsening with inspiration",
            "fatigue and malaise",
        ],
        "exam_findings": (
            "T 38.9°C, HR 96 bpm, RR 22/min, BP 128/78 mmHg, O2 sat 94% on room air. "
            "Dullness to percussion at right base. Decreased breath sounds with bronchial breathing "
            "at right lower lobe. Egophony positive. No wheezes. No calf tenderness."
        ),
        "labs": (
            "WBC 15.2 ×10⁹/L (neutrophilia 88%), CRP 96 mg/L, Procalcitonin 1.8 ng/mL. "
            "Sputum Gram stain: gram-positive diplococci in pairs. Blood cultures ×2 pending. "
            "BMP normal. LDH 280 U/L. Urine pneumococcal antigen: positive."
        ),
        "imaging_dataset": "pneumonia",
        "imaging_label": "Pneumonia",
        "imaging_type": "Chest X-ray (PA)",
        "imaging_description": (
            "Right lower lobe consolidation with air bronchograms. "
            "No pleural effusion. Cardiac silhouette normal size. Left lung clear. "
            "No hilar adenopathy."
        ),
        "diagnosis": "Community-Acquired Pneumonia (CAP) — Streptococcus pneumoniae",
        "treatment": (
            "Amoxicillin 1g PO TID × 5 days (first-line for non-severe CAP, likely pneumococcal). "
            "Add azithromycin if atypical organisms suspected. Adequate hydration. Paracetamol 1g QID PRN. "
            "CURB-65 score = 0 → outpatient safe. Follow-up in 48–72h. Repeat CXR at 6–8 weeks "
            "to confirm resolution and exclude underlying malignancy."
        ),
        "key_learning_points": [
            "Classic bacterial CAP: fever, productive cough, focal consolidation, elevated WBC/CRP/PCT",
            "Egophony and bronchial breathing on exam = consolidation (not effusion)",
            "CURB-65 guides admission decision: Confusion, Urea >7, RR ≥30, BP <90/60, Age ≥65",
            "Gram-positive diplococci → S. pneumoniae → beta-lactam first line (amoxicillin PO or ampicillin IV)",
            "Always follow up CXR at 6–8 weeks in adults to exclude post-obstructive pneumonia from tumor",
        ],
        "description": (
            "45-year-old male teacher with 3-day history of fever, productive cough with yellow sputum, "
            "right-sided pleuritic chest pain, and right lower lobe consolidation on X-ray with gram-positive "
            "diplococci on sputum — community-acquired pneumonia requiring antibiotic treatment"
        ),
    },
    {
        "id": "case_002",
        "title": "Routine Pre-Employment Medical",
        "specialty": "General Medicine",
        "difficulty": "Beginner",
        "patient": {
            "name": "Linda Chen",
            "age": 55,
            "sex": "female",
            "occupation": "accountant",
        },
        "chief_complaint": "Routine pre-employment health check — no symptoms",
        "history": (
            "No chronic illnesses. Never smoked. No respiratory symptoms. No family history of lung or "
            "cardiac disease. Previous chest X-ray 5 years ago was normal. Drinks 1–2 glasses of wine weekly."
        ),
        "presenting_symptoms": [
            "No active respiratory symptoms",
            "mild work-related stress",
            "occasional mild headaches",
        ],
        "exam_findings": (
            "T 36.6°C, HR 72, RR 16, BP 132/82, O2 sat 99% RA. "
            "Chest clear to auscultation bilaterally. No added sounds. Normal respiratory excursion. "
            "No lymphadenopathy. No peripheral oedema. BMI 26."
        ),
        "labs": (
            "CBC normal. BMP normal. Total cholesterol 210 mg/dL, LDL 138, HDL 52, TG 102. "
            "HbA1c 5.4%. TSH normal. Urine dipstick negative."
        ),
        "imaging_dataset": "pneumonia",
        "imaging_label": "Normal",
        "imaging_type": "Chest X-ray (PA)",
        "imaging_description": (
            "Clear lung fields bilaterally. Normal cardiac silhouette (CTR <0.5). "
            "No hilar enlargement. No pleural effusion or pneumothorax. No bony abnormalities. "
            "Trachea midline."
        ),
        "diagnosis": "Normal chest X-ray. Borderline elevated LDL — lifestyle modification warranted.",
        "treatment": (
            "No pulmonary treatment needed. Lifestyle counseling: Mediterranean diet, aerobic exercise "
            "150 min/week. Calculate 10-year ASCVD risk (ACC/AHA calculator) — statin only if ≥7.5%. "
            "Recheck lipids in 6 months. Annual BP monitoring given borderline hypertension."
        ),
        "key_learning_points": [
            "Normal CXR features: clear lung fields, CTR <0.5, no effusion, no hilar adenopathy, trachea midline",
            "Routine screens reveal incidental findings — always address them (opportunistic prevention)",
            "Lipid management: lifestyle first, calculate ASCVD risk before initiating statin therapy",
            "Borderline hypertension (130–139/80–89): lifestyle modification before medication in low-risk patients",
        ],
        "description": (
            "55-year-old asymptomatic female for routine pre-employment check. Normal chest X-ray. "
            "Borderline hyperlipidemia and mild hypertension found incidentally — cardiovascular risk assessment needed"
        ),
    },

    # ─── DERMATOLOGY ──────────────────────────────────────────────────────────
    {
        "id": "case_003",
        "title": "Changing Pigmented Skin Lesion",
        "specialty": "Dermatology / Oncology",
        "difficulty": "Intermediate",
        "patient": {
            "name": "Michael Torres",
            "age": 52,
            "sex": "male",
            "occupation": "outdoor construction worker",
        },
        "chief_complaint": "Mole on back that has been growing and changing color for 6 months",
        "history": (
            "Fair skin, Fitzpatrick type II. Heavy sun exposure throughout career. No sunscreen use. "
            "One blistering sunburn in his 20s. No family history of melanoma. "
            "No prior skin cancers. No immunosuppression."
        ),
        "presenting_symptoms": [
            "pigmented lesion on upper back — asymmetric, multiple colors",
            "lesion has grown from ~5mm to ~12mm over 6 months",
            "occasional mild itching",
            "no bleeding yet",
        ],
        "exam_findings": (
            "12mm lesion on upper back. Markedly asymmetric. Irregular, notched borders. "
            "Multiple colors: tan, brown, dark brown, with focal areas of blue-black. "
            "No satellite lesions. No palpable regional lymphadenopathy (axillary, cervical). "
            "Remainder of skin: multiple common nevi, no other atypical lesions."
        ),
        "labs": (
            "Not yet indicated pre-biopsy. Post-diagnosis: LFTs, LDH (prognostic), "
            "BRAF/NRAS/KIT mutation testing on specimen."
        ),
        "imaging_dataset": "derma",
        "imaging_label": "Melanoma",
        "imaging_type": "Dermoscopy image",
        "imaging_description": (
            "Atypical pigment network with irregular meshwork. Regression structures (white scar-like areas). "
            "Irregular blotches of dark pigment. Blue-white veil present. "
            "No benign features. High suspicion for melanoma."
        ),
        "diagnosis": "Malignant Melanoma — superficial spreading type (clinical stage to be determined post-excision)",
        "treatment": (
            "Urgent excisional biopsy with 2mm margins (diagnostic). "
            "If confirmed melanoma: wide local excision with margins based on Breslow thickness "
            "(1cm for <1mm, 1–2cm for 1–2mm, 2cm for >2mm). "
            "Sentinel lymph node biopsy if Breslow >1mm or ulceration. "
            "Refer to melanoma MDT. Staging CT (chest/abdomen/pelvis) ± PET-CT for thick lesions. "
            "BRAF mutation → consider targeted therapy (vemurafenib + cobimetinib) if metastatic. "
            "Immunotherapy (pembrolizumab or nivolumab) for advanced disease."
        ),
        "key_learning_points": [
            "ABCDE criteria: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolution",
            "Blue-white veil on dermoscopy is highly specific for melanoma",
            "Breslow thickness (depth in mm) is the single most important prognostic factor",
            "BRAF V600E mutation in ~50% of melanomas — actionable for targeted therapy",
            "Never shave-biopsy a suspected melanoma — excisional biopsy preserves Breslow measurement",
        ],
        "description": (
            "52-year-old male construction worker with 12mm asymmetric, multi-colored, growing back lesion "
            "with dermoscopic blue-white veil and atypical network — malignant melanoma requiring urgent excision"
        ),
    },
    {
        "id": "case_004",
        "title": "Non-Healing Facial Lesion in Elderly Patient",
        "specialty": "Dermatology",
        "difficulty": "Beginner",
        "patient": {
            "name": "Dorothy Hendricks",
            "age": 74,
            "sex": "female",
            "occupation": "retired farmer",
        },
        "chief_complaint": "Sore on the nose that won't heal for 4 months",
        "history": (
            "Lifelong outdoor worker, extensive UV exposure. Fitzpatrick skin type I (always burned, never tanned). "
            "No prior skin cancers. No immunosuppression. Takes amlodipine for hypertension. "
            "First noticed it as a small 'pimple' that kept returning."
        ),
        "presenting_symptoms": [
            "pearly, rolled-border lesion on left nasal ala",
            "central ulceration with occasional bleeding",
            "present and slowly growing for 4 months",
            "no pain",
        ],
        "exam_findings": (
            "8mm pearly papule on left nasal ala. Rolled, translucent borders with telangiectasias. "
            "Central ulceration with crusting. No satellite lesions. "
            "No regional lymphadenopathy. Extensive solar lentigines and actinic keratoses elsewhere on face."
        ),
        "labs": "None required pre-biopsy for suspected BCC.",
        "imaging_dataset": "derma",
        "imaging_label": "Basal_cell_carcinoma",
        "imaging_type": "Dermoscopy image",
        "imaging_description": (
            "Arborizing (tree-like) blood vessels on a white-pink background. "
            "Shiny white streaks (chrysalis structures). Blue-gray ovoid nests. "
            "Classic dermoscopic features of nodular basal cell carcinoma."
        ),
        "diagnosis": "Nodular Basal Cell Carcinoma (BCC) of the nose",
        "treatment": (
            "Mohs micrographic surgery (first-line for facial/high-risk BCC) — "
            "highest cure rate (99%), maximal tissue conservation. "
            "Alternatively: surgical excision with 4mm clinical margins + frozen section. "
            "Radiation therapy if surgery not feasible. "
            "For superficial BCC: topical imiquimod or 5-FU (not nodular type). "
            "Vismodegib (Hedgehog pathway inhibitor) for locally advanced/metastatic BCC."
        ),
        "key_learning_points": [
            "BCC is the most common skin cancer. Sun exposure is the primary risk factor",
            "Classic triad: pearly papule + rolled borders + telangiectasias + central ulceration",
            "Dermoscopy: arborizing vessels + blue-gray ovoid nests = nodular BCC",
            "Mohs surgery for high-risk locations (face, H-zone) — best cure rate with tissue sparing",
            "BCC rarely metastasizes but is locally destructive — treat early on face",
        ],
        "description": (
            "74-year-old retired farmer with 4-month non-healing pearly ulcerated nasal lesion, "
            "arborizing vessels on dermoscopy, classic for nodular basal cell carcinoma requiring Mohs surgery"
        ),
    },
    {
        "id": "case_005",
        "title": "Worried Patient with Mole",
        "specialty": "Dermatology",
        "difficulty": "Beginner",
        "patient": {
            "name": "Sarah Kim",
            "age": 28,
            "sex": "female",
            "occupation": "graduate student",
        },
        "chief_complaint": "Worried about a mole on her leg — saw a news story about skin cancer",
        "history": (
            "No significant history. Fitzpatrick type III. Occasional sunburn. Uses sunscreen intermittently. "
            "No family history of melanoma. No prior skin biopsies. Anxiety disorder, on sertraline."
        ),
        "presenting_symptoms": [
            "6mm round dark-brown lesion on left calf, unchanged for years",
            "no itching, bleeding, or color change",
            "patient very anxious about cancer",
        ],
        "exam_findings": (
            "6mm uniformly pigmented brown macule, left calf. Symmetric. Smooth, well-defined border. "
            "Homogeneous brown color. No satellites. No regional lymphadenopathy. "
            "Skin otherwise unremarkable with a few similar-looking lesions elsewhere."
        ),
        "labs": "None indicated.",
        "imaging_dataset": "derma",
        "imaging_label": "Melanocytic_nevi",
        "imaging_type": "Dermoscopy image",
        "imaging_description": (
            "Uniform pigment network of regular meshwork. No atypical features. "
            "No regression structures, blue-white veil, or atypical vessels. "
            "Symmetric distribution. Classic benign melanocytic nevus pattern."
        ),
        "diagnosis": "Benign melanocytic nevus (common mole). No features of malignancy.",
        "treatment": (
            "Reassurance — no treatment needed. Educate on ABCDE criteria for self-monitoring. "
            "Total body skin exam annually if multiple nevi or risk factors. "
            "Advise sun protection: SPF 30+, protective clothing, avoid peak sun hours. "
            "Return if any change in size, shape, color, or new symptoms. "
            "Address anxiety: validate concern, explain dermoscopy findings clearly."
        ),
        "key_learning_points": [
            "ABCDE: this lesion is Symmetric, regular Border, uniform Color, <6mm stable Diameter, no Evolution",
            "Dermoscopy: regular pigment network without atypical features = benign nevus, no biopsy needed",
            "Reassurance and education are key — patient anxiety is valid and common",
            "Distinguish clinical features of common nevi vs atypical/dysplastic nevi vs melanoma",
            "Annual skin checks recommended for patients with >50 nevi or family history of melanoma",
        ],
        "description": (
            "28-year-old anxious student with stable 6mm uniform brown calf lesion, "
            "regular dermoscopic pigment network — benign melanocytic nevus requiring reassurance not excision"
        ),
    },
    {
        "id": "case_006",
        "title": "Rough Scaly Patch on Sun-Damaged Skin",
        "specialty": "Dermatology",
        "difficulty": "Intermediate",
        "patient": {
            "name": "Robert Steele",
            "age": 68,
            "sex": "male",
            "occupation": "retired naval officer",
        },
        "chief_complaint": "Rough, scaly patches on forehead and scalp — won't go away",
        "history": (
            "Decades of intense sun exposure during naval career. Fitzpatrick type II. "
            "Multiple previous actinic keratoses treated with cryotherapy. "
            "One prior squamous cell carcinoma in situ on left ear, excised 2 years ago. "
            "Hypertension and T2DM on metformin and lisinopril."
        ),
        "presenting_symptoms": [
            "multiple rough, scaly, sandpaper-like patches on forehead and scalp",
            "mild tenderness when touched",
            "present for months, some new, some returning after previous treatment",
        ],
        "exam_findings": (
            "Extensive actinic damage: solar lentigines, diffuse skin atrophy. "
            "4 discrete erythematous, scaly plaques on forehead (5–15mm) and 2 on scalp vertex. "
            "No ulceration. No induration suggesting invasive SCC. "
            "No cervical or parotid lymphadenopathy."
        ),
        "labs": "None required.",
        "imaging_dataset": "derma",
        "imaging_label": "Actinic_keratoses",
        "imaging_type": "Dermoscopy image",
        "imaging_description": (
            "Strawberry pattern: pseudo-network of vessels surrounding follicular openings. "
            "White to yellow surface scale. Erythematous background. "
            "No features of invasive SCC (ulceration, irregular vessels). "
            "Consistent with actinic keratosis."
        ),
        "diagnosis": "Multiple actinic keratoses (AK) — pre-malignant, risk of progression to SCC",
        "treatment": (
            "Field therapy for multiple lesions: topical 5-fluorouracil (Efudix) 5% BID × 2–4 weeks "
            "OR imiquimod 5% 3×/week × 16 weeks OR ingenol mebutate gel. "
            "Individual thick lesions: cryotherapy with liquid nitrogen (2 freeze-thaw cycles). "
            "Photodynamic therapy (PDT) excellent for field cancerization on scalp/face. "
            "Daily broad-spectrum SPF 50+ and sun-protective clothing. "
            "3–6 monthly surveillance. Biopsy any indurated, ulcerated, or rapidly growing lesion."
        ),
        "key_learning_points": [
            "Actinic keratosis = pre-malignant; ~1% per lesion per year risk of progression to SCC",
            "Field cancerization: treat the whole area, not just visible lesions",
            "Dermoscopy strawberry pattern (vessels around follicles) is characteristic of AK",
            "Induration or ulceration in an AK → biopsy to exclude SCC",
            "High-risk patients (immunosuppressed, transplant recipients) need aggressive management",
        ],
        "description": (
            "68-year-old retired naval officer with extensive sun damage, multiple actinic keratoses "
            "on forehead and scalp with dermoscopic strawberry pattern — field therapy and surveillance needed"
        ),
    },
    {
        "id": "case_007",
        "title": "Thickened Warty Lesion on Trunk",
        "specialty": "Dermatology",
        "difficulty": "Beginner",
        "patient": {
            "name": "Patricia Nguyen",
            "age": 63,
            "sex": "female",
            "occupation": "librarian",
        },
        "chief_complaint": "Brownish 'stuck-on' warty lesion on chest — growing slowly for years",
        "history": (
            "No significant past medical history. Minimal sun exposure (indoor work). "
            "No family history of skin cancer. She has several similar lesions elsewhere. "
            "Reports the lesion has been there for at least 10 years, recently got 'bumpier'."
        ),
        "presenting_symptoms": [
            "waxy, brown-black, 'stuck-on' appearing plaque on chest",
            "2cm in diameter, present for >10 years",
            "occasional mild itching",
            "no bleeding, no change in color recently",
        ],
        "exam_findings": (
            "2cm well-defined, raised, waxy brown-black verrucous plaque on chest. "
            "Stuck-on appearance with visible keratin plugs (horn pseudocysts). "
            "Sharp borders. Multiple similar lesions on trunk and arms. "
            "No lymphadenopathy."
        ),
        "labs": "None required.",
        "imaging_dataset": "derma",
        "imaging_label": "Benign_keratosis",
        "imaging_type": "Dermoscopy image",
        "imaging_description": (
            "Multiple milia-like cysts (white round structures). Comedo-like openings. "
            "Fissures and ridges giving a brain-like pattern. Hairpin vessels. "
            "Classic seborrhoeic keratosis pattern. No atypical pigment network or melanoma features."
        ),
        "diagnosis": "Seborrhoeic keratosis (benign keratosis) — completely benign, no malignant potential",
        "treatment": (
            "No treatment required — benign condition. "
            "If cosmetically troublesome or frequently irritated: cryotherapy, curettage, or laser ablation. "
            "Hydrogen peroxide 40% topical solution (Eskata) — FDA approved for SK removal. "
            "Important: reassure patient. Biopsy only if diagnosis uncertain or suspicious features arise. "
            "Sign of Leser-Trélat: sudden eruption of multiple SKs may signal internal malignancy — "
            "relevant only for rapid new onset of many lesions."
        ),
        "key_learning_points": [
            "Seborrhoeic keratosis: most common benign skin tumor in adults over 50",
            "Stuck-on appearance, horn pseudocysts, and comedo-like openings are characteristic",
            "Dermoscopy: milia-like cysts + comedo-like openings = SK (highly specific)",
            "No malignant potential — treatment is purely cosmetic",
            "Sign of Leser-Trélat: abrupt onset of many SKs can be a paraneoplastic sign",
        ],
        "description": (
            "63-year-old librarian with a 10-year-old waxy, stuck-on, brown-black verrucous chest plaque "
            "and classic dermoscopic milia-like cysts — benign seborrhoeic keratosis requiring reassurance"
        ),
    },

    # ─── GASTROENTEROLOGY / PATHOLOGY ─────────────────────────────────────────
    {
        "id": "case_008",
        "title": "Rectal Bleeding and Weight Loss",
        "specialty": "Gastroenterology / Oncology",
        "difficulty": "Advanced",
        "patient": {
            "name": "Frank O'Brien",
            "age": 61,
            "sex": "male",
            "occupation": "retired",
        },
        "chief_complaint": "Bright red blood in stool and 8kg unintentional weight loss over 3 months",
        "history": (
            "No prior colonoscopy. Father died of colorectal cancer at age 65. "
            "Longstanding constipation alternating with diarrhea. "
            "High red meat, low fiber diet. Ex-smoker (20 pack-years). "
            "No IBD history. Hypertension on ramipril."
        ),
        "presenting_symptoms": [
            "bright red rectal bleeding mixed with stool",
            "8kg weight loss over 3 months",
            "change in bowel habits — pencil-thin stools",
            "fatigue and exertional dyspnoea",
            "lower abdominal discomfort",
        ],
        "exam_findings": (
            "Pale conjunctivae. Abdomen: mild left lower quadrant tenderness, no palpable mass. "
            "Digital rectal exam: blood on glove, no palpable rectal mass. "
            "No hepatomegaly. No palpable lymphadenopathy."
        ),
        "labs": (
            "Hb 9.2 g/dL (microcytic, iron-deficiency pattern), MCV 72, ferritin 6 ng/mL. "
            "CEA 18.4 ng/mL (elevated, normal <5). CA19-9 mildly elevated. "
            "LFTs normal. Renal function normal. Faecal occult blood: strongly positive."
        ),
        "imaging_dataset": "path",
        "imaging_label": "Colorectal_cancer",
        "imaging_type": "Colorectal biopsy histopathology (H&E stain)",
        "imaging_description": (
            "Moderately differentiated colonic adenocarcinoma. Irregular glandular structures "
            "with nuclear pleomorphism and frequent mitoses. Desmoplastic stromal reaction. "
            "Mucin production present. Lymphovascular invasion cannot be excluded on this specimen."
        ),
        "diagnosis": "Colorectal Adenocarcinoma (CRC) — staging required",
        "treatment": (
            "Urgent staging: CT chest/abdomen/pelvis with contrast. MRI pelvis if rectal primary. "
            "Colonoscopy for full bowel assessment. Surgical referral (colorectal MDT). "
            "Resectable colon cancer: surgical resection (hemicolectomy) + adjuvant FOLFOX if stage III. "
            "Rectal cancer: neoadjuvant chemoradiotherapy → total mesorectal excision (TME). "
            "Microsatellite instability (MSI) testing for Lynch syndrome and immunotherapy eligibility. "
            "KRAS/NRAS/BRAF testing for metastatic disease (anti-EGFR eligibility)."
        ),
        "key_learning_points": [
            "Red flags for CRC: rectal bleeding + weight loss + change in bowel habit + microcytic anaemia",
            "CEA is a prognostic and surveillance marker, not a screening test",
            "Iron-deficiency anaemia in men or post-menopausal women = colorectal cancer until proven otherwise",
            "Histology: adenocarcinoma (glandular structures, mucin production, nuclear atypia)",
            "MSI testing: high MSI = better prognosis + Lynch syndrome screening + immunotherapy response",
        ],
        "description": (
            "61-year-old male with 3-month history of rectal bleeding, 8kg weight loss, pencil stools, "
            "iron-deficiency anaemia, and elevated CEA — colorectal adenocarcinoma on biopsy requiring staging"
        ),
    },
    {
        "id": "case_009",
        "title": "Chronic Diarrhea with Mucus and Urgency",
        "specialty": "Gastroenterology",
        "difficulty": "Intermediate",
        "patient": {
            "name": "Aisha Patel",
            "age": 34,
            "sex": "female",
            "occupation": "software engineer",
        },
        "chief_complaint": "Chronic diarrhea with mucus and urgency for 6 months, worsening",
        "history": (
            "No significant prior medical history. No recent travel. Non-smoker. No antibiotics recently. "
            "Family history: mother has Crohn's disease. Stress at work has increased. "
            "Reports episodes of bloody diarrhea twice in the past month."
        ),
        "presenting_symptoms": [
            "6+ loose stools per day with urgency",
            "mucus and occasional blood in stool",
            "crampy lower abdominal pain relieved by defecation",
            "fatigue, low-grade fever (37.8°C)",
            "mouth ulcers (aphthous)",
        ],
        "exam_findings": (
            "T 37.8°C, HR 88, BP 118/74. Mild pallor. "
            "Abdomen: mild diffuse tenderness worse in lower quadrants. No peritonism. "
            "Bowel sounds active. No palpable mass. No perianal disease. "
            "Aphthous ulcers in oral cavity. No arthritis or rash today."
        ),
        "labs": (
            "WBC 11.8 (mild leukocytosis), Hb 10.6 (normocytic), CRP 48, ESR 62. "
            "Faecal calprotectin 820 μg/g (markedly elevated, normal <50). "
            "Stool cultures negative. C. difficile negative. Iron studies: low ferritin. "
            "Albumin 32 g/L (low). ASCA positive, pANCA negative."
        ),
        "imaging_dataset": "path",
        "imaging_label": "Lymphocytes",
        "imaging_type": "Colonic mucosal biopsy (H&E stain)",
        "imaging_description": (
            "Dense lymphocytic and plasmacytic infiltrate in the lamina propria. "
            "Crypt architectural distortion with branching. Cryptitis and crypt abscesses present. "
            "No granulomas identified. Goblet cell depletion. "
            "Consistent with active chronic inflammatory bowel disease."
        ),
        "diagnosis": "Inflammatory Bowel Disease — Crohn's disease (ileocolonic pattern likely), moderate-to-severe",
        "treatment": (
            "Induction: oral prednisolone 40mg/day tapering over 8 weeks (or IV hydrocortisone if severe). "
            "Maintenance: azathioprine 2–2.5 mg/kg/day (check TPMT before starting). "
            "Biologic if steroid-dependent/refractory: anti-TNF (infliximab or adalimumab) ± immunomodulator. "
            "Nutritional support: iron IV, vitamin D, folate supplementation. "
            "Smoking cessation (worsens Crohn's). GI MDT referral. "
            "MRI enterography to assess small bowel extent. Colonoscopy for full assessment."
        ),
        "key_learning_points": [
            "Faecal calprotectin >200 strongly suggests organic bowel disease (IBD vs IBS distinction)",
            "ASCA+ / pANCA- pattern favors Crohn's disease over ulcerative colitis",
            "Biopsy: crypt distortion + lymphocytic infiltrate without granulomas is consistent with Crohn's",
            "Granulomas in biopsy are pathognomonic of Crohn's but only found in ~30%",
            "Check TPMT enzyme activity before starting azathioprine to avoid myelosuppression",
        ],
        "description": (
            "34-year-old female with 6-month history of bloody diarrhea, elevated calprotectin, "
            "lymphocytic mucosal infiltrate with crypt distortion — Crohn's disease requiring immunomodulatory therapy"
        ),
    },
]
