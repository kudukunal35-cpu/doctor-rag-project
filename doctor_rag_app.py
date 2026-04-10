import re
import html
import warnings
from pathlib import Path
from dataclasses import dataclass, field

import gradio as gr
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

warnings.filterwarnings("ignore")

DATA_PATH = Path("data")
VECTORSTORE_PATH = Path("vectorstore")
DATA_PATH.mkdir(exist_ok=True)
VECTORSTORE_PATH.mkdir(exist_ok=True)


@dataclass
class Config:
    DATA_PATH: Path = DATA_PATH
    VECTORSTORE_PATH: Path = VECTORSTORE_PATH
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    CHUNK_SIZE: int = 700
    CHUNK_OVERLAP: int = 120
    CHUNK_SEPARATORS: list = field(default_factory=lambda: [
        "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""
    ])
    EMBEDDING_BATCH_SIZE: int = 16
    TOP_K: int = 5


config = Config()


def extract_patient_id(text: str):
    match = re.search(r"\bP\d+\b", text.upper())
    return match.group(0) if match else None


def extract_topic_keyword(text: str):
    text = text.lower()
    topics = [
        "fever", "diabetes", "blood pressure", "bp", "cold",
        "hypertension", "heart", "chest pain", "precautions"
    ]
    for topic in topics:
        if topic in text:
            return topic
    return None


def load_all_sources():
    all_documents = []

    txt_files = list(config.DATA_PATH.glob("*.txt"))
    if not txt_files:
        return all_documents

    loader = DirectoryLoader(
        str(config.DATA_PATH),
        glob="*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )
    txt_docs = loader.load()

    for doc in txt_docs:
        if "source" in doc.metadata:
            doc.metadata["filename"] = Path(doc.metadata["source"]).name
        else:
            doc.metadata["filename"] = "Unknown"

        patient_match = re.search(r"Patient ID:\s*(P\d+)", doc.page_content, re.IGNORECASE)
        if patient_match:
            doc.metadata["record_type"] = "patient"
            doc.metadata["patient_id"] = patient_match.group(1).upper()
        else:
            doc.metadata["record_type"] = "knowledge"
            doc.metadata["patient_id"] = "NONE"

        topic_match = re.search(r"Medical Topic:\s*(.*)", doc.page_content, re.IGNORECASE)
        doc.metadata["topic"] = topic_match.group(1).strip().lower() if topic_match else ""

        all_documents.append(doc)

    return all_documents


def create_vectorstore():
    documents = load_all_sources()
    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.CHUNK_SEPARATORS,
        length_function=len
    )

    chunks = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={
            "batch_size": config.EMBEDDING_BATCH_SIZE,
            "normalize_embeddings": True
        },
        show_progress=True
    )

    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
    save_path = config.VECTORSTORE_PATH / "doctor_patient_index"
    vectorstore.save_local(str(save_path))
    return vectorstore


def get_relevant_docs(question, retriever):
    docs = retriever.invoke(question)

    patient_id = extract_patient_id(question)
    topic = extract_topic_keyword(question)

    if patient_id:
        filtered = [doc for doc in docs if doc.metadata.get("patient_id", "").upper() == patient_id]
        if filtered:
            return filtered

    if topic:
        filtered = []
        for doc in docs:
            filename = doc.metadata.get("filename", "").lower()
            doc_topic = doc.metadata.get("topic", "").lower()
            page = doc.page_content.lower()
            if topic in filename or topic in doc_topic or topic in page:
                filtered.append(doc)
        if filtered:
            return filtered

    return docs


def get_value(label, text):
    match = re.search(label + r":\s*(.*)", text, re.IGNORECASE)
    return match.group(1).strip() if match else "N/A"


def parse_percentage(value):
    try:
        return int(str(value).replace("%", "").strip())
    except Exception:
        return 0


def parse_int(value, default=0):
    try:
        return int(str(value).strip())
    except Exception:
        return default


def dedupe_keep_order(items):
    return list(dict.fromkeys([item for item in items if item and item.strip()]))


def analyze_patient_risk(conditions, missed_int, symptoms, adherence, meds, note, labs):
    conditions_l = conditions.lower()
    symptoms_l = symptoms.lower()
    meds_l = meds.lower()
    note_l = note.lower()
    labs_l = labs.lower()

    good_factors = []
    bad_factors = []
    predicted_risks = []
    long_term_outcomes = []
    doctor_attention = []
    reasoning = []

    adherence_val = parse_percentage(adherence)
    risk_score = 0

    # Good factors
    if adherence_val >= 95:
        good_factors.append("Excellent medication adherence documented")
        risk_score -= 1
    elif adherence_val >= 85:
        good_factors.append("Good medication adherence")
    elif adherence_val >= 80:
        good_factors.append("Fair medication adherence")

    if missed_int == 0:
        good_factors.append("No missed doses reported in last 7 days")
        risk_score -= 1
    elif missed_int == 1:
        bad_factors.append("1 missed dose reported in last 7 days")
        risk_score += 1
    elif missed_int >= 2:
        bad_factors.append(f"{missed_int} missed doses reported in last 7 days")
        risk_score += 2
        predicted_risks.extend([
            "Poor treatment response",
            "Disease progression"
        ])
        long_term_outcomes.extend([
            "Higher hospitalization risk",
            "Chronic disease worsening over time"
        ])
        doctor_attention.append("Improve medication adherence urgently")
        reasoning.append("Repeated missed doses may reduce treatment effectiveness")

    if "stable" in note_l or "controlled" in note_l:
        good_factors.append("Previous note suggests relatively controlled disease state")

    if "normal" in labs_l or "stable" in labs_l:
        good_factors.append("Lab summary appears relatively stable")

    if adherence_val < 80:
        bad_factors.append("Low medication adherence")
        risk_score += 2
        predicted_risks.extend([
            "Suboptimal disease control",
            "Higher complication probability"
        ])
        long_term_outcomes.append("Long-term organ damage risk may increase")
        doctor_attention.append("Review causes of poor adherence")
        reasoning.append("Low adherence is strongly associated with avoidable complications")

    # Symptoms
    if "chest" in symptoms_l:
        bad_factors.append("Chest-related symptoms present")
        risk_score += 3
        predicted_risks.extend([
            "Acute Coronary Syndrome",
            "Myocardial Ischemia"
        ])
        doctor_attention.append("Urgent chest symptom assessment required")
        reasoning.append("Chest symptoms can indicate active cardiac ischemia")

    if "breath" in symptoms_l or "shortness of breath" in symptoms_l:
        bad_factors.append("Breathlessness symptoms present")
        risk_score += 2
        predicted_risks.extend([
            "Cardiopulmonary decompensation",
            "Respiratory distress progression"
        ])
        doctor_attention.append("Evaluate oxygenation and cardiopulmonary status")

    if "dizziness" in symptoms_l:
        bad_factors.append("Dizziness reported")
        risk_score += 1
        predicted_risks.extend([
            "Syncope / Fall Risk",
            "Cerebral hypoperfusion risk"
        ])
        long_term_outcomes.append("Risk of injury may rise if dizziness continues")
        doctor_attention.append("Assess fall risk and circulation")

    if "fatigue" in symptoms_l:
        bad_factors.append("Persistent fatigue reported")
        risk_score += 1

    # Conditions
    if "hypertension" in conditions_l or "blood pressure" in conditions_l:
        bad_factors.append("History of hypertension / BP instability")
        risk_score += 2
        predicted_risks.extend([
            "Hypertensive Crisis",
            "Stroke",
            "Chronic Kidney Disease"
        ])
        long_term_outcomes.extend([
            "Progressive vascular damage",
            "Renal impairment over time"
        ])
        doctor_attention.append("Strict blood pressure monitoring required")
        reasoning.append("Uncontrolled hypertension raises stroke and kidney risk")

    if "diabetes" in conditions_l:
        bad_factors.append("Diabetes increases vascular complication risk")
        risk_score += 2
        predicted_risks.extend([
            "Diabetic Neuropathy",
            "Diabetic Nephropathy",
            "Cardiovascular Disease"
        ])
        long_term_outcomes.extend([
            "Progressive nerve damage",
            "Kidney function decline"
        ])
        doctor_attention.append("Monitor glucose control and vascular risk")
        reasoning.append("Diabetes increases microvascular and macrovascular complications")

    if "heart" in conditions_l or "cardiac" in conditions_l or "coronary" in conditions_l:
        bad_factors.append("Existing cardiovascular disease history")
        risk_score += 3
        predicted_risks.extend([
            "Myocardial Infarction",
            "Heart Failure",
            "Arrhythmia"
        ])
        long_term_outcomes.extend([
            "Progressive cardiac dysfunction",
            "Recurrent cardiac events"
        ])
        doctor_attention.append("Close cardiovascular monitoring required")
        reasoning.append("Pre-existing heart disease significantly raises event risk")

    # Medication / notes clues
    if "aspirin" in meds_l:
        reasoning.append("Cardiovascular medication use suggests cardiac risk background")

    if "high dose" in note_l or "overdose" in note_l or "extra dose" in note_l:
        bad_factors.append("Possible dose misuse / high-dose exposure")
        risk_score += 3
        predicted_risks.extend([
            "Drug Toxicity",
            "Serious Adverse Drug Reaction",
            "Acute Organ Stress"
        ])
        long_term_outcomes.extend([
            "Medication-related organ injury",
            "Recurrent toxicity if misuse continues"
        ])
        doctor_attention.append("Review dosing safety immediately")
        reasoning.append("Dose misuse can trigger toxicity and acute complications")

    if "fluctuating" in labs_l or "bp fluctuating" in labs_l:
        bad_factors.append("Lab summary suggests fluctuating clinical parameters")
        risk_score += 1

    good_factors = dedupe_keep_order(good_factors)
    bad_factors = dedupe_keep_order(bad_factors)
    predicted_risks = dedupe_keep_order(predicted_risks)
    long_term_outcomes = dedupe_keep_order(long_term_outcomes)
    doctor_attention = dedupe_keep_order(doctor_attention)
    reasoning = dedupe_keep_order(reasoning)

    if risk_score >= 7:
        severity = "High Risk"
        risk_color = "#dc2626"
        risk_bg = "#fee2e2"
        accent_class = "high"
    elif risk_score >= 4:
        severity = "Medium Risk"
        risk_color = "#d97706"
        risk_bg = "#fef3c7"
        accent_class = "medium"
    else:
        severity = "Low Risk"
        risk_color = "#16a34a"
        risk_bg = "#dcfce7"
        accent_class = "low"

    if not good_factors:
        good_factors = ["No strong protective factors clearly documented"]
    if not bad_factors:
        bad_factors = ["No major active risk factor documented in current record"]
    if not predicted_risks:
        predicted_risks = ["No major immediate complication predicted from available record"]
    if not long_term_outcomes:
        long_term_outcomes = ["Continue follow-up to reduce long-term complications"]
    if not doctor_attention:
        doctor_attention = ["Routine follow-up and observation recommended"]
    if not reasoning:
        reasoning = ["Prediction based on available diseases, symptoms, adherence, and recent notes"]

    return {
        "severity": severity,
        "risk_color": risk_color,
        "risk_bg": risk_bg,
        "risk_score": risk_score,
        "accent_class": accent_class,
        "good_factors": good_factors[:5],
        "bad_factors": bad_factors[:6],
        "predicted_risks": predicted_risks[:6],
        "long_term_outcomes": long_term_outcomes[:6],
        "doctor_attention": doctor_attention[:5],
        "reasoning": reasoning[:4],
    }


def build_list(items, ordered=False):
    tag = "ol" if ordered else "ul"
    class_name = "focus-list" if ordered else "sources-list"
    inner = "".join(f"<li>{html.escape(item)}</li>" for item in items)
    return f"<{tag} class='{class_name}'>{inner}</{tag}>"


def build_patient_summary(text):
    patient_id = get_value("Patient ID", text)
    patient_name = get_value("Name", text)
    age = get_value("Age", text)
    gender = get_value("Gender", text)
    conditions = get_value("Medical History", text)
    meds = get_value("Current Medicines", text)
    missed = get_value("Missed Doses Last 7 Days", text)
    adherence = get_value("Medication Adherence", text)
    symptoms = get_value("Recent Symptoms", text)
    note = get_value("Previous Doctor Note", text)
    labs = get_value("Lab Summary", text)

    missed_int = parse_int(missed, 0)

    risk_data = analyze_patient_risk(
        conditions=conditions,
        missed_int=missed_int,
        symptoms=symptoms,
        adherence=adherence,
        meds=meds,
        note=note,
        labs=labs
    )

    if risk_data["severity"] == "High Risk":
        main_risk = "Patient shows multiple active clinical risk indicators and needs close medical review."
    elif risk_data["severity"] == "Medium Risk":
        main_risk = "Patient has moderate clinical risk indicators that may worsen if the current pattern continues."
    else:
        main_risk = "Current record suggests relatively lower immediate risk, but ongoing monitoring remains important."

    answer_html = f"""
    <div class="result-card fade-card accent-{risk_data['accent_class']}">
        <div class="summary-banner banner-{risk_data['accent_class']}">
            <div class="summary-banner-left">
                <div class="summary-banner-title">Clinical Risk Summary</div>
                <div class="summary-banner-subtitle">Risk-based patient overview for doctor/admin review</div>
            </div>
            <div class="summary-banner-score">Score: {risk_data['risk_score']}</div>
        </div>

        <div class="top-row">
            <div>
                <div class="patient-name">{html.escape(patient_name)} ({html.escape(patient_id)})</div>
                <div class="patient-meta">Age: {html.escape(age)} &nbsp;|&nbsp; Gender: {html.escape(gender)}</div>
            </div>
            <div class="risk-pill" style="background:{risk_data['risk_bg']}; color:{risk_data['risk_color']}; border:1px solid {risk_data['risk_color']};">
                {risk_data['severity']}
            </div>
        </div>

        <div class="info-box">
            <div class="section-title">Main Risk</div>
            <div class="section-text">{html.escape(main_risk)}</div>
        </div>

        <div class="info-box">
            <div class="section-title">Clinical Summary</div>
            <div class="section-text"><b>Medical History:</b> {html.escape(conditions)}</div>
            <div class="section-text"><b>Current Medicines:</b> {html.escape(meds)}</div>
            <div class="section-text"><b>Missed Doses:</b> {html.escape(missed)} &nbsp;|&nbsp; <b>Adherence:</b> {html.escape(adherence)}</div>
            <div class="section-text"><b>Recent Symptoms:</b> {html.escape(symptoms)}</div>
        </div>

        <div class="two-col">
            <div class="info-box good-box">
                <div class="section-title green">Good Factors</div>
                {build_list(risk_data["good_factors"])}
            </div>
            <div class="info-box bad-box">
                <div class="section-title red">Risk Factors</div>
                {build_list(risk_data["bad_factors"])}
            </div>
        </div>

        <div class="two-col">
            <div class="info-box">
                <div class="section-title">Doctor Focus</div>
                {build_list(risk_data["doctor_attention"], ordered=True)}
            </div>
            <div class="info-box">
                <div class="section-title">Clinical Notes</div>
                <div class="section-text"><b>Previous Note:</b> {html.escape(note)}</div>
                <div class="section-text"><b>Lab Summary:</b> {html.escape(labs)}</div>
            </div>
        </div>

        <div class="two-col">
            <div class="info-box risk-alert-box">
                <div class="section-title red">Predicted Clinical Risks</div>
                {build_list(risk_data["predicted_risks"])}
            </div>
            <div class="info-box">
                <div class="section-title">Possible Long-Term Outcomes</div>
                {build_list(risk_data["long_term_outcomes"])}
            </div>
        </div>

        <div class="info-box">
            <div class="section-title">Why This Risk Was Assigned</div>
            {build_list(risk_data["reasoning"])}
        </div>
    </div>
    """
    return answer_html


def build_topic_summary(text):
    topic = get_value("Medical Topic", text)
    overview = get_value("Overview", text)
    symptoms = get_value("Common Symptoms", text)
    what_to_do = get_value("What To Do", text)
    what_not_to_do = get_value("What Not To Do", text)
    precautions = get_value("Precautions", text)
    doctor_focus = get_value("Doctor Focus", text)
    quick_summary = get_value("Quick Summary", text)

    answer_html = f"""
    <div class="result-card fade-card accent-low">
        <div class="summary-banner banner-low">
            <div class="summary-banner-left">
                <div class="summary-banner-title">General Health Knowledge</div>
                <div class="summary-banner-subtitle">Quick structured knowledge summary</div>
            </div>
            <div class="summary-banner-score">Guide</div>
        </div>

        <div class="top-row">
            <div>
                <div class="patient-name">{html.escape(topic.title())}</div>
                <div class="patient-meta">General Health Knowledge Summary</div>
            </div>
            <div class="risk-pill" style="background:#e0f2fe; color:#0284c7; border:1px solid #38bdf8;">
                Knowledge Base
            </div>
        </div>

        <div class="info-box">
            <div class="section-title">Overview</div>
            <div class="section-text">{html.escape(overview)}</div>
        </div>

        <div class="info-box">
            <div class="section-title">Common Symptoms</div>
            <div class="section-text">{html.escape(symptoms)}</div>
        </div>

        <div class="two-col">
            <div class="info-box good-box">
                <div class="section-title green">What To Do</div>
                <div class="section-text">{html.escape(what_to_do)}</div>
            </div>
            <div class="info-box bad-box">
                <div class="section-title red">What Not To Do</div>
                <div class="section-text">{html.escape(what_not_to_do)}</div>
            </div>
        </div>

        <div class="two-col">
            <div class="info-box">
                <div class="section-title">Precautions</div>
                <div class="section-text">{html.escape(precautions)}</div>
            </div>
            <div class="info-box">
                <div class="section-title">Doctor Focus</div>
                <div class="section-text">{html.escape(doctor_focus)}</div>
            </div>
        </div>

        <div class="info-box">
            <div class="section-title">Quick Summary</div>
            <div class="section-text">{html.escape(quick_summary)}</div>
        </div>
    </div>
    """
    return answer_html


def wrap_modal(content_html):
    return f"""
    <div id="summary-modal" class="modal-overlay" onclick="if(event.target===this) this.style.display='none'">
        <div class="modal-card">
            <div class="modal-header">
                <div class="modal-title">Patient Summary Popup</div>
                <button class="modal-close" onclick="document.getElementById('summary-modal').style.display='none'">×</button>
            </div>
            <div class="modal-body">
                {content_html}
            </div>
        </div>
    </div>
    """


print("🚀 Building vectorstore...")
vectorstore = create_vectorstore()
if vectorstore is None:
    raise ValueError("No files found inside data folder.")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": config.TOP_K}
)


def generate_summary(question):
    if not question.strip():
        empty_html = """
        <div class="result-card fade-card accent-low">
            <div class="patient-name">No Query Entered</div>
            <div class="section-text">Please type a doctor or health query first.</div>
        </div>
        """
        return empty_html, "", wrap_modal(empty_html)

    docs = get_relevant_docs(question, retriever)
    if not docs:
        error_html = """
        <div class="result-card fade-card accent-medium">
            <div class="patient-name">No Data Found</div>
            <div class="section-text">No matching patient or topic record was found for this query.</div>
        </div>
        """
        return error_html, "", wrap_modal(error_html)

    text = docs[0].page_content
    record_type = docs[0].metadata.get("record_type", "")

    if record_type == "patient":
        answer_html = build_patient_summary(text)
    else:
        answer_html = build_topic_summary(text)

    sources_html = f"""
    <div class="sources-card fade-card">
        <div class="section-title" style="margin-top:0;">Sources</div>
        <ul class="sources-list">
            <li>{html.escape(docs[0].metadata.get('filename', 'Unknown'))}</li>
        </ul>
    </div>
    """

    modal_html = wrap_modal(answer_html)
    return answer_html, sources_html, modal_html


CUSTOM_CSS = """
:root {
    --bg-1: #eef7ff;
    --bg-2: #dff1ff;
    --text-main: #0f172a;
    --text-soft: #475569;
    --text-light: #e2e8f0;
    --card: rgba(255, 255, 255, 0.95);
    --border: rgba(148, 163, 184, 0.28);
    --shadow: 0 18px 45px rgba(15, 23, 42, 0.10);
    --blue: #2563eb;
    --sky: #0ea5e9;
    --green: #16a34a;
    --red: #dc2626;
    --amber: #d97706;
}

html, body {
    background:
        radial-gradient(circle at top left, #ffffff 0%, #eef7ff 35%, #dff1ff 100%) !important;
    color: var(--text-main) !important;
}

body, .gradio-container, .gr-block, .gr-box, .gr-form, .gr-group, .gr-panel {
    color: var(--text-main) !important;
}

.gradio-container {
    max-width: 1220px !important;
    margin: auto !important;
    padding: 22px 10px 28px 10px !important;
    background: transparent !important;
}

h1, h2, h3, h4, p, label, li, span, div, strong, b {
    color: var(--text-main) !important;
}

.gr-html, .gr-html *, .gr-markdown, .gr-markdown * {
    color: var(--text-main) !important;
}

.gr-box, .gr-form, .gr-group {
    background: transparent !important;
    border: none !important;
}

textarea, input, .gr-textbox textarea, .gr-textbox input {
    background: rgba(255, 255, 255, 0.98) !important;
    color: #0f172a !important;
    border: 1.2px solid rgba(125, 211, 252, 0.7) !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 20px rgba(37, 99, 235, 0.06);
    font-size: 15px !important;
}

textarea::placeholder, input::placeholder {
    color: #64748b !important;
    opacity: 1 !important;
}

.gr-textbox label, .gr-textbox span, .gr-textbox p {
    color: #dbeafe !important;
    font-weight: 700 !important;
}

.gr-button-primary, button.primary {
    background: linear-gradient(135deg, #4f46e5 0%, #2563eb 55%, #0ea5e9 100%) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 14px 28px rgba(37, 99, 235, 0.24) !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
}

.gr-button-secondary, button.secondary {
    background: linear-gradient(135deg, #475569 0%, #334155 100%) !important;
    border: none !important;
    color: white !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
}

.hero-box {
    background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(248,252,255,0.94));
    border: 1px solid rgba(148, 163, 184, 0.22);
    border-radius: 28px;
    padding: 30px;
    box-shadow: 0 20px 50px rgba(2, 132, 199, 0.10);
    margin-bottom: 18px;
    backdrop-filter: blur(10px);
}

.hero-title {
    margin-top: 0;
    margin-bottom: 8px;
    font-size: 40px;
    font-weight: 900;
    color: #0f172a !important;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    font-size: 18px;
    color: #1e293b !important;
    margin-bottom: 20px;
    font-weight: 600;
    line-height: 1.75;
    opacity: 1 !important;
}

.hero-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 18px;
}

.hero-mini-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(191, 219, 254, 0.9);
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 10px 24px rgba(14, 165, 233, 0.07);
}

.example-title {
    margin-bottom: 8px;
    color: #0f172a !important;
    font-size: 20px;
    font-weight: 800;
}

.example-list, .example-list li {
    color: #334155 !important;
    line-height: 1.85;
    margin-top: 0;
}

.query-wrap {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(96, 165, 250, 0.18);
    border-radius: 24px;
    padding: 16px;
    margin-bottom: 10px;
    box-shadow: 0 18px 35px rgba(15, 23, 42, 0.18);
}

.result-card,
.sources-card {
    background: rgba(255,255,255,0.95);
    border: 1px solid rgba(203, 213, 225, 0.8);
    border-radius: 26px;
    padding: 24px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(8px);
    color: #0f172a !important;
}

.top-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 16px;
}

.patient-name {
    font-size: 30px;
    font-weight: 900;
    color: #0f172a !important;
    margin-bottom: 6px;
    letter-spacing: -0.02em;
}

.patient-meta {
    color: #475569 !important;
    font-size: 15px;
    font-weight: 600;
}

.risk-pill {
    padding: 11px 18px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 14px;
    white-space: nowrap;
    box-shadow: 0 8px 18px rgba(15, 23, 42, 0.08);
}

.summary-banner {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;
    padding: 15px 18px;
    border-radius: 18px;
    margin-bottom: 18px;
    flex-wrap: wrap;
}

.banner-high {
    background: linear-gradient(135deg, #fff1f2, #ffe4e6);
    border: 1px solid rgba(248, 113, 113, 0.4);
}

.banner-medium {
    background: linear-gradient(135deg, #fff7ed, #fef3c7);
    border: 1px solid rgba(245, 158, 11, 0.35);
}

.banner-low {
    background: linear-gradient(135deg, #eff6ff, #e0f2fe);
    border: 1px solid rgba(56, 189, 248, 0.35);
}

.summary-banner-title {
    font-size: 18px;
    font-weight: 900;
}

.summary-banner-subtitle {
    font-size: 13px;
    color: #475569 !important;
    margin-top: 3px;
    font-weight: 600;
}

.summary-banner-score {
    background: rgba(255,255,255,0.9);
    border: 1px solid rgba(203, 213, 225, 0.7);
    padding: 8px 14px;
    border-radius: 999px;
    font-weight: 800;
    box-shadow: 0 6px 14px rgba(15, 23, 42, 0.05);
}

.info-box {
    background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,250,252,0.92));
    border: 1.6px solid rgba(56, 189, 248, 0.35);
    border-radius: 20px;
    padding: 18px;
    margin-top: 14px;
    box-shadow: 0 10px 24px rgba(14, 165, 233, 0.08);
}

.good-box {
    border-color: rgba(74, 222, 128, 0.45) !important;
    box-shadow: 0 10px 24px rgba(34, 197, 94, 0.09);
}

.bad-box, .risk-alert-box {
    border-color: rgba(248, 113, 113, 0.42) !important;
    box-shadow: 0 10px 24px rgba(239, 68, 68, 0.10);
}

.section-title {
    font-size: 18px;
    font-weight: 900;
    color: #0284c7 !important;
    margin-bottom: 10px;
    letter-spacing: -0.01em;
}

.section-title.green {
    color: #16a34a !important;
}

.section-title.red {
    color: #dc2626 !important;
}

.section-text {
    color: #0f172a !important;
    line-height: 1.8;
    margin-bottom: 8px;
    font-weight: 500;
}

.two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin-top: 14px;
}

.focus-list, .sources-list {
    margin: 0;
    padding-left: 20px;
    line-height: 1.95;
    color: #0f172a !important;
    font-weight: 500;
}

.focus-list li, .sources-list li {
    color: #0f172a !important;
    margin-bottom: 2px;
}

.action-row {
    margin-top: 10px;
    margin-bottom: 12px;
}

.fade-card {
    animation: fadeIn 0.35s ease-in-out;
}

.accent-high {
    border-top: 5px solid #dc2626;
}

.accent-medium {
    border-top: 5px solid #d97706;
}

.accent-low {
    border-top: 5px solid #0ea5e9;
}

.modal-overlay {
    position: fixed;
    inset: 0;
    background: rgba(2, 6, 23, 0.56);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    padding: 24px;
    backdrop-filter: blur(7px);
    animation: modalFade 0.25s ease;
}

.modal-card {
    width: min(1100px, 96vw);
    max-height: 90vh;
    overflow-y: auto;
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.96));
    border: 1px solid rgba(203, 213, 225, 0.9);
    border-radius: 28px;
    padding: 0;
    box-shadow: 0 30px 80px rgba(15, 23, 42, 0.28);
}

.modal-header {
    position: sticky;
    top: 0;
    z-index: 3;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 22px;
    background: rgba(255,255,255,0.95);
    border-bottom: 1px solid rgba(226, 232, 240, 0.9);
    backdrop-filter: blur(10px);
    border-top-left-radius: 28px;
    border-top-right-radius: 28px;
}

.modal-title {
    font-size: 22px;
    font-weight: 900;
    color: #0f172a !important;
}

.modal-close {
    width: 42px;
    height: 42px;
    border: none;
    border-radius: 999px;
    background: linear-gradient(135deg, #ef4444, #dc2626);
    color: white;
    font-size: 24px;
    font-weight: 800;
    cursor: pointer;
    box-shadow: 0 10px 20px rgba(220, 38, 38, 0.22);
}

.modal-body {
    padding: 20px;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes modalFade {
    from { opacity: 0; }
    to { opacity: 1; }
}

@media (max-width: 900px) {
    .hero-grid,
    .two-col {
        grid-template-columns: 1fr;
    }
}

@media (max-width: 768px) {
    .patient-name { font-size: 22px; }
    .hero-title { font-size: 29px; }
    .hero-subtitle { font-size: 16px; }
    .modal-card { width: 100%; max-height: 92vh; }
}
"""


with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="sky",
        neutral_hue="slate"
    ),
    css=CUSTOM_CSS,
    title="Doctor/Admin Patient Summary Assistant"
) as demo:
    gr.HTML("""
    <div class="hero-box">
        <div class="hero-title">🩺 Doctor/Admin Patient Summary Assistant</div>
        <div class="hero-subtitle">
            Ask <b>patient-specific</b> or <b>general health</b> queries and get a clean,
            professional summary with <b>risk-based prediction</b>, <b>future outcomes</b>,
            and a <b>popup modal review</b>.
        </div>

        <div class="hero-grid">
            <div class="hero-mini-card">
                <div class="example-title">Example Patient Queries</div>
                <ul class="example-list">
                    <li>Give a short summary of patient P001 for doctor review</li>
                    <li>What are the main risks for patient P003?</li>
                    <li>Predict future complications for patient P003</li>
                    <li>Does patient P001 need medication review?</li>
                </ul>
            </div>

            <div class="hero-mini-card">
                <div class="example-title">Example General Queries</div>
                <ul class="example-list">
                    <li>Give summary about fever precautions</li>
                    <li>What to do in diabetes?</li>
                    <li>What are blood pressure precautions?</li>
                    <li>Give summary for cold and cough care</li>
                </ul>
            </div>
        </div>
    </div>
    """)

    gr.HTML('<div class="query-wrap">')

    query_input = gr.Textbox(
        label="Doctor / Health Query",
        placeholder="Example: Predict future complications for patient P003",
        lines=2
    )

    gr.HTML('</div>')

    with gr.Row(elem_classes=["action-row"]):
        submit_btn = gr.Button("Generate Summary", variant="primary")
        clear_btn = gr.Button("Clear", variant="secondary")

    answer_output = gr.HTML(label="Summary Preview")
    sources_output = gr.HTML(label="Sources")
    modal_output = gr.HTML(label="Popup Modal")

    submit_btn.click(
        fn=generate_summary,
        inputs=[query_input],
        outputs=[answer_output, sources_output, modal_output]
    )

    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[answer_output, sources_output, modal_output]
    )

print("🚀 Launching Gradio Interface...")
demo.launch(
    share=true,
    debug=True,
    show_error=True,
    server_name="127.0.0.1",
    server_port=7860
)
