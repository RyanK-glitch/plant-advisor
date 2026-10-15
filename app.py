import json
import os
from pathlib import Path
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from google import genai
from google.genai import errors, types
from src.input_guard import InputGuard
from src.labels import SUPPORTED_CROPS
from src.rule_engine import AgriculturalRuleEngine
from src.rag_pipeline import RegulatoryRAG
from src import ui_components as ui
from src import weather_risk


st.set_page_config(
    page_title="Plant Disease Safe-RAG Advisory",
    page_icon="🌿",
    layout="centered"
)

ui.load_css()
ui.sidebar()

# Free-tier Gemini text models, tried in order until one responds
GEMINI_MODELS = [
    
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
]

# Report languages: display label -> name given to Gemini
REPORT_LANGUAGES = {
    "English": "English",
    "हिन्दी (Hindi)": "Hindi",
    "தமிழ் (Tamil)": "Tamil",
    "తెలుగు (Telugu)": "Telugu",
    "ಕನ್ನಡ (Kannada)": "Kannada",
    "മലയാളം (Malayalam)": "Malayalam",
    "मराठी (Marathi)": "Marathi",
    "বাংলা (Bengali)": "Bengali",
    "ગુજરાતી (Gujarati)": "Gujarati",
    "ਪੰਜਾਬੀ (Punjabi)": "Punjabi",
    "ଓଡ଼ିଆ (Odia)": "Odia",
}

# Busy, rate-limited, retired or timed-out models: move to the next one
FALLBACK_CODES = {404, 408, 429, 500, 502, 503, 504}


def generate_with_fallback(api_key, prompt):
    # Give each model 60 seconds so a stalled one does not block the fallback
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=60_000))
    failures = []

    for model_name in GEMINI_MODELS:
        try:
            response = client.models.generate_content(model=model_name, contents=prompt)
            if response.text:
                return response.text, model_name, failures
            failures.append(f"{model_name}: empty response")
        except errors.APIError as error:
            # Other client errors (bad key, bad request) affect every model, so stop
            if error.code not in FALLBACK_CODES:
                raise
            failures.append(f"{model_name}: {error.code} {error.status}")
        except Exception as error:
            # Network timeouts and dropped connections
            failures.append(f"{model_name}: {type(error).__name__}")

    raise RuntimeError("All Gemini models are currently unavailable. " + "; ".join(failures))


@st.cache_resource
def load_heavy_models():
    # Resolve best.pt from the project folder, not the terminal's current folder
    app_dir = Path(__file__).resolve().parent
    configured_path = os.environ.get("YOLO_MODEL_PATH")
    model_path = (
        Path(configured_path).expanduser().resolve()
        if configured_path
        else app_dir / "runs/classify/crop_model_runs/yolo11m_staples_480_split70/weights/best.pt"
    )

    # Never silently use the untrained base model
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}. "
            "Copy best.pt there or set YOLO_MODEL_PATH to its full path."
        )

    print(f"✅ Loading trained YOLOv11m model: {model_path}")
    return InputGuard(YOLO(str(model_path))), RegulatoryRAG(), model_path


def get_api_key():
    # st.secrets raises when no secrets.toml exists, so fall back to the environment
    try:
        key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        key = None
    return key or os.environ.get("GEMINI_API_KEY")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def find_places(name):
    return weather_risk.geocode(name)


@st.cache_data(ttl=30 * 60, show_spinner=False)
def get_forecast(latitude, longitude):
    return weather_risk.fetch_forecast(latitude, longitude)


try:
    guard, rag, loaded_model_path = load_heavy_models()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

rule_engine = AgriculturalRuleEngine()

ui.hero(f"YOLO11m ({loaded_model_path.name})")
ui.steps()

ui.section_title("🌐 Report language")
language_label = st.selectbox(
    "Report language", list(REPORT_LANGUAGES), label_visibility="collapsed"
)
report_language = REPORT_LANGUAGES[language_label]

ui.section_title("🌾 Crop")
crop_options = {"Not sure: detect automatically": None}
crop_options.update({crop.capitalize(): crop for crop in SUPPORTED_CROPS})
selected_crop = crop_options[st.selectbox(
    "Crop", list(crop_options), label_visibility="collapsed",
    help="Choosing your crop limits the diagnosis to that crop's diseases."
)]

ui.section_title("📍 Location for weather risk (optional)")
place = None
place_query = st.text_input(
    "Village, town or district", placeholder="e.g. Ludhiana", label_visibility="collapsed"
).strip()
if len(place_query) == 1:
    st.caption("Type at least two letters of the place name.")
elif place_query:
    try:
        places = find_places(place_query)
    except Exception:
        places = None
        st.caption("Location search is unavailable right now. The advisory works without it.")
    if places == []:
        st.caption("No matching place was found in India. Try the district name.")
    elif places:
        place = places[st.selectbox(
            "Matching places", range(len(places)), format_func=lambda i: places[i]["label"],
            label_visibility="collapsed"
        )]

ui.section_title("📷 Upload a leaf image")
uploaded_file = st.file_uploader(
    "Choose a leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
)

if uploaded_file is None:
    ui.photo_tips()
else:
    image = Image.open(uploaded_file).convert("RGB")
    image_col, action_col = st.columns([3, 2], gap="large")
    with image_col:
        st.image(image, caption="Uploaded leaf specimen", width="stretch")
    with action_col:
        ui.photo_tips()
        st.write("")
        run_analysis = st.button("🔍 Run Safe-RAG Analysis", type="primary")

    if run_analysis:
        with st.spinner("Analyzing the image and retrieving verified evidence..."):
            # Classify the image, blocking advice when it does not match the selected crop
            check = guard.analyse(image, crop=selected_crop)
            ui.section_title("🩺 Diagnosis")

            if check["status"] == guard.CROP_MISMATCH:
                ui.crop_mismatch_card(
                    selected_crop, check["likely_crop"], check["crop_mass"][selected_crop]
                )
                st.stop()

            predicted_class = check["label"]
            confidence = check["confidence"]
            crop, disease_key = check["crop"], check["disease_key"]
            ui.diagnosis_card(crop, disease_key, predicted_class, confidence)

            # Retrieve exact regulatory records and supporting disease evidence
            query = (
                f"{crop} {disease_key.replace('_', ' ')} symptoms management "
                "verified chemical dose water timing PHI re-entry PPQS CIBRC"
            )
            rag_result = rag.query_knowledge_base(
                query, crop_key=crop, disease_key=disease_key, k=4
            )
            context = rag_result["context"]

            with st.expander("📚 View retrieved RAG evidence"):
                st.info(context or "No matching RAG evidence was retrieved.")

            # Apply deterministic regulatory safety rules
            rule = rule_engine.evaluate(crop, disease_key)
            ui.section_title("🛡️ Regulatory safety check")
            ui.rule_card(rule, rule_engine.APPROVED, rule_engine.WARNING)

            # Indicative weather risk: affects timing advice only, never the chemical or dose
            weather_summary = ""
            if place:
                try:
                    hours = get_forecast(place["latitude"], place["longitude"])
                    if disease_key.endswith("_healthy"):
                        # Healthy leaf: show the outlook for every disease of this crop
                        risk_keys = [key for key in rule_engine.rules
                                     if key.startswith(f"{crop}_") and not key.endswith("_healthy")]
                    else:
                        risk_keys = [disease_key]
                    risks = [r for r in (weather_risk.disease_risk(k, hours) for k in risk_keys) if r]

                    sprayable = (rule.get("engine_status") in (rule_engine.APPROVED, rule_engine.WARNING)
                                 and rule.get("chemical") not in (None, "", "NONE"))
                    spray = weather_risk.spray_window(hours) if sprayable else None

                    ui.section_title("🌦️ Weather risk")
                    ui.weather_card(place["label"], risks, spray)
                    weather_summary = weather_risk.summary_for_prompt(place["label"], risks, spray)
                except Exception:
                    st.info("The weather forecast is unavailable right now, "
                            "so the advisory below does not use it.")

            # Generate only when evidence was retrieved
            if not context:
                st.warning("A grounded report cannot be generated without matching RAG evidence.")
                st.stop()

            try:
                api_key = get_api_key()
                if not api_key:
                    st.error("⚠️ GEMINI_API_KEY was not found.")
                    st.stop()

                # RAG provides the facts; Gemini explains them as practical farmer guidance
                prompt = f"""
You are a patient agricultural instructor speaking directly to a farmer.
Explain clearly what the farmer should do about {predicted_class}
({confidence:.2%} image-model confidence). Be helpful, calm and respectful. Make sure to sounds like a real person, not a robot. Use the following RAG evidence and safety rule to ground your advice.

PRIMARY RAG EVIDENCE:
{context}

SAFETY RULE:
{json.dumps(rule, indent=2)}

WEATHER OUTLOOK (indicative forecast, not a validated disease forecast):
{weather_summary or "Not provided."}

Important instructions:
1. Ground every disease, symptom and management claim in the RAG evidence or safety rule.
   Chemical names, doses, timing, maximum applications, PHI and re-entry periods must come
   ONLY from the SAFETY RULE. Never recommend a chemical or dose that is not in the safety rule.
2. Focus on actions, not a list of facts. Say what to do now, what to do next and why each
   action helps. Use complete, natural sentences rather than short technical fragments.
3. Use simple everyday language. If a technical term is necessary, explain it immediately.
4. Do not invent missing advice, pesticide details, measurements or regulatory claims.
   Say when verified information is unavailable and advise contacting a local extension officer.
5. If the status starts with VERIFIED, explain the exact chemical, dose, timing, maximum
   applications, PHI, re-entry period and source URL without changing any value. Explain PHI
   and re-entry in plain language. Never say treatment is unverified when a verified rule exists.
6. For PARTIAL status, give no automatic dose. For a healthy class, recommend no chemical.
7. Remind the farmer that an image result is an initial assessment and should be confirmed
   in the field, especially before applying a chemical.
8. If a weather outlook is provided, use it only to advise on when to spray and how closely
   to watch the crop, and say that it is an indicative forecast. It must never change the
   chemical, dose or any other value from the safety rule.

Use these headings in this practical order:
Assessment; What You Should Do Now; How to Manage and Prevent Further Spread;
Verified Treatment; Safety and Field Confirmation; Sources.

LANGUAGE:
Write the entire report, including the headings, in {report_language}, using simple
everyday words a farmer would use. Keep chemical and product names, doses, units, numbers,
days, hours and source URLs exactly as written in the safety rule, in English letters and
digits, so they match the product label. You may add a descriptive {report_language} explanation
next to them.
"""
                with st.spinner("Generating the advisory report..."):
                    report_text, used_model, failures = generate_with_fallback(api_key, prompt)

                ui.section_title("🤖 RAG-grounded advisory report")
                with st.container(key="advisory_report"):
                    st.markdown(report_text)

                st.caption(f"Report generated by `{used_model}`")
                if failures:
                    with st.expander("Models skipped because they were unavailable"):
                        st.write("\n".join(f"- {failure}" for failure in failures))

            except Exception as error:
                st.error(f"⚠️ Error connecting to Gemini API: {error}")

ui.footer()
