import os
from pathlib import Path
import streamlit as st
from PIL import Image
from ultralytics import YOLO
from src import feedback
from src import read_aloud
from src.advisory_prompt import build_report_prompt
from src.gemini_client import generate_with_fallback
from src.input_guard import InputGuard
from src.labels import SUPPORTED_CROPS
from src.rule_engine import AgriculturalRuleEngine
from src.rag_pipeline import RegulatoryRAG
from src import ui_components as ui
from src import weather_risk


st.set_page_config(
    page_title="LeafDoc · A doctor for your crops, in your pocket",
    page_icon="🌿",
    layout="centered"
)

ui.load_css()
ui.load_motion()
with st.container(key="hud_layer"):
    ui.hud_background()
ui.sidebar()

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

@st.cache_resource
def load_heavy_models():
    # Resolve best.pt from the project folder, not the terminal's current folder
    app_dir = Path(__file__).resolve().parent
    configured_path = os.environ.get("YOLO_MODEL_PATH")
    model_path = (
        Path(configured_path).expanduser().resolve()
        if configured_path
        else app_dir / "runs/classify/crop_model_runs/yolo11m_field_ft/weights/best.pt"
    )

    # Never silently use the untrained base model
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at: {model_path}. "
            "Copy best.pt there or set YOLO_MODEL_PATH to its full path."
        )

    print(f"✅ Loading trained YOLOv11m model: {model_path}")
    return InputGuard(YOLO(str(model_path)), model_path=model_path), RegulatoryRAG(), model_path


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


@st.cache_data(max_entries=32, show_spinner=False)
def report_audio(text, language):
    return read_aloud.synthesize(read_aloud.speakable_text(text), language)


# Fragments rerun on their own, so pressing these buttons keeps the report on screen
@st.fragment
def read_aloud_section(report_text, language, key):
    if not read_aloud.has_voice(language):
        st.caption(f"🔇 Read-aloud is not available in {language} yet.")
        return
    audio_key = f"audio_{key}"
    if audio_key not in st.session_state:
        if not st.button("🔊 Read the report aloud", key=f"listen_{key}"):
            return
        try:
            with st.spinner(f"Preparing the {language} audio. This takes a few seconds..."):
                st.session_state[audio_key] = report_audio(report_text, language)
        except Exception as error:
            print(f"Read-aloud failed: {type(error).__name__}: {error}")
            st.warning("The audio could not be prepared. Check the internet connection and press "
                       f"the button again. ({type(error).__name__})")
            return

    # Kept in session state so the player stays after other buttons are pressed
    st.caption("🔊 Press ▶ to listen if it does not start by itself.")
    st.audio(st.session_state[audio_key], format="audio/mpeg", autoplay=True)


class ReportSkipped(Exception):
    """The advisory report cannot be written, but the rest of the page still runs."""


FEEDBACK_ANSWERS = {
    "✅ Yes, correct": feedback.CORRECT,
    "❌ No, wrong": feedback.WRONG,
    "🤔 Not sure": feedback.UNSURE,
}
UNKNOWN_DISEASE = "other"


@st.fragment
def feedback_section(prediction, image, image_id, selected_crop, place_label, language, model_name):
    ui.section_title("🙋 Was this diagnosis correct?")
    saved_key = f"feedback_saved_{image_id}"
    if st.session_state.get(saved_key):
        st.success("Thank you. Your answer was saved and will help improve the model.")
        return

    answer = st.radio(
        "Your answer", list(FEEDBACK_ANSWERS), index=None, horizontal=True,
        key=f"answer_{image_id}", label_visibility="collapsed"
    )
    verdict = FEEDBACK_ANSWERS.get(answer)

    correct_key = None
    if verdict == feedback.WRONG:
        # Same-crop diseases first, since a wrong crop is less likely than a wrong disease
        crop = prediction["crop"]
        options = sorted(
            {key for _, key in guard.class_keys} - {prediction["disease_key"]},
            key=lambda key: (not key.startswith(f"{crop}_"), key)
        ) + [UNKNOWN_DISEASE]
        correct_key = st.selectbox(
            "What is it actually?", options, key=f"correct_{image_id}",
            format_func=lambda key: "Another problem / I don't know" if key == UNKNOWN_DISEASE
            else ui.pretty_label(key)
        )

    comment = st.text_input(
        "Anything else? (optional)", max_chars=feedback.MAX_COMMENT_LENGTH,
        key=f"comment_{image_id}", placeholder="e.g. confirmed by the extension officer"
    )
    share_photo = st.checkbox(
        "Share this photo to help improve the model. Location data inside the photo is removed.",
        key=f"share_{image_id}"
    )

    if st.button("Send answer", key=f"send_{image_id}", disabled=verdict is None):
        try:
            feedback.save_feedback(
                verdict=verdict, prediction=prediction, image=image, share_photo=share_photo,
                correct_key=correct_key, comment=comment, selected_crop=selected_crop,
                place=place_label, report_language=language, model_name=model_name,
            )
        except Exception:
            st.error("Your answer could not be saved. Please try again.")
            return
        st.session_state[saved_key] = True
        st.rerun(scope="fragment")


try:
    guard, rag, loaded_model_path = load_heavy_models()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

rule_engine = AgriculturalRuleEngine()

ui.mobile_header()   # phones only: app bar and short intro instead of the hero
ui.hero(f"YOLO11m ({loaded_model_path.name})")
ui.steps()

# Settings in one container: on phones the stylesheet shows them after the upload card
with st.container(key="settings_block"):
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

ui.mobile_scan_header()   # phones only: the "Scan a leaf" card around the uploader
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
        run_analysis = st.button("🔍 Run Safe-RAG Analysis", type="primary", key="run_analysis")

    if run_analysis:
        with st.spinner("Analyzing the image and retrieving verified evidence..."):
            # Classify the image, blocking advice when it does not match the selected crop
            with ui.analysis_scan("Scanning the leaf",
                                  "YOLO11m is classifying the disease and checking the photo.",
                                  step=0, language=report_language):
                check = guard.analyse(image, crop=selected_crop)
            ui.section_title("🩺 Diagnosis")

            if check["status"] == guard.UNKNOWN_INPUT:
                ui.unknown_input_card()
                st.stop()

            if check["status"] == guard.CROP_MISMATCH:
                ui.crop_mismatch_card(
                    selected_crop, check["likely_crop"], check["crop_mass"][selected_crop]
                )
                st.stop()

            if check["status"] == guard.LOW_CONFIDENCE:
                # Calibrated confidence below the abstention threshold: no treatment advice
                ui.low_confidence_card(
                    check["crop"], check["disease_key"], check["confidence"],
                    check["threshold"], check["alternatives"]
                )
                st.stop()

            predicted_class = check["label"]
            confidence = check["confidence"]
            image_id = feedback.image_digest(image)[:16]
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

            try:
                # Generate only when evidence was retrieved
                if not context:
                    raise ReportSkipped("A grounded report cannot be generated without matching RAG evidence.")
                api_key = get_api_key()
                if not api_key:
                    raise ReportSkipped("⚠️ GEMINI_API_KEY was not found.")

                # RAG provides the facts; Gemini explains them as practical farmer guidance
                prompt = build_report_prompt(
                    predicted_class, confidence, context, rule, weather_summary, report_language
                )
                with st.spinner("Generating the advisory report..."), ui.analysis_scan(
                    "Writing your advisory",
                    f"Gemini is explaining the verified evidence and safety rule in {report_language}.",
                    step=2, language=report_language
                ):
                    report_text, used_model, failures = generate_with_fallback(api_key, prompt)

                ui.section_title("🤖 RAG-grounded advisory report")
                with st.container(key="advisory_report"):
                    st.markdown(report_text)

                read_aloud_section(report_text, report_language, key=image_id)
                st.caption(f"Report generated by `{used_model}`")
                if failures:
                    with st.expander("Models skipped because they were unavailable"):
                        st.write("\n".join(f"- {failure}" for failure in failures))

            except ReportSkipped as reason:
                st.warning(str(reason))
            except Exception as error:
                st.error(f"⚠️ Error connecting to Gemini API: {error}")

        # Asked even when the report failed, because the diagnosis is what is being judged
        feedback_section(
            {key: check[key] for key in ("label", "crop", "disease_key", "confidence")},
            image, image_id, selected_crop, place["label"] if place else None,
            report_language, loaded_model_path.parent.parent.name
        )

ui.footer()
