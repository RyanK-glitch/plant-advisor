from html import escape
from pathlib import Path

import streamlit as st

CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "style.css"


def _render(markup):
    # Join lines without indentation so Markdown never treats HTML as a code block
    st.markdown(" ".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def load_css():
    st.markdown(f"<style>{CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


def pretty_label(key):
    # Turn "potato_late_blight" into "Potato Late Blight"
    return " ".join(word.capitalize() for word in key.split("_") if word)


def hero(model_name):
    _render(f"""
    <div class="hero">
        <p class="hero-eyebrow">ICAR / CIBRC compliance-aware advisory</p>
        <h1>🌿 Plant Disease Safe-RAG Advisory</h1>
        <p>Upload a leaf photo to get an AI-assisted diagnosis, a regulatory safety check
        and treatment guidance grounded in verified sources.</p>
        <div class="hero-chips">
            <span class="hero-chip">Model: {escape(model_name)}</span>
            <span class="hero-chip">Maize · Potato · Rice · Wheat</span>
            <span class="hero-chip">PPQS / CIBRC verified rules</span>
        </div>
    </div>
    """)


def steps():
    items = [
        ("Upload", "A clear photo of a single affected leaf."),
        ("Analyse", "YOLO11m classifies the disease and the rule engine checks safety."),
        ("Act", "Read the grounded advisory and confirm in the field."),
    ]
    cells = "".join(
        f'<div class="step"><div class="step-num">{i}</div><div>'
        f'<p class="step-title">{title}</p><p class="step-text">{text}</p></div></div>'
        for i, (title, text) in enumerate(items, start=1)
    )
    _render(f'<div class="steps">{cells}</div>')


def section_title(text):
    _render(f'<div class="section-title">{text}</div>')


def photo_tips():
    _render("""
    <div class="hint-card">
        <p class="card-label">Tips for an accurate result</p>
        <ul>
            <li>Photograph one leaf, filling most of the frame.</li>
            <li>Use daylight and avoid harsh shadows or glare.</li>
            <li>Keep the lesions in sharp focus.</li>
            <li>Plain backgrounds work best.</li>
        </ul>
    </div>
    """)


def diagnosis_card(crop, disease_key, raw_label, confidence):
    percent = confidence * 100
    if percent >= 80:
        level, note = "high", "High confidence. Still confirm symptoms in the field before spraying."
    elif percent >= 50:
        level, note = "medium", "Moderate confidence. Compare with nearby plants or retake the photo."
    else:
        level, note = "low", "Low confidence. Retake a clearer photo or consult an extension officer."

    condition = disease_key.removeprefix(f"{crop}_")
    _render(f"""
    <div class="card diagnosis">
        <div class="diagnosis-main">
            <p class="card-label">Diagnosis</p>
            <p class="card-value">{escape(pretty_label(condition))}</p>
            <p class="card-sub">Crop: <strong>{escape(crop.capitalize())}</strong>
            <code>{escape(raw_label)}</code></p>
            <p class="confidence-note {level}">{note}</p>
        </div>
        <div class="ring {level}" style="--pct:{percent:.1f}" role="img"
             aria-label="Model confidence {percent:.1f} percent">
            <div class="ring-inner"><strong>{percent:.1f}%</strong><span>confidence</span></div>
        </div>
    </div>
    """)


def rule_card(rule, approved, warning):
    status = rule["engine_status"]

    if status == approved:
        css, badge = "approved", "✓ APPROVED"
    elif status == warning:
        css, badge = "warning", "! MANUAL CHECK"
    else:
        css, badge = "rejected", "✕ NO VERIFIED RULE"

    if status not in (approved, warning):
        _render(f"""
        <div class="card status-{css}">
            <div class="rule-head"><p class="rule-chemical">No treatment can be recommended</p>
            <span class="badge {css}">{badge}</span></div>
            <p class="card-sub">{escape(rule.get("message", ""))}</p>
        </div>
        """)
        return

    facts = [
        ("Dose", rule.get("dose")),
        ("Timing", rule.get("timing")),
        ("Max applications", rule.get("max_applications")),
        ("Pre-harvest interval", rule.get("phi")),
        ("Re-entry period", rule.get("re_entry")),
    ]
    fact_html = "".join(
        f'<div class="fact"><p class="fact-label">{label}</p>'
        f'<p class="fact-value">{escape(str(value))}</p></div>'
        for label, value in facts if value
    )

    source = rule.get("source_url", "")
    source_html = (
        f'<p class="rule-source">Source: <a href="{escape(source)}" target="_blank" '
        f'rel="noopener">{escape(source)}</a></p>' if source else ""
    )

    chemical = rule.get("chemical", "")
    title = "No chemical advised" if chemical == "NONE" else chemical

    _render(f"""
    <div class="card status-{css}">
        <div class="rule-head">
            <p class="rule-chemical">{escape(title)}</p>
            <span class="badge {css}">{badge}</span>
        </div>
        <p class="card-sub">Regulatory status: <strong>{escape(rule.get("regulatory_status", ""))}</strong></p>
        <div class="facts">{fact_html}</div>
        <p class="rule-note">{escape(rule.get("ipm_note", ""))}</p>
        {source_html}
    </div>
    """)


def unknown_photo_card():
    _render("""
    <div class="card status-rejected">
        <div class="rule-head">
            <p class="rule-chemical">This photo does not look like a supported leaf</p>
            <span class="badge rejected">✕ NOT RECOGNISED</span>
        </div>
        <p class="card-sub">The image is too different from the maize, potato, rice and wheat leaves
        the model was trained on, so no diagnosis or treatment is given. It may be another crop,
        a pest or nutrient problem, or an unclear photo.</p>
        <p class="rule-note">Retake the photo following the tips above, or show the plant to your
        local agricultural extension officer.</p>
    </div>
    """)


def crop_mismatch_card(selected_crop, likely_crop, crop_mass):
    _render(f"""
    <div class="card status-warning">
        <div class="rule-head">
            <p class="rule-chemical">This leaf looks more like {escape(likely_crop.capitalize())}
            than {escape(selected_crop.capitalize())}</p>
            <span class="badge warning">! CHECK CROP</span>
        </div>
        <p class="card-sub">The model gives {escape(selected_crop.capitalize())} only
        {crop_mass * 100:.0f}% likelihood, so no treatment is given.</p>
        <p class="rule-note">Check the crop you selected, or retake the photo with a single leaf
        filling the frame.</p>
    </div>
    """)


RISK_BADGES = {"High": "rejected", "Moderate": "warning", "Low": "approved"}


def weather_card(place, risks, spray):
    rows = ""
    for risk in risks:
        days = "".join(
            f'<span class="risk-day">{date:%a %d}{": " + escape(value) if value else ""}</span>'
            for date, value in risk["days"]
        )
        rows += f"""
        <div class="risk-row">
            <div class="rule-head">
                <p class="fact-value">{escape(pretty_label(risk["disease_key"]))}</p>
                <span class="badge {RISK_BADGES[risk["level"]]}">{risk["level"].upper()} RISK</span>
            </div>
            <p class="card-sub">{escape(risk["reason"])}</p>
            <div class="risk-days">{days}</div>
        </div>
        """

    spray_html = ""
    if spray:
        if spray["window"]:
            start, end = spray["window"]
            window = f"{start:%a %d %b, %H:%M}–{end:%H:%M}"
        else:
            window = "None in the next 48 hours. Wait and check again."
        notes = []
        if spray["rain_next_24h"]:
            notes.append("Rain is likely in the next 24 hours.")
        if spray["max_wind_next_24h"] >= 15:
            notes.append(f"Wind up to {spray['max_wind_next_24h']:.0f} km/h.")
        spray_html = f"""
        <div class="facts">
            <div class="fact"><p class="fact-label">Next spray window</p>
            <p class="fact-value">{escape(window)}</p></div>
        </div>
        <p class="card-sub">Daytime, wind under 15 km/h and no rain for 6 hours after spraying.
        {escape(" ".join(notes))}</p>
        """

    _render(f"""
    <div class="card">
        <p class="card-label">Weather outlook · next 4 days</p>
        <p class="card-sub">📍 {escape(place)} · Forecast by Open-Meteo</p>
        {rows}
        {spray_html}
        <p class="rule-note">Indicative only: based on forecast temperature and humidity, not a
        validated disease forecast. Weather never changes the verified chemical or dose.</p>
    </div>
    """)


def sidebar():
    with st.sidebar:
        _render("""
        <div class="side-card">
            <h4>About</h4>
            <strong>Developed by Ryan K</strong><br>
            Final-year Computer Science dissertation<br>
            <em>ICAR / CIBRC compliance standards</em>
        </div>
        <div class="side-card">
            <h4>How it works</h4>
            <ol>
                <li>YOLO11m classifies the leaf image and rejects photos it does not recognise.</li>
                <li>RAG retrieves verified PPQS / CIBRC evidence.</li>
                <li>A deterministic rule engine approves or blocks chemical advice.</li>
                <li>Gemini explains the evidence in plain language.</li>
            </ol>
        </div>
        <div class="side-card">
            <h4>Supported crops</h4>
            Maize, Potato, Rice, Wheat
        </div>
        """)


def footer():
    _render("""
    <div class="footer">
        Image results are an initial assessment. Always confirm in the field and follow the
        registered product label before applying any chemical.
    </div>
    """)
