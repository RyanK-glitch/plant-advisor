from contextlib import contextmanager
from html import escape
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent / "assets"
CSS_PATH = ASSETS / "style.css"
DARK_TOKENS_PATH = ASSETS / "dark-tokens.css"
MOTION_PATH = ASSETS / "motion.js"
FONTS_URL = ("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
             + "".join(f"&family=Noto+Sans+{script}:wght@500" for script in (
                 "Devanagari", "Tamil", "Telugu", "Kannada", "Malayalam", "Bengali", "Gujarati", "Gurmukhi", "Oriya"))
             + "&display=swap")
# Still render of the Blender leaf: the fallback when 3D is off (served from static/, see config.toml)
LEAF_POSTER = '<img class="leaf3d-poster" src="app/static/3d/leaf.webp" alt="" width="250" height="250">'


def _render(markup):
    # Join lines without indentation so Markdown never treats HTML as a code block
    st.markdown(" ".join(line.strip() for line in markup.splitlines()), unsafe_allow_html=True)


def load_css():
    # Dark tokens apply when motion.js marks the page html.theme-dark (it reads the active Streamlit
    # theme in the browser). Until then, the server's view of the theme is used as a first guess.
    dark = DARK_TOKENS_PATH.read_text()
    guess = f":root:not(.theme-light) {{{dark}}}" if st.context.theme.type == "dark" else ""
    # st.html, not st.markdown: a style-only st.html takes no space and is not parsed as Markdown
    st.html(f"<style>@import url('{FONTS_URL}');\n{CSS_PATH.read_text()}\nhtml.theme-dark {{{dark}}}\n{guess}</style>")


def load_motion():
    # Runs assets/motion.js on the page itself (st.html is not iframed), inside a hidden container
    with st.container(key="motion_script"):
        try:
            st.html(f"<script>{MOTION_PATH.read_text()}</script>", unsafe_allow_javascript=True)
        except TypeError:
            pass   # Streamlit too old for unsafe_allow_javascript: the still leaf image stays


def pretty_label(key):
    # Turn "potato_late_blight" into "Potato Late Blight"
    return " ".join(word.capitalize() for word in key.split("_") if word)


# One leaf per supported crop, drawn with its disease's typical marks; the hero cycles through them.
CROP_LEAVES = [
    ("maize_common_rust · 0.97", """
    <defs><linearGradient id="hero-g-maize" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#3f8a3a"/><stop offset="1" stop-color="#7cc46a"/></linearGradient></defs>
    <g transform="rotate(-38 100 100)">
    <path d="M8 96 C60 78 130 76 194 94 C130 101 62 117 8 126 Z" fill="url(#hero-g-maize)"/>
    <path d="M8 111 C70 97 130 90 194 94" stroke="#d6f0c8" stroke-width="1.6" fill="none" opacity="0.8"/>
    <path d="M14 102 C70 89 130 85 186 93 M14 120 C70 108 130 98 186 96" stroke="#d6f0c8" stroke-width="0.7" fill="none" opacity="0.5"/>
    <g fill="#a4521f"><ellipse cx="40" cy="106" rx="2.6" ry="1.3"/><ellipse cx="52" cy="114" rx="2.2" ry="1.1"/><ellipse cx="66" cy="101" rx="2.6" ry="1.3"/><ellipse cx="78" cy="109" rx="2.4" ry="1.2"/><ellipse cx="92" cy="98" rx="2.2" ry="1.1"/><ellipse cx="104" cy="105" rx="2.6" ry="1.3"/><ellipse cx="118" cy="96" rx="2.2" ry="1.1"/><ellipse cx="130" cy="101" rx="2.4" ry="1.2"/><ellipse cx="146" cy="94" rx="2" ry="1"/><ellipse cx="60" cy="118" rx="2" ry="1"/><ellipse cx="88" cy="112" rx="2" ry="1"/></g>
    <g fill="#d98a3a"><ellipse cx="46" cy="99" rx="1.6" ry="0.9"/><ellipse cx="72" cy="117" rx="1.6" ry="0.9"/><ellipse cx="98" cy="90" rx="1.6" ry="0.9"/><ellipse cx="124" cy="107" rx="1.6" ry="0.9"/><ellipse cx="138" cy="91" rx="1.4" ry="0.8"/></g>
    </g>"""),
    ("potato_early_blight · 0.95", """
    <defs><linearGradient id="hero-g-potato" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#3d8240"/><stop offset="1" stop-color="#6fb85f"/></linearGradient></defs>
    <path d="M36 176 L152 38" stroke="#5c9a4d" stroke-width="3" stroke-linecap="round"/>
    <g fill="url(#hero-g-potato)">
    <ellipse cx="160" cy="30" rx="30" ry="18" transform="rotate(-50 160 30)"/>
    <ellipse cx="132" cy="106" rx="25" ry="14" transform="rotate(40 132 106)"/>
    <ellipse cx="84" cy="68" rx="25" ry="14" transform="rotate(-140 84 68)"/>
    <ellipse cx="96" cy="146" rx="20" ry="12" transform="rotate(40 96 146)"/>
    <ellipse cx="50" cy="110" rx="20" ry="12" transform="rotate(-140 50 110)"/>
    </g>
    <path d="M140 52 L178 10 M116 92 L152 122 M100 80 L64 54 M86 134 L112 160 M68 120 L32 98" stroke="#d6f0c8" stroke-width="1" opacity="0.6"/>
    <circle cx="162" cy="34" r="8" fill="#d9c25a" opacity="0.65"/><circle cx="162" cy="34" r="5" fill="#5b3a1e"/><circle cx="162" cy="34" r="3" fill="none" stroke="#a37446" stroke-width="0.8"/>
    <circle cx="138" cy="108" r="7" fill="#d9c25a" opacity="0.65"/><circle cx="138" cy="108" r="4.4" fill="#5b3a1e"/><circle cx="138" cy="108" r="2.6" fill="none" stroke="#a37446" stroke-width="0.8"/>
    <circle cx="78" cy="64" r="6.5" fill="#d9c25a" opacity="0.65"/><circle cx="78" cy="64" r="4" fill="#5b3a1e"/><circle cx="78" cy="64" r="2.3" fill="none" stroke="#a37446" stroke-width="0.8"/>
    <circle cx="148" cy="22" r="5" fill="#d9c25a" opacity="0.65"/><circle cx="148" cy="22" r="3" fill="#5b3a1e"/>"""),
    ("rice_leaf_blast · 0.93", """
    <defs><linearGradient id="hero-g-rice" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#4c9a44"/><stop offset="1" stop-color="#8fd27a"/></linearGradient></defs>
    <path d="M30 190 C70 140 120 80 170 18 C124 82 76 142 38 194 Z" fill="#3f7f39" opacity="0.7"/>
    <path d="M14 182 C76 122 136 62 192 10 C142 64 84 126 26 190 Z" fill="url(#hero-g-rice)"/>
    <path d="M20 186 C80 124 140 62 192 10" stroke="#d6f0c8" stroke-width="1" fill="none" opacity="0.7"/>
    <g stroke="#7a4a22" stroke-width="1.2" fill="#e2dccd">
    <path d="M62 136 L70 126 L74 128 L66 138 Z"/><path d="M98 100 L108 88 L112 90 L102 102 Z"/>
    <path d="M132 66 L140 56 L144 58 L136 68 Z"/><path d="M82 124 L88 116 L91 118 L85 126 Z"/>
    <path d="M156 44 L162 36 L165 38 L159 46 Z"/>
    </g>"""),
    ("wheat_yellow_rust · 0.97", """
    <defs><linearGradient id="hero-g-wheat" x1="0" y1="1" x2="1" y2="0"><stop offset="0" stop-color="#4a8a40"/><stop offset="1" stop-color="#86c46e"/></linearGradient>
    <clipPath id="hero-clip-wheat"><path d="M10 176 C70 120 132 66 194 18 C140 72 82 128 24 192 Z"/></clipPath></defs>
    <path d="M10 176 C70 120 132 66 194 18 C140 72 82 128 24 192 Z" fill="url(#hero-g-wheat)"/>
    <g clip-path="url(#hero-clip-wheat)" fill="none" stroke-linecap="round">
    <path d="M14 182 C76 124 136 70 194 20" stroke="#d6f0c8" stroke-width="0.7" opacity="0.5"/>
    <path d="M22 186 C82 128 140 74 194 26" stroke="#d6f0c8" stroke-width="0.7" opacity="0.5"/>
    <path d="M30 160 C80 116 120 82 160 50" stroke="#e7b416" stroke-width="2.6" stroke-dasharray="0.1 5"/>
    <path d="M44 162 C90 120 130 86 168 56" stroke="#f0c63a" stroke-width="2.2" stroke-dasharray="0.1 4.4"/>
    <path d="M60 150 C96 118 126 92 150 72" stroke="#e7b416" stroke-width="2.4" stroke-dasharray="0.1 5"/>
    </g>"""),
]


def hero(model_name):
    # Classes: delay-N staggers the cycle; "still" marks the leaf shown when motion is reduced (potato)
    leaves = "".join(
        f'<svg class="hero-crop delay-{i}{" still" if i == 1 else ""}" viewBox="0 0 200 200">{svg}</svg>'
        for i, (_, svg) in enumerate(CROP_LEAVES))
    tags = "".join(
        f'<span class="hero-tag delay-{i}{" still" if i == 1 else ""}">{label}</span>'
        for i, (label, _) in enumerate(CROP_LEAVES))
    _render(f"""
    <div class="hero">
        <div class="hero-grid" aria-hidden="true"></div>
        <span class="hero-corner tl" aria-hidden="true"></span><span class="hero-corner tr" aria-hidden="true"></span>
        <span class="hero-corner bl" aria-hidden="true"></span><span class="hero-corner br" aria-hidden="true"></span>
        <div class="hero-text">
            <div class="hero-top">
                <span class="hero-live"><span class="hero-dot"></span>LIVE · {escape(model_name)}</span>
                <p class="hero-eyebrow">ICAR / CIBRC compliance-aware advisory</p>
            </div>
            <h1>Plant Disease <span class="hero-grad">Safe-RAG</span> Advisory</h1>
            <p class="hero-lead">Upload a leaf photo to get an AI-assisted diagnosis, a regulatory safety check
            and treatment guidance grounded in verified sources.</p>
            <div class="hero-stats">
                <div class="hero-stat"><strong>92.4%</strong><span>accuracy on field photos</span></div>
                <div class="hero-stat"><strong>99.3%</strong><span>correct when it answers a field photo</span></div>
                <div class="hero-stat"><strong>0 / 70</strong><span>off-label advice in tests</span></div>
                <div class="hero-stat"><strong>11</strong><span>report languages</span></div>
            </div>
            <div class="hero-chips">
                <span class="hero-chip">Maize · Potato · Rice · Wheat</span>
                <span class="hero-chip">PPQS / CIBRC verified rules</span>
            </div>
        </div>
        <div class="hero-visual" aria-hidden="true">
            <span class="hero-ring sweep"></span><span class="hero-ring dashed"></span><span class="hero-ring thin"></span>
            {leaves}
            <span class="hero-lockon"></span><span class="hero-scan"></span>
            {tags}
        </div>
    </div>
    """)


# Vision-scan background: the real pipeline with values from models/reliability.json, and after
# some steps what it means for the farmer in one of the report languages. Translations are
# short everyday phrases and should be checked by native speakers.
HUD_LOG = [
    ("", "load yolo11m_field_ft/best.pt"), ("", "resize 480x480 · rgb"),
    ("HI", "फ़ोटो मिल गई"),
    ("", "backbone C3k2 · SPPF · C2PSA"), ("", "logits[14] captured"),
    ("PA", "ਪੱਤੇ ਦੀ ਜਾਂਚ ਜਾਰੀ ਹੈ"),
    ("", "temperature T = 1.359"), ("", "softmax · p_max = 0.948"),
    ("TA", "உங்கள் இலையை கவனமாக பார்க்கிறோம்"),
    ("", "kNN k=10 score -0.41"), ("", "threshold -0.515 · known ✓"),
    ("BN", "এই ফসল আমরা চিনি"),
    ("", "abstain tau = 0.930 · answer ✓"),
    ("MR", "उत्तर देण्याइतकी खात्री आहे"),
    ("", "crop check · potato ✓"), ("", "label potato_early_blight"),
    ("OR", "କ୍ଷେତରେ ନିଶ୍ଚିତ କରନ୍ତୁ"),
    ("", "retrieve bge-large · k = 4"),
    ("TE", "ప్రభుత్వ సలహా వెతుకుతున్నాం"),
    ("", "hit ADV-004 · potato_early_blight"),
    ("GU", "સરકારી સલાહ તપાસી"),
    ("", "rule engine · APPROVED"), ("", "Propineb 70% WP · 300 g/100 L"),
    ("KN", "ಅನುಮೋದಿತ ಔಷಧಿ, ಸರಿಯಾದ ಪ್ರಮಾಣ"),
    ("", "gemini · report · English"),
    ("ML", "നിങ്ങളുടെ ഭാഷയിൽ ഉപദേശം തയ്യാർ"),
]
# (colour, top, width, height, machine label, confidence %, farmer caption, language)
HUD_BOXES = [
    ("green", "10vh", 170, 130, "potato_early_blight 0.95", 95, "हाँ, यह अगेती झुलसा है", "HI"),
    ("amber", "34vh", 150, 140, "rice_brown_spot 0.71 · withheld", 71, "இன்னொரு தெளிவான புகைப்படம் எடுங்கள்", "TA"),
    ("red", "58vh", 175, 110, "unknown_input · refused", 18, "ఈ పంటను మేము గుర్తించలేము", "TE"),
    ("green", "80vh", 140, 64, "wheat_yellow_rust 0.97", 97, "ਹਾਂ, ਇਹ ਪੀਲੀ ਕੁੰਗੀ ਹੈ", "PA"),
]
HUD_ASKS = [
    ("EN", "What is wrong with my crop?", ""),
    ("HI", "कौन सी दवा सुरक्षित है?", "Which spray is safe?"),
    ("TA", "என் பயிருக்கு என்ன நோய்?", "What disease does my crop have?"),
    ("TE", "పిచికారీ తర్వాత ఎప్పుడు కోయవచ్చు?", "When can I harvest after spraying?"),
    ("KN", "ಈ ಔಷಧಿ ಸುರಕ್ಷಿತವೇ?", "Is this spray safe?"),
    ("ML", "എന്റെ വിളയ്ക്ക് എന്ത് രോഗമാണ്?", "What disease does my crop have?"),
    ("MR", "फवारणीनंतर कापणी कधी करावी?", "When should I harvest after spraying?"),
    ("BN", "আমার ফসলে কী রোগ হয়েছে?", "What disease does my crop have?"),
    ("GU", "લિટર દીઠ કેટલી દવા?", "How much per litre?"),
    ("PA", "ਕਿਹੜੀ ਦਵਾਈ ਸੁਰੱਖਿਅਤ ਹੈ?", "Which spray is safe?"),
    ("OR", "ମୋ ଫସଲରେ କେଉଁ ରୋଗ ହୋଇଛି?", "What disease does my crop have?"),
]
HUD_STAGES = ["image", "YOLO11m", "kNN gate", "T-scale", "abstain", "RAG", "rules", "Gemini"]
HUD_HEAT = [(3, 8), (9, 17), (14, 27), (5, 37), (11, 47), (16, 57), (4, 67), (10, 77), (15, 87),
            (82, 10), (89, 20), (95, 29), (85, 39), (92, 49), (80, 59), (88, 69), (96, 78), (83, 88)]
LEAF_SHAPE = '<path d="M50 4 C82 22 92 58 50 96 C8 58 18 22 50 4 Z" fill="currentColor"/>'


def hud_background():
    """Animated vision-scan layer behind the page (CSS only; see the HUD section of style.css)."""
    def line(lang, text):
        if lang:
            return f'<p class="hud-line farmer"><b>↳</b> {text}<span class="hud-lang">{lang}</span></p>'
        return f'<p class="hud-line"><b>›</b> {text}</p>'

    log = "".join(line(lang, text) for lang, text in HUD_LOG) * 2   # twice, so the scroll loops
    heat = "".join(f'<span class="hud-heat" style="left:{x}%;top:{y}%;animation-delay:{y * 0.18:.2f}s"></span>'
                   for x, y in HUD_HEAT)
    boxes = "".join(
        f'<div class="hud-box {colour} d{i}" style="top:{top};width:{w}px;height:{h}px">'
        f'<svg viewBox="0 0 100 100">{LEAF_SHAPE}</svg><span class="hud-reticle"></span>'
        f'<span class="hud-tag">{label}</span><span class="hud-conf"><i style="width:{conf}%"></i></span>'
        f'<span class="hud-say">{say}<span class="hud-lang">{lang}</span></span></div>'
        for i, (colour, top, w, h, label, conf, say, lang) in enumerate(HUD_BOXES))
    asks = "".join(
        f'<p class="hud-ask" style="animation-delay:{i * 6:.1f}s">“{text}”<span class="hud-lang">{lang}</span>'
        f'{f"<small>{gloss}</small>" if gloss else ""}</p>'
        for i, (lang, text, gloss) in enumerate(HUD_ASKS))
    stages = "".join(f'<span class="hud-stage" style="animation-delay:{i * 1.5:.2f}s">{name}</span>'
                     for i, name in enumerate(HUD_STAGES))
    _render(f"""
    <div class="hud" aria-hidden="true">
        <div class="hud-view">
            <div class="hud-dots"></div><div class="hud-glow"></div>
            <div class="hud-floor"><span></span></div><div class="hud-radar"></div>
            {heat}<div class="hud-beam"></div>
            <div class="hud-log"><div class="hud-log-list">{log}</div></div>
            <div class="hud-boxes">{boxes}</div>
        </div>
        <div class="hud-foot"><div class="hud-asks">{asks}</div><div class="hud-stages">{stages}</div></div>
    </div>
    """)


@contextmanager
def analysis_scan(title, detail, step=0, language="English"):
    """Show the scanning-leaf card while the wrapped work runs, then remove it.

    step is the index of the stage in progress; phones also show the stages as a checklist."""
    stages = ["Checking the photo and identifying the disease", "Checking PPQS / CIBRC rules",
              f"Writing advice in {language}"]
    checklist = "".join(
        f'<li class="{"done" if i < step else "active" if i == step else "todo"}">'
        f'<span class="scan-step-mark">{"✓" if i < step else ""}</span>{escape(stage)}</li>'
        for i, stage in enumerate(stages))
    slot = st.empty()
    with slot.container():
        _render(f"""
        <div class="scan-card" role="status" aria-live="polite">
            <div class="leaf3d" data-leaf3d="scan" aria-hidden="true">{LEAF_POSTER}</div>
            <div class="scan-text">
                <p class="card-label">Working</p>
                <p class="scan-title">{escape(title)}</p>
                <p class="scan-sub">{escape(detail)}</p>
                <div class="scan-bar"><span></span></div>
            </div>
        </div>
        <ul class="scan-steps">{checklist}</ul>
        """)
    try:
        yield
    finally:
        slot.empty()


SUN_MOON = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
            'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
            '<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>')


def mobile_header():
    """Phones only (hidden on wider screens by style.css): app bar and a short intro instead of the hero."""
    stats = [("92.4%", "field accuracy"), ("99.3%", "right when it answers"),
             ("0/70", "off-label"), ("11", "languages")]
    chips = "".join(f'<span class="m-stat"><strong>{value}</strong> {label}</span>' for value, label in stats)
    _render(f"""
    <div class="m-only m-appbar">
        <span class="m-logo" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="none"
            stroke="#d6f0c8" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M5 19c0-8 5-13 14-14 0 9-5 14-13 14"/><path d="M5 19 13 11"/></svg></span>
        <span class="m-title"><strong>Safe-RAG Advisory</strong>
            <span class="m-live"><span class="hero-dot"></span>LIVE · YOLO11m field model</span></span>
        <button type="button" class="m-theme" aria-label="Switch between light and dark mode">{SUN_MOON}</button>
    </div>
    <div class="m-only m-intro">
        <h1>Diagnose a leaf. Get <span>verified, safe</span> advice.</h1>
        <p>AI diagnosis, a PPQS / CIBRC safety check and guidance in your language.</p>
        <div class="m-stats">{chips}</div>
    </div>
    """)


def mobile_scan_header():
    """Phones only: the top of the green "Scan a leaf" card; the file uploader below completes it."""
    leaves = "".join(
        f'<svg class="hero-crop delay-{i}{" still" if i == 1 else ""}" viewBox="0 0 200 200">{svg}</svg>'
        for i, (_, svg) in enumerate(CROP_LEAVES))
    _render(f"""
    <div class="m-only m-scan">
        <div class="m-scan-frame" aria-hidden="true">
            <span class="hero-ring dashed"></span>{leaves}<span class="hero-scan"></span>
        </div>
        <div>
            <h2>Scan a leaf</h2>
            <p>Take a photo or choose one: a single leaf, in daylight, in focus.</p>
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
        f'<div class="fact{" fact-dose" if label == "Dose" else ""}"><p class="fact-label">{label}</p>'
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


def unknown_input_card():
    _render("""
    <div class="card status-rejected">
        <div class="rule-head">
            <p class="rule-chemical">This photo does not match anything the model was trained on</p>
            <span class="badge rejected">✕ UNKNOWN IMAGE</span>
        </div>
        <p class="card-sub">It may be another crop, a disease outside the 14 supported conditions,
        or not a leaf photo. No diagnosis or treatment is given.</p>
        <p class="rule-note">Retake the photo with one maize, potato, rice or wheat leaf filling the frame,
        or show the plant to your local extension officer.</p>
    </div>
    """)


def low_confidence_card(crop, disease_key, confidence, threshold, alternatives):
    options = [disease_key] + [key for key, _ in alternatives]
    likely = " or ".join(escape(pretty_label(key.removeprefix(f"{crop}_"))) for key in options[:3])
    _render(f"""
    <div class="card status-warning">
        <div class="rule-head">
            <p class="rule-chemical">The model is not sure enough to advise</p>
            <span class="badge warning">! NOT SURE</span>
        </div>
        <p class="card-sub">Most likely: <strong>{likely}</strong>. Calibrated confidence is
        {confidence * 100:.0f}%, below the {threshold * 100:.0f}% needed before treatment advice is given.</p>
        <p class="rule-note">No chemical is recommended. Retake a sharper photo of one affected leaf in daylight,
        or confirm the disease with your local extension officer.</p>
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
            <div class="about-dev">
                <div class="about-avatar">RK</div>
                <div>
                    <p class="about-eyebrow">Developed by</p>
                    <p class="about-name">Ryan Kekulthotuwage</p>
                </div>
            </div>
            <p class="about-meta">Final-year Computer Science dissertation</p>
            <span class="about-badge">ICAR / CIBRC compliance standards</span>
        </div>
        <div class="side-card">
            <h4>How it works</h4>
            <ol>
                <li>YOLO11m classifies the leaf image.</li>
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
