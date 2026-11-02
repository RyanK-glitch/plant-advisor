# Interface: dark mode, vision-scan background and 3D leaf

Presentation-only changes. The diagnosis, rule engine, retrieval, Gemini calls, feedback and data handling are unchanged.

Screenshots in [`docs/ui/`](ui/): `dark-desktop.png`, `dark-hero.png`, `dark-working.png`, `dark-results.png`, `dark-bottom.png`, `light-desktop.png`, `dark-tablet.png`, `dark-mobile.png`, `reduced-hero.png`.

## What it looks like

| Part | What it shows |
|---|---|
| **Light and dark mode** | Both themes are defined, so Streamlit's Settings menu offers Light / Dark / System. **Dark is the default**: on a browser's first visit the page saves "Dark" and reloads once. After that the visitor's own choice is kept, including System. |
| **Hero ("AI console")** | Dark panel with a glowing outline, fine grid and HUD corner brackets; a pulsing `LIVE · model` pill; "Safe-RAG" in a shimmering gradient; four evaluation numbers (92.4 % field accuracy, 99.3 % correct when it answers a field photo, 0 / 70 off-label advice, 11 languages); and a scanning frame in which leaves of the four crops take turns (maize common rust, potato early blight, rice leaf blast, wheat yellow rust), each with its own tag. |
| **Vision-scan background** | Behind the content: dot grid, a perspective grid floor, a scanning beam, feature-map cells that flash as it passes, and a radar sweep. In the left gutter a log of the real pipeline (values from `models/reliability.json`) with farmer-friendly lines in the report languages; in the right gutter detections that lock on, labelled with real classes and outcomes (answered / withheld / refused) and what each means for the farmer; at the bottom of the page farmers' questions in all 11 languages and the pipeline ticker. |
| **Working card** | While YOLO11m and Gemini run, a card with the spinning 3D Blender leaf and a progress bar. |
| **Cards** | Rise into place and tilt slightly towards the mouse. |

The farmer-friendly lines are short everyday phrases written for this interface. **Have them checked by native speakers before relying on them.** They are in `HUD_LOG`, `HUD_BOXES` and `HUD_ASKS` in `src/ui_components.py`.


## Phones (≤ 640 px): app-style layout

Approved as phone mock-ups M1–M5 first. Desktop is unchanged: every rule is inside `@media (max-width: 640px)`, and the phone-only markup (`.m-only`, `.scan-steps`) is hidden on wider screens.

| Screen | On phones |
|---|---|
| Home | App bar (logo, name, LIVE, light / dark button) and a short intro with stat chips replace the hero. The green **Scan a leaf** card, with the cycling crop leaves, wraps the uploader and its large button, so it is on the first screen. Language and crop sit side by side under "Before you scan", location below; then the steps |
| Photo chosen | Full-width photo; **Run Safe-RAG Analysis is pinned to the bottom of the screen** until a result appears |
| Analysing | One card plus a checklist of the stages (photo and disease, rules, advice); the duplicate spinner line is hidden |
| Result | Compact diagnosis with the ring beside it; in the safety card the **dose is highlighted** and the other facts are rows |
| Advice | Small section labels, 0.92 rem text, a pill "read aloud" button, and Yes / No / Not sure as three large buttons |

How it works: the settings are wrapped in `st.container(key="settings_block")` and shown after the scan card with CSS `order`; `ui.mobile_header()`, `ui.mobile_scan_header()` and the checklist in `ui.analysis_scan(..., step=)` add the phone-only pieces; the app bar's theme button saves Light / Dark the way Streamlit's Settings menu does and reloads (`assets/motion.js`). Screenshots: `docs/ui/mobile-p1-home.png` … `mobile-p5-feedback.png`, `mobile-p1-light.png`, and `desktop-after-mobile-work.png` for comparison.

## Files

| File | Change |
|---|---|
| `.streamlit/config.toml` | `[theme.light]` and `[theme.dark]` (turns on the Light / Dark switch); static serving for `static/` |
| `assets/style.css` | Colour tokens instead of fixed colours; console hero; vision-scan background; content column 840 px so the gutters fit the log and detections |
| `assets/dark-tokens.css` | **New.** Dark values of the tokens (declarations only) |
| `assets/motion.js` | Dark-by-default on first visit; follows the active theme (`html.theme-dark` / `theme-light`); measures the main area for the background; 3D leaf for the Working card; card tilt |
| `src/ui_components.py` | `load_css()` (stylesheet + dark tokens via `st.html`), new `hero()` with the four crop leaves, new `hud_background()`, `analysis_scan()` |
| `app.py` | Calls `ui.load_motion()` and `ui.hud_background()`; wraps the two waits in `ui.analysis_scan(...)`. No logic changed |
| `scripts/assets/build_leaf.py`, `static/3d/` | Blender leaf (`leaf.blend`, `leaf.glb`, `leaf.webp`) used by the Working card |
| `tests/test_ui_assets.py` | 4 checks: motion.js survives Streamlit's sanitiser, assets exist and are small, static serving on, both themes configured with matching tokens |

## How the theming works

- Streamlit's own widgets (selects, uploader, buttons, expanders) take their colours from `config.toml`.
- The custom HTML cards use CSS tokens. `motion.js` reads the background colour Streamlit paints and sets `html.theme-dark` or `html.theme-light`, so the cards switch at the same moment as the widgets. Before the script runs, the server's `st.context.theme.type` is used as a first guess.
- Dark-by-default relies on Streamlit's stored setting `stActiveTheme-(page path)-v2` (Streamlit 1.66). If a future Streamlit changes that key, the app falls back to following the device setting; update the key in `motion.js`.
- The first visit on a device set to light mode reloads once to switch to dark.

## Libraries

- **three.js 0.160.0** from the jsDelivr CDN (pinned), only for the Working card's 3D leaf. The project's only package manager is pip, and Streamlit's static folder serves `.js` as `text/plain`, which browsers will not run as modules. If the CDN cannot be reached, the still image stays.
- **Fonts:** IBM Plex Mono (technical lines) and Noto Sans for Devanagari, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Gurmukhi and Odia (farmer lines), from Google Fonts. Without internet the system fonts are used.
- Everything else is CSS: no extra JavaScript for the background or the hero.

## Running it

```bash
streamlit run app.py
```

Restart the server after changing `src/ui_components.py`: Streamlit keeps imported modules cached. Docker: rebuild the image, since `static/` and the assets changed.

## Editing

- **Background text and values:** `HUD_LOG`, `HUD_BOXES`, `HUD_ASKS`, `HUD_STAGES` in `src/ui_components.py`.
- **Hero leaves:** `CROP_LEAVES` (inline SVG, one per crop, with its tag).
- **Colours:** light values at the top of `assets/style.css`, dark values in `assets/dark-tokens.css` (same names).
- **The Blender leaf:** change `scripts/assets/build_leaf.py` and run `blender -b -P scripts/assets/build_leaf.py`, or open `static/3d/leaf.blend` and export **File → Export → glTF 2.0 (.glb)** with *Selected Objects*, *Animation* and *Shape Keys*.
- Never add tag-like text such as `<div` to `assets/motion.js`, even in comments: Streamlit's sanitiser silently drops the whole script (a test guards this).

## Performance, accessibility and limits

- The background and hero are CSS animations of transform and opacity, which browsers run on the compositor. The gutter log and detections only appear when the main area is at least 1270 / 1300 px wide; on tablets and phones only the grid, beam and bottom questions remain.
- **Reduced motion:** everything holds still; the hero shows the potato leaf and its tag, and the background shows a static frame.
- Checked in Chromium at 1600, 820 and 390 px in dark and light, with reduced motion: no console errors, no horizontal scrolling. Not tested on Safari, Firefox or real phones.
- Content column narrowed from 960 to 840 px so the side gutters can hold the background's log and detections.
