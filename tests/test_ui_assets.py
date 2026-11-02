"""Checks for the 3D leaf interface assets (presentation layer only)."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_motion_script_survives_streamlit_sanitiser():
    # st.html cleans its body with DOMPurify, which silently drops a script whose text contains
    # anything tag-like ("<div", "</p", ...). The 3D leaf would then never start.
    script = (ROOT / "assets" / "motion.js").read_text()
    assert not re.search(r"<[/A-Za-z]", script)


def test_leaf_assets_are_present_and_small():
    folder = ROOT / "static" / "3d"
    assert (folder / "leaf.blend").is_file(), "Blender source must sit next to the exports"
    assert (folder / "leaf.glb").stat().st_size < 150_000
    assert (folder / "leaf.webp").stat().st_size < 50_000


def test_static_serving_is_enabled():
    assert re.search(r"enableStaticServing\s*=\s*true", (ROOT / ".streamlit" / "config.toml").read_text())


def test_dark_mode_is_configured():
    config = (ROOT / ".streamlit" / "config.toml").read_text()
    assert "[theme.light]" in config and "[theme.dark]" in config   # both: Settings offers Light / Dark
    tokens = (ROOT / "assets" / "dark-tokens.css").read_text()
    assert "{" not in tokens and "}" not in tokens                    # declarations only, wrapped in Python
    light = (ROOT / "assets" / "style.css").read_text()
    for name in ("--page", "--card", "--ink", "--sidebar", "--hud-mono", "--friendly"):
        assert f"{name}:" in light and f"{name}:" in tokens
