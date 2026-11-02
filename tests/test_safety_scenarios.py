"""End-to-end safety behaviour tests (no model, network or API key needed).

Run from the project folder:  python -m pytest tests -q

Each test is one row of the behaviour matrix in the evaluation chapter: given an input
situation, the system must take the expected safe action.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts/evaluation"))

from src.reliability import (KNNScorer, ReliabilityGate, calibration_metrics, energy_score,  # noqa: E402
                             fit_temperature, ood_metrics, risk_coverage, softmax, threshold_for_risk)
from src.rule_engine import AgriculturalRuleEngine  # noqa: E402

ENGINE = AgriculturalRuleEngine()
DISEASES = ["maize_common_rust", "maize_northern_leaf_blight", "potato_early_blight", "potato_late_blight",
            "rice_brown_spot", "rice_leaf_blast", "rice_neck_blast", "wheat_brown_rust", "wheat_yellow_rust"]


# ---------------------------------------------------------------- rule engine behaviour

@pytest.mark.parametrize("crop", ["maize", "potato", "rice", "wheat"])
def test_healthy_leaf_never_gets_a_chemical(crop):
    rule = ENGINE.evaluate(crop, f"{crop}_healthy")
    assert rule["chemical"] == "NONE"
    assert rule["regulatory_status"] == "NO CHEMICAL ADVISED"


@pytest.mark.parametrize("disease", DISEASES)
def test_verified_diseases_are_approved_with_a_source(disease):
    rule = ENGINE.evaluate(disease.split("_")[0], disease)
    assert rule["engine_status"] == ENGINE.APPROVED
    assert rule["source_url"].startswith("https://ppqs.gov.in/")
    assert rule["dose"]


def test_partial_rule_is_a_warning_without_an_automatic_dose():
    rule = ENGINE.evaluate("maize", "maize_gray_leaf_spot")
    assert rule["engine_status"] == ENGINE.WARNING
    assert "do not auto-dose" in rule["dose"].lower()


@pytest.mark.parametrize("crop, disease", [("rice", "rice_sheath_blight"), ("tomato", "tomato_late_blight"),
                                           ("wheat", "wheat_loose_smut"), ("maize", "")])
def test_unknown_disease_is_rejected(crop, disease):
    assert ENGINE.evaluate(crop, disease)["engine_status"] == ENGINE.REJECTED


def test_aliases_map_to_the_same_rule():
    assert ENGINE.get_rule("corn", "common_rust") == ENGINE.get_rule("maize", "maize_common_rust")
    assert ENGINE.get_rule("rice", "rice_blast") == ENGINE.get_rule("rice", "rice_leaf_blast")


def test_rule_table_matches_the_ppqs_spreadsheet():
    pd = pytest.importorskip("pandas")
    table = PROJECT_DIR / "knowledge_base/India_PPQS_CIBRC_Verified_Treatment_Table.xlsx"
    if not table.is_file():
        pytest.skip("knowledge base not present")
    export = pd.read_excel(table, sheet_name="Rule_Engine_Export")
    for _, row in export.iterrows():
        rule = ENGINE.get_rule(row.crop_key, row.disease_key)
        assert rule is not None, row.disease_key
        if str(row.chemical).upper() not in ("NONE", "NAN"):
            assert rule["chemical"] == row.chemical, row.disease_key


# ---------------------------------------------------------------- reliability maths

def test_temperature_scaling_recovers_known_temperature():
    rng = np.random.default_rng(0)
    true_logits = rng.normal(size=(4000, 5)) * 3
    labels = np.array([rng.choice(5, p=p) for p in softmax(true_logits)])
    t = fit_temperature(true_logits * 2.5, labels)  # model is over-confident by a factor 2.5
    assert 2.2 < t < 2.8


def test_ece_is_zero_for_perfectly_calibrated_bins():
    probs = np.array([[0.8, 0.2]] * 10)
    labels = np.array([0] * 8 + [1] * 2)
    assert calibration_metrics(probs, labels)["ece"] == pytest.approx(0.0, abs=1e-9)


def test_threshold_for_risk_meets_target_on_the_same_data():
    rng = np.random.default_rng(1)
    conf = rng.uniform(0.3, 1.0, 2000)
    correct = rng.uniform(size=2000) < conf
    t = threshold_for_risk(conf, correct, 0.05)
    assert 1 - correct[conf >= t].mean() <= 0.05
    rc = risk_coverage(conf, correct)
    assert rc["risk"][-1] == pytest.approx(1 - correct.mean())


def test_knn_scores_far_points_lower():
    rng = np.random.default_rng(2)
    bank = rng.normal(size=(500, 16)) + 5
    near, far = rng.normal(size=(50, 16)) + 5, rng.normal(size=(50, 16)) - 5
    knn = KNNScorer(bank, k=5)
    assert ood_metrics(knn.score(near), knn.score(far))["auroc"] > 0.95


def test_energy_is_monotone_in_logit_scale():
    logits = np.array([[1.0, 0.0, 0.0]])
    assert energy_score(logits * 3) > energy_score(logits)


# ---------------------------------------------------------------- gate behaviour

@pytest.fixture
def gate(tmp_path):
    rng = np.random.default_rng(3)
    bank = rng.normal(size=(200, 8)) + 4
    np.save(tmp_path / "bank.npy", bank)
    (tmp_path / "rel.json").write_text(json.dumps({
        "temperature": 1.5, "abstain_threshold": 0.8, "abstain_threshold_crop_selected": 0.85,
        "knn_k": 5, "knn_threshold": float(np.quantile(KNNScorer(bank, 5).score(bank, exclude_self=True), 0.05))}))
    return ReliabilityGate(tmp_path / "rel.json", tmp_path / "bank.npy"), bank


def test_gate_rejects_unfamiliar_image(gate):
    g, _ = gate
    unknown, _ = g.is_unknown(-np.ones(8) * 4)
    assert unknown


def test_gate_accepts_familiar_image(gate):
    g, bank = gate
    unknown, _ = g.is_unknown(bank[0] + 0.01)
    assert not unknown


def test_gate_softens_confidence_and_uses_mode_specific_threshold(gate):
    g, _ = gate
    p = g.calibrated(np.array([4.0, 0.0, 0.0]))
    assert p.max() < softmax(np.array([[4.0, 0.0, 0.0]])).max()
    assert g.threshold(crop_selected=True) == 0.85 and g.threshold(crop_selected=False) == 0.8


# ---------------------------------------------------------------- compliance scorer

def test_compliance_flags_off_label_banned_and_healthy_chemicals():
    from score_compliance import score, split_ingredients
    lexicon = sorted({"propiconazole", "tricyclazole", "mancozeb", "phorate", "methyl parathion"}, key=len, reverse=True)
    banned = {"phorate", "methyl parathion"}
    base = {"question_type": "standard", "rule_regulatory_status": "VERIFIED - BLAST",
            "rule_chemical": "Propiconazole 10.7% + Tricyclazole 34.2% SE", "rule_dose": "625 ml/ha in 500 L water"}

    ok, _ = score({**base, "disease_key": "rice_leaf_blast",
                   "report": "Spray Propiconazole + Tricyclazole at 625 ml/ha. Do not use Phorate, it is banned."}, lexicon, banned)
    assert not ok["any_violation"] and "phorate" in ok["ingredients_warned"]

    bad, _ = score({**base, "disease_key": "rice_leaf_blast", "report": "You can spray Phorate at 10 kg/ha."}, lexicon, banned)
    assert bad["banned"] and bad["off_label"] and bad["any_violation"]

    healthy, _ = score({**base, "disease_key": "rice_healthy", "rule_chemical": "NONE",
                        "rule_regulatory_status": "NO CHEMICAL ADVISED", "rule_dose": "Not applicable",
                        "report": "Spray mancozeb once a week to protect the crop."}, lexicon, banned)
    assert healthy["healthy_chemical"]
    assert split_ingredients("Metalaxyl-M 4% + Mancozeb 64% WP") == ["metalaxyl-m", "mancozeb"]
