import re


class AgriculturalRuleEngine:
    """Deterministic safety layer for verified India-focused treatment rules."""

    APPROVED = "APPROVED"
    WARNING = "WARNING"
    REJECTED = "REJECTED"

    def __init__(self):
        # Rules audited against PPQS/CIBRC sources on 22 September 2026
        self.rules = {
            "maize_common_rust": self._rule(
                "Azoxystrobin 18.2% + Cyproconazole 7.3% SC",
                "1 ml/L; 500 L water/ha", "2 per season", "52 days", "24 hours",
                "Use only if rust is confirmed. Spray evenly and do not increase the label rate.",
                "https://ppqs.gov.in/sites/default/files/azoxystrobin_18.2_cyproconazole_7.3_sc93fsyngenta_india_limited_1-merged.pdf"
            ),
            "maize_gray_leaf_spot": self._rule(
                "Azoxystrobin 7.5% + Propiconazole 12.5% SE",
                "Do not auto-dose", "Check label", "Check label", "Check label",
                "The source states maize leaf spot, not specifically gray leaf spot. Manual label confirmation is required.",
                "https://ppqs.gov.in/sites/default/files/427_rc_minutes.pdf",
                "PARTIAL - MANUAL CHECK"
            ),
            "maize_northern_leaf_blight": self._rule(
                "Azoxystrobin 18.2% + Cyproconazole 7.3% SC",
                "1 ml/L; 500 L water/ha", "2 per season", "52 days", "24 hours",
                "Start early when long cigar-shaped lesions appear. Good leaf coverage is important.",
                "https://ppqs.gov.in/sites/default/files/azoxystrobin_18.2_cyproconazole_7.3_sc93fsyngenta_india_limited_1-merged.pdf"
            ),
            "potato_early_blight": self._rule(
                "Propineb 70% WP", "300 g/100 L water (0.30%)", "Follow label",
                "Follow label", "Follow label",
                "Remove badly infected leaves and avoid keeping foliage wet for long periods.",
                "https://ppqs.gov.in/sites/default/files/potato.pdf",
                "VERIFIED - PPQS IPM / RC"
            ),
            "potato_late_blight": self._rule(
                "Metalaxyl-M 4% + Mancozeb 64% WP",
                "2500 g formulation/ha in 500-1000 L water", "Maximum 2", "24 days", "24 hours",
                "Late blight spreads rapidly in cool, wet weather. Start control early. "
                "ICAR has reported metalaxyl-resistant late-blight strains in India, so do not rely on "
                "this product alone; rotate with other registered fungicides as locally advised.",
                "https://ppqs.gov.in/sites/default/files/metalaxyl-m_4_mancozeb_64_wp93fsyngenta_india_limited_1.pdf"
            ),
            "rice_brown_spot": self._rule(
                "Azoxystrobin 16.7% + Tricyclazole 33.3% SC",
                "500 ml/ha in 500 L water", "2-3 as stated on label", "24 days", "24 hours",
                "Correct crop nutrition because brown spot is often worse in weak crops.",
                "https://ppqs.gov.in/sites/default/files/azoxystrobin_16.7_tricyclazole_33.3_sc93fcrystal_crop_protection_ltd_1-merged.pdf"
            ),
            "rice_leaf_blast": self._rule(
                "Propiconazole 10.7% + Tricyclazole 34.2% SE",
                "625 ml/ha in 500 L water", "Maximum 2", "32 days", "48 hours",
                "Act when spindle-shaped lesions appear. Avoid excessive nitrogen.",
                "https://ppqs.gov.in/sites/default/files/propiconazole_10.7_tricyclazole_34.2_se93fsyngenta_india_limited_1.pdf",
                "VERIFIED - BLAST"
            ),
            "rice_neck_blast": self._rule(
                "Propiconazole 10.7% + Tricyclazole 34.2% SE",
                "625 ml/ha in 500 L water", "Maximum 2", "32 days", "48 hours",
                "Protect the crop around the reproductive stage and follow the current registered label.",
                "https://ppqs.gov.in/sites/default/files/propiconazole_10.7_tricyclazole_34.2_se93fsyngenta_india_limited_1.pdf",
                "VERIFIED - BLAST"
            ),
            "wheat_brown_rust": self._rule(
                "Propiconazole 25% EC", "500 ml/ha in 750 L water", "Follow label",
                "30 days", "Follow label",
                "Treat confirmed rust early and use resistant varieties where available.",
                "https://ppqs.gov.in/sites/default/files/wheat_rust-english.pdf"
            ),
            "wheat_yellow_rust": self._rule(
                "Propiconazole 25% EC", "500 ml/ha in 750 L water", "Follow label",
                "30 days", "Follow label",
                "Treat confirmed yellow rust early before infection becomes severe.",
                "https://ppqs.gov.in/sites/default/files/wheat_rust-english.pdf"
            )
        }

        # Application timing copied from the corresponding audited records
        timings = {
            "maize_common_rust": "Preventive or at first appearance; repeat after 10-15 days if needed",
            "maize_gray_leaf_spot": "Use only after the exact current label is manually confirmed",
            "maize_northern_leaf_blight": "Preventive or at first appearance; repeat after 10-15 days if needed",
            "potato_early_blight": "Apply when disease appears and follow current label intervals",
            "potato_late_blight": "Preventive or at first appearance; repeat after 10-15 days if needed",
            "rice_brown_spot": "Pre-tillering to mid-tillering and at flowering",
            "rice_leaf_blast": "Preventive or at first infection; repeat after 10-15 days if needed",
            "rice_neck_blast": "Preventive or at first infection; repeat after 10-15 days if needed",
            "wheat_brown_rust": "Apply when rust appears and follow the current label",
            "wheat_yellow_rust": "Apply when yellow rust appears and follow the current label"
        }
        for disease_key, timing in timings.items():
            self.rules[disease_key]["timing"] = timing

        # Healthy predictions must never trigger pesticide advice
        for crop in ("maize", "potato", "rice", "wheat"):
            self.rules[f"{crop}_healthy"] = self._rule(
                "NONE", "Not applicable", "0", "Not applicable", "Not applicable",
                "No target disease was detected. Continue monitoring and do not spray based only on this image.",
                "", "NO CHEMICAL ADVISED"
            )
            self.rules[f"{crop}_healthy"]["timing"] = "Monitor only"

    @staticmethod
    def _rule(chemical, dose, applications, phi, re_entry, note, source, status="VERIFIED"):
        return {
            "regulatory_status": status, "chemical": chemical, "dose": dose,
            "max_applications": applications, "phi": phi, "re_entry": re_entry,
            "ipm_note": note, "source_url": source
        }

    @staticmethod
    def _key(value):
        return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", value.lower())).strip("_")

    def get_rule(self, crop, disease):
        disease_key = self._key(disease)
        crop_key = "maize" if self._key(crop) == "corn" else self._key(crop)

        if disease_key == "rice_blast":
            disease_key = "rice_leaf_blast"
        elif not disease_key.startswith(f"{crop_key}_"):
            disease_key = f"{crop_key}_{disease_key}"

        return self.rules.get(disease_key)

    def evaluate(self, crop, disease):
        rule = self.get_rule(crop, disease)

        if not rule:
            return {"engine_status": self.REJECTED, "message": "No verified rule exists."}

        status = rule["regulatory_status"].upper()
        if status.startswith("VERIFIED"):
            engine_status = self.APPROVED
        elif status == "NO CHEMICAL ADVISED":
            engine_status = self.APPROVED
        else:
            engine_status = self.WARNING

        return {**rule, "engine_status": engine_status}
