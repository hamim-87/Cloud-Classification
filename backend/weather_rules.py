"""
Structured fusion weather analysis from satellite cloud segmentation outputs.

Pipeline:
  1. Normalize coverage
  2. Compute fusion scores  (confidence × normalized_coverage)
  3. Convert to probability distribution
  4. Rank cloud types
  5. Interpret weather (dominant / mixed / uncertain)
  6. Return strict JSON-compatible dict
"""

from __future__ import annotations

EPSILON = 1e-8

CLASS_NAMES = ["Fish", "Flower", "Gravel", "Sugar"]

# ── Dominant weather descriptions (prob > 0.55) ─────────────────────────────

DOMINANT_DESCRIPTIONS = {
    "Gravel": (
        "Thick Gravel cloud cover dominates the scene, indicating an overcast boundary "
        "layer with high optical depth. Expect heavy cloud cover with limited solar "
        "radiation reaching the surface. There is a significant chance of drizzle or "
        "light rain from these dense, uniform cloud formations. Visibility may be "
        "reduced to 5–10 km. Surface temperatures will remain suppressed."
    ),
    "Flower": (
        "Flower cloud patterns dominate, indicating mid-level open-cell convection "
        "characteristic of a post-frontal or unstable environment. Expect variable "
        "conditions with intermittent showers possible as convective cells cycle. "
        "The mix of cloudy and clear cells allows 40–60 % solar transmission. "
        "Conditions may change over the next 6–12 hours as the instability evolves."
    ),
    "Fish": (
        "Fish cloud formations dominate, revealing organized deep convection along "
        "convergence zones and strong wind-shear patterns. Expect moderate to heavy "
        "rainfall, converging wind fields, and reduced visibility. Thunderstorm "
        "activity is likely in the core of these formations. Fish patterns block "
        "70–90 % of incoming solar radiation."
    ),
    "Sugar": (
        "Sugar cloud patterns dominate — thin, scattered shallow cumulus typical of "
        "trade-wind regions. Fair weather prevails with excellent visibility exceeding "
        "30 km, no precipitation, and calm to light easterly winds. Solar radiation "
        "transmission is 60–80 %. Conditions are expected to remain stable."
    ),
}

# ── Mixed weather descriptions (top-2 both > 0.30) ──────────────────────────

MIXED_DESCRIPTIONS = {
    frozenset(["Fish", "Flower"]): (
        "A mix of Fish and Flower patterns indicates organized convection interacting "
        "with post-frontal instability. Showers are likely near Fish formations while "
        "Flower cells bring intermittent clearing. Conditions may shift rapidly within "
        "6–12 hours. Visibility is variable."
    ),
    frozenset(["Fish", "Gravel"]): (
        "Fish formations embedded within a Gravel-dominated boundary layer suggest "
        "localized deep convection within an otherwise overcast environment. Heavy "
        "rain is confined to convergence zones while surrounding areas see drizzle "
        "or persistent cloud cover."
    ),
    frozenset(["Fish", "Sugar"]): (
        "Fish patterns alongside Sugar clouds indicate a convective disturbance "
        "embedded in an otherwise fair and stable trade-wind environment. Localized "
        "showers are expected near Fish formations while surrounding areas remain "
        "clear and dry."
    ),
    frozenset(["Flower", "Gravel"]): (
        "Flower and Gravel patterns coexist, indicating a post-frontal atmosphere "
        "where open-cell convection is gradually giving way to stable overcast. "
        "Shower activity is diminishing and conditions are trending toward thick "
        "but steady cloud cover with possible drizzle."
    ),
    frozenset(["Flower", "Sugar"]): (
        "Flower cells mixed with Sugar clouds suggest a post-frontal environment "
        "transitioning toward stability. Brief showers are possible near Flower "
        "formations but the overall trend is fair. Sunshine becomes more prevalent "
        "as Sugar patterns expand."
    ),
    frozenset(["Gravel", "Sugar"]): (
        "Gravel and Sugar patterns together indicate a boundary layer split between "
        "thick low cloud and scattered fair-weather cumulus. Expect partial cloud "
        "cover with no significant precipitation. Conditions are stable with moderate "
        "visibility."
    ),
}

UNCERTAIN_DESCRIPTION = (
    "All cloud types show similar, relatively low probabilities, indicating a "
    "transitional or uncertain atmospheric state. No single cloud regime dominates. "
    "Weather conditions are likely variable with mixed cloud cover, light and "
    "variable winds, and no strong precipitation signal. Further satellite passes "
    "are recommended to refine the forecast."
)

NO_CLOUDS_DESCRIPTION = (
    "No significant cloud patterns are detected in the satellite imagery. Clear "
    "or near-clear skies prevail under strong high-pressure dominance. Fair "
    "weather with excellent visibility (>30 km), no precipitation, and calm to "
    "light winds. Daytime surface temperatures may be above average due to "
    "unobstructed solar heating."
)



# ── Public API ───────────────────────────────────────────────────────────────

def compute_weather_fusion(
    class_results: dict[str, dict],
) -> dict:
    """
    Full 6-step structured fusion pipeline.

    Parameters
    ----------
    class_results : dict
        Keyed by class name ("Fish", "Flower", "Gravel", "Sugar").
        Each value must contain ``confidence`` (0–1 float) and
        ``coverage_percent`` (0–100 float).

    Returns
    -------
    dict  matching the strict output schema.
    """

    # Collect raw values (use 0 for missing classes) ──────────────────────
    confidences: dict[str, float] = {}
    coverages: dict[str, float] = {}
    for cls in CLASS_NAMES:
        info = class_results.get(cls, {})
        confidences[cls] = float(info.get("confidence", 0.0))
        coverages[cls] = float(info.get("coverage_percent", 0.0))

    # Step 1: Normalize coverage ──────────────────────────────────────────
    cov_sum = sum(coverages.values()) + EPSILON
    normalized_coverage = {
        cls: round(coverages[cls] / cov_sum, 4) for cls in CLASS_NAMES
    }

    # Step 2: Fusion score  (confidence × normalized_coverage) ────────────
    scores = {
        cls: round(confidences[cls] * normalized_coverage[cls], 4)
        for cls in CLASS_NAMES
    }

    # Step 3: Probability distribution ────────────────────────────────────
    score_sum = sum(scores.values()) + EPSILON
    probabilities = {
        cls: round(scores[cls] / score_sum, 4) for cls in CLASS_NAMES
    }

    # Step 4: Ranking (descending by probability) ─────────────────────────
    ranking = sorted(CLASS_NAMES, key=lambda c: probabilities[c], reverse=True)

    # Step 5: Weather interpretation ──────────────────────────────────────
    top1 = ranking[0]
    top2 = ranking[1]
    p1 = probabilities[top1]
    p2 = probabilities[top2]

    all_zero = all(coverages[c] == 0.0 for c in CLASS_NAMES)

    if all_zero:
        forecast_type = "dominant"
        primary = None
        secondary = None
        description = NO_CLOUDS_DESCRIPTION
    elif p1 > 0.55:
        forecast_type = "dominant"
        primary = top1
        secondary = None
        description = DOMINANT_DESCRIPTIONS.get(top1, "")
    elif p1 > 0.30 and p2 > 0.30:
        forecast_type = "mixed"
        primary = top1
        secondary = top2
        key = frozenset([top1, top2])
        description = MIXED_DESCRIPTIONS.get(key, "")
        if not description:
            # Fallback: concatenate dominant descriptions
            description = (
                f"{DOMINANT_DESCRIPTIONS.get(top1, '')} "
                f"Additionally, {DOMINANT_DESCRIPTIONS.get(top2, '')}"
            )
    else:
        forecast_type = "uncertain"
        primary = top1
        secondary = top2
        description = UNCERTAIN_DESCRIPTION

    description 

    forecast = {
        "type": forecast_type,
        "primary_cloud": primary,
        "secondary_cloud": secondary,
        "description": description,
    }

    return {
        "normalized_coverage": normalized_coverage,
        "scores": scores,
        "probabilities": probabilities,
        "ranking": ranking,
        "forecast": forecast,
    }


# ── Legacy helper (kept for backward compat) ─────────────────────────────────

def get_weather_analysis(detected_classes: list[str]) -> str:
    """Simple text-only analysis — used if caller has no coverage data."""
    if not detected_classes:
        return NO_CLOUDS_DESCRIPTION 
    if len(detected_classes) == 1:
        desc = DOMINANT_DESCRIPTIONS.get(detected_classes[0], "")
        return desc 
    parts = [
        f"**{c}**: {DOMINANT_DESCRIPTIONS.get(c, '')}"
        for c in detected_classes if c in DOMINANT_DESCRIPTIONS
    ]
    return "\n\n".join(parts) 
