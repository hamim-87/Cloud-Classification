"""
Rule-based weather analysis from detected cloud patterns.
Maps combinations of detected cloud types to meteorological descriptions.
"""

# ── Single cloud type descriptions ───────────────────────────────────────────

SINGLE_DESCRIPTIONS = {
    "Fish": (
        "Fish cloud patterns indicate organized deep convection along convergence zones. "
        "These formations are associated with moderate to heavy rainfall, converging wind "
        "fields, and reduced visibility. Expect showers and possible thunderstorm activity "
        "in the affected region. Fish patterns block 70-90% of incoming solar radiation."
    ),
    "Flower": (
        "Flower cloud patterns represent open-cell convection typically found in post-frontal "
        "environments. These indicate a stabilizing atmosphere with diminishing precipitation "
        "as conditions improve. The mix of cloudy and clear cells allows 40-60% solar "
        "transmission. Cold air outbreaks often produce these patterns with northwesterly "
        "flow in the northern hemisphere."
    ),
    "Gravel": (
        "Gravel cloud patterns are small, uniform cloud formations indicating a stable, "
        "well-mixed boundary layer. These shallow clouds produce no significant precipitation "
        "and permit 50-70% solar transmission. Weather conditions are generally fair with "
        "good visibility. Gravel patterns often form in stable trade wind environments."
    ),
    "Sugar": (
        "Sugar cloud patterns are very thin, scattered shallow clouds found in trade wind "
        "regions. They indicate steady easterly trade winds and a stable lower atmosphere. "
        "No precipitation is expected. These clouds allow 60-80% of solar radiation through "
        "and are associated with fair weather, excellent visibility, and calm to light winds."
    ),
}

# ── Combined cloud type descriptions ─────────────────────────────────────────

COMBINED_DESCRIPTIONS = {
    frozenset(["Fish", "Sugar"]): (
        "When both Fish and Sugar cloud patterns are observed together, it indicates "
        "a transition zone between active convective regions and stable trade wind "
        "environments. The weather may be variable with localized showers near Fish "
        "formations while surrounding areas remain fair. This combination suggests "
        "a convective disturbance embedded within an otherwise stable atmosphere."
    ),
    frozenset(["Flower", "Gravel"]): (
        "When Flower and Gravel patterns coexist, it indicates a post-frontal environment "
        "where the atmosphere is stabilizing. The open-cell Flower structures are gradually "
        "breaking down into the smaller, more uniform Gravel patterns. Weather conditions "
        "are improving with decreasing shower activity and increasing sunshine. This "
        "transition typically completes within 12-24 hours."
    ),
    frozenset(["Fish", "Flower"]): (
        "The simultaneous presence of Fish and Flower patterns indicates a complex "
        "atmospheric situation with both organized convection and post-frontal clearing. "
        "This may signal an approaching secondary cold front or a tropical disturbance "
        "interacting with a mid-latitude weather system. Forecast uncertainty is higher "
        "and conditions may change rapidly within 6-12 hours."
    ),
    frozenset(["Fish", "Gravel"]): (
        "Fish and Gravel patterns together suggest localized convective activity within "
        "a generally stable environment. Showers are likely confined to narrow convergence "
        "zones marked by Fish formations, while surrounding areas under Gravel patterns "
        "remain dry. This contrast indicates a spatially variable weather regime."
    ),
    frozenset(["Fish", "Flower", "Gravel"]): (
        "The presence of Fish, Flower, and Gravel patterns indicates a transitioning "
        "atmosphere with active convection (Fish), post-frontal clearing (Flower), and "
        "stable regions (Gravel). Weather is highly variable across the area — expect "
        "showers near Fish formations, improving conditions near Flower patterns, and "
        "fair weather under Gravel. The system is evolving and conditions may change "
        "within 6-12 hours."
    ),
    frozenset(["Fish", "Flower", "Sugar"]): (
        "This combination of Fish, Flower, and Sugar patterns reveals a complex "
        "interaction between organized convection, post-frontal dynamics, and trade wind "
        "stability. Precipitation is likely near Fish formations with clearing near "
        "Flower patterns and fair conditions under Sugar clouds. Rapid weather changes "
        "are possible as these regimes interact."
    ),
    frozenset(["Fish", "Gravel", "Sugar"]): (
        "Fish patterns alongside Gravel and Sugar indicate a convective disturbance "
        "embedded in an otherwise stable and fair environment. Localized heavy showers "
        "are expected near Fish formations while the broader region remains dry and "
        "sunny. The stable background suggests the convective activity is isolated "
        "and may dissipate within hours."
    ),
    frozenset(["Flower", "Gravel", "Sugar"]): (
        "Flower, Gravel, and Sugar patterns together indicate a predominantly stable "
        "and fair weather regime with remnants of post-frontal activity (Flower). "
        "No significant precipitation is expected. The atmosphere is trending toward "
        "full stability with excellent visibility and mostly sunny conditions."
    ),
    frozenset(["Fish", "Sugar", "Flower"]): None,  # covered above
    frozenset(["Fish", "Flower", "Gravel", "Sugar"]): (
        "When multiple cloud pattern types (Fish, Flower, Gravel, Sugar) are all detected "
        "in a satellite image, it indicates a highly complex atmospheric situation with "
        "multiple weather regimes coexisting. This is common near the boundaries of large "
        "weather systems, tropical convergence zones, or areas where different air masses "
        "interact. The forecast should consider the dominant pattern by coverage area as "
        "the primary weather driver while noting potential for rapid changes."
    ),
    frozenset(["Flower", "Sugar"]): (
        "Flower and Sugar patterns together indicate a fair weather environment with "
        "remnants of post-frontal clearing. The atmosphere is stable with no precipitation "
        "expected. Conditions are improving with increasing sunshine and light winds."
    ),
    frozenset(["Gravel", "Sugar"]): (
        "Gravel and Sugar patterns together indicate a very stable trade wind environment "
        "with fair weather. No precipitation is expected. These shallow cloud formations "
        "allow significant solar radiation through, resulting in warm, pleasant conditions "
        "with excellent visibility."
    ),
}

NO_CLOUDS_DESCRIPTION = (
    "No significant cloud patterns are detected in the satellite imagery. The "
    "atmosphere is dominated by clear or near-clear skies, indicating strong "
    "high-pressure dominance with subsiding air preventing cloud formation. Weather "
    "conditions are expected to be fair with excellent visibility exceeding 30 km, "
    "no precipitation, and calm to light winds. Surface temperatures may be higher "
    "than average due to unobstructed solar heating during the day, and cooler at "
    "night due to radiative cooling."
)

GENERAL_CONTEXT = (
    "\n\nGeneral context: Cloud altitude classification plays a crucial role in weather "
    "prediction. Atmospheric stability is a key factor — stable atmospheres produce flat, "
    "layered clouds with mild weather, while unstable atmospheres produce towering cumulus "
    "with showers and thunderstorms. Cloud pattern orientation and movement reveal wind "
    "conditions at cloud level."
)


def get_weather_analysis(detected_classes: list[str]) -> str:
    """
    Return a weather analysis string based on detected cloud types.
    Uses exact combination matching first, then falls back to individual descriptions.
    """
    if not detected_classes:
        return NO_CLOUDS_DESCRIPTION

    detected_set = frozenset(detected_classes)

    # Try exact combination match
    if detected_set in COMBINED_DESCRIPTIONS and COMBINED_DESCRIPTIONS[detected_set] is not None:
        return COMBINED_DESCRIPTIONS[detected_set] + GENERAL_CONTEXT

    # Single class
    if len(detected_classes) == 1:
        desc = SINGLE_DESCRIPTIONS.get(detected_classes[0], "")
        return desc + GENERAL_CONTEXT if desc else "Unknown cloud pattern detected."

    # Fallback: combine individual descriptions
    parts = []
    for cls in detected_classes:
        if cls in SINGLE_DESCRIPTIONS:
            parts.append(f"**{cls}**: {SINGLE_DESCRIPTIONS[cls]}")
    combined = "\n\n".join(parts)
    return combined + GENERAL_CONTEXT
