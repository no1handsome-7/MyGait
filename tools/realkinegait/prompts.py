NUISANCE_PROMPT = """
You are generating structured annotations for gait recognition. Describe only nuisance factors that should NOT be identity cues.
Return strict JSON with keys: attire, clothing_geometry_effect, carried_objects, viewpoint, illumination, background, occlusion, image_quality.
Rules:
- Do not guess identity, age, gender, race, or sensitive attributes.
- Focus on clothing color/texture, loose/tight clothing, bags, umbrellas, camera view, lighting, background and occlusion.
- clothing_geometry_effect should explain how clothes may distort the silhouette, e.g. expanded torso, hidden arm gap, skirt-like lower-body contour.
- Use short phrases; no markdown.
""".strip()

MOTION_PROMPT = """
You are generating motion-language annotations for gait recognition. Describe only body-part motion and walking phase information that SHOULD be useful for identity-invariant gait.
Return strict JSON with keys: gait_phase, left_arm, right_arm, left_leg, right_leg, torso_pelvis, coordination, confidence.
Rules:
- Do not describe clothing, color, background, or identity.
- Use biomechanical language when possible: stance, swing, heel-strike proxy, cadence, arm-leg anti-phase, torso stability.
- If motion is uncertain, set confidence to low and state why.
- Use short phrases; no markdown.
""".strip()
