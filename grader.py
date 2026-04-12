import math


# ─────────────────────────────────────────────
# STRICT CLAMP (NEVER returns 0 or 1)
# ─────────────────────────────────────────────

def clamp(score: float) -> float:
    try:
        val = float(score)

        # Handle NaN / inf
        if not math.isfinite(val):
            return 0.021

        # Strict boundaries (never equal)
        if val <= 0.02:
            return 0.021
        if val >= 0.98:
            return 0.979

        return round(val, 4)

    except Exception:
        return 0.021


# ─────────────────────────────────────────────
# SAFE VALUE EXTRACTION
# ─────────────────────────────────────────────

def _get_val(obj, key, default=0.0):
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    except Exception:
        return default


# ─────────────────────────────────────────────
# EASY
# ─────────────────────────────────────────────

def grade_easy(obs):
    try:
        pos = float(_get_val(obs, "position", 0.0))
        goal = float(_get_val(obs, "goal", 50.0))

        progress = pos / goal if goal > 0 else 0.0
        score = 0.05 + 0.9 * progress

        return clamp(score)

    except Exception:
        return clamp(0.05)


# ─────────────────────────────────────────────
# MEDIUM
# ─────────────────────────────────────────────

def grade_medium(obs):
    try:
        pos = float(_get_val(obs, "position", 0.0))
        goal = float(_get_val(obs, "goal", 75.0))
        speed = float(_get_val(obs, "speed", 0.0))
        visibility = float(_get_val(obs, "visibility", 1.0))

        progress = pos / goal if goal > 0 else 0.0

        penalty = 0.0
        if visibility < 0.6:
            penalty += 0.2
        if speed > 10:
            penalty += 0.2

        score = progress - penalty

        if pos >= goal:
            score = 0.9 - penalty

        return clamp(score)

    except Exception:
        return clamp(0.05)


# ─────────────────────────────────────────────
# HARD
# ─────────────────────────────────────────────

def grade_hard(obs):
    try:
        pos = float(_get_val(obs, "position", 0.0))
        goal = float(_get_val(obs, "goal", 110.0))
        battery = float(_get_val(obs, "battery", 100.0))

        progress = pos / goal if goal > 0 else 0.0
        battery_factor = battery / 100.0

        score = 0.05 + (0.65 * progress) + (0.25 * battery_factor)

        if pos >= goal and battery > 0:
            score = 0.9

        return clamp(score)

    except Exception:
        return clamp(0.05)


# ─────────────────────────────────────────────
# TASK MAPPING
# ─────────────────────────────────────────────

GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


# ─────────────────────────────────────────────
# MAIN ENTRYPOINT
# ─────────────────────────────────────────────

def grade(task_id: str, *args, **kwargs):
    """
    Final validator-safe entrypoint.
    ALWAYS returns strictly between (0,1)
    """
    try:
        fn = GRADERS.get(str(task_id).lower())

        if not fn:
            return clamp(0.05)

        # Extract observation safely
        obs = args[0] if args else (
            kwargs.get("observation") or kwargs.get("obs")
        )

        if obs is None:
            return clamp(0.05)

        raw_score = fn(obs)
        return clamp(raw_score)

    except Exception:
        return clamp(0.05)