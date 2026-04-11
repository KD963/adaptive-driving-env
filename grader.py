def clamp(score: float) -> float:
    score = float(score)
    return max(0.02, min(round(score, 4), 0.98))


def _get(obs, key, default=0.0):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


# ─────────────────────────────────────────────────────────────

def grade_easy(action, observation):
    position = float(_get(observation, "position", 0.0))
    goal     = float(_get(observation, "goal", 50.0))

    progress = position / goal if goal > 0 else 0.0
    score = 0.05 + 0.9 * progress

    if position >= goal:
        score = 0.95

    return clamp(score)


def grade_medium(action, observation):
    position   = float(_get(observation, "position", 0.0))
    speed      = float(_get(observation, "speed", 0.0))
    visibility = float(_get(observation, "visibility", 1.0))
    goal       = float(_get(observation, "goal", 70.0))

    progress = position / goal if goal > 0 else 0.0

    penalty = 0.0
    if visibility < 0.6:
        penalty += 0.2
    if speed > 10:
        penalty += 0.2

    raw = progress - penalty
    score = 0.05 + 0.9 * max(0.01, raw)

    if position >= goal:
        score = 0.9 - penalty

    return clamp(score)


def grade_hard(action, observation):
    position = float(_get(observation, "position", 0.0))
    battery  = float(_get(observation, "battery", 0.0))
    goal     = float(_get(observation, "goal", 100.0))

    progress = position / goal if goal > 0 else 0.0
    battery_factor = max(0.0, min(battery / 100.0, 1.0))

    combined = 0.7 * progress + 0.3 * battery_factor
    score = 0.05 + 0.9 * combined

    if position >= goal and battery > 0:
        score = 0.9 + 0.05 * battery_factor

    return clamp(score)


# ─────────────────────────────────────────────


GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}


def grade(task_id: str, obs):
    fn = GRADERS.get(task_id)
    if not fn:
        raise ValueError(f"Invalid task_id: {task_id}")

    score = fn(None, obs)   # ✅ FIXED

    # 🚨 FINAL VALIDATOR GUARD
    if score <= 0.0:
        score = 0.02
    elif score >= 1.0:
        score = 0.98

    return score