"""
Graders for Adaptive Driving Environment.

The OpenEnv validator may call these as:
    grade_easy(observation)              # 1-arg
    grade_easy(action, observation)      # 2-arg

Both signatures are supported via *args.
Each grader returns a float STRICTLY in (0.02, 0.98).
"""


def clamp(score: float) -> float:
    return max(0.02, min(round(float(score), 4), 0.98))


def _get(obs, key, default=0.0):
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


# ── Graders ───────────────────────────────────────────────────────────────────

def grade_easy(*args) -> float:
    """
    Accepts: grade_easy(obs)  OR  grade_easy(action, obs)
    Score = 0.05 + 0.9 * progress. Reaching goal → 0.95.
    """
    observation = args[-1]   # last arg is always the observation
    position = float(_get(observation, "position", 0.0))
    goal     = float(_get(observation, "goal", 50.0))

    if goal <= 0:
        return 0.05

    progress = min(position / goal, 1.0)
    score    = 0.05 + 0.9 * progress

    if position >= goal:
        return 0.95

    return clamp(score)


def grade_medium(*args) -> float:
    """
    Accepts: grade_medium(obs)  OR  grade_medium(action, obs)
    Score = progress − safety penalties. Penalises low visibility and high speed.
    """
    observation = args[-1]
    position   = float(_get(observation, "position",   0.0))
    speed      = float(_get(observation, "speed",      0.0))
    visibility = float(_get(observation, "visibility", 1.0))
    goal       = float(_get(observation, "goal", 70.0))

    if goal <= 0:
        return 0.05

    progress = min(position / goal, 1.0)

    penalty = 0.0
    if visibility < 0.6:
        penalty += 0.2
    if speed > 10:
        penalty += 0.2

    if position >= goal:
        return clamp(0.9 - penalty)

    raw   = max(0.0, progress - penalty)
    score = 0.05 + 0.9 * raw
    return clamp(score)


def grade_hard(*args) -> float:
    """
    Accepts: grade_hard(obs)  OR  grade_hard(action, obs)
    Score = 70% progress + 30% battery efficiency.
    """
    observation = args[-1]
    position = float(_get(observation, "position", 0.0))
    battery  = float(_get(observation, "battery",  0.0))
    goal     = float(_get(observation, "goal", 100.0))

    if goal <= 0:
        return 0.05

    progress       = min(position / goal, 1.0)
    battery_factor = max(0.0, min(battery / 100.0, 1.0))

    if position >= goal and battery > 0:
        score = 0.9 + 0.05 * battery_factor
        return clamp(score)

    combined = 0.7 * progress + 0.3 * battery_factor
    score    = 0.05 + 0.9 * combined
    return clamp(score)


# ── Registry ──────────────────────────────────────────────────────────────────

GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}


def grade(task_id: str, obs) -> float:
    """Internal entry point used by the environment's step()."""
    fn = GRADERS.get(task_id)
    if not fn:
        raise ValueError(f"Invalid task_id: {task_id!r}. Valid: {list(GRADERS)}")
    score = fn(obs)
    return max(0.02, min(float(score), 0.98))