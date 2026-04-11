"""
Graders for Adaptive Driving Environment.

Called by the OpenEnv validator as:
    score = grade_easy(obs)

where `obs` is either:
  - an AdaptiveDrivingObservation Pydantic object, OR
  - a plain dict  (e.g. from JSON deserialization)

Each grader returns a float STRICTLY in (0.01, 0.99) — never 0.0 or 1.0.
"""

from tasks import TASKS


def clamp(score: float) -> float:
    """Ensure score is strictly inside (0, 1). Never 0.0 or 1.0."""
    return max(0.01, min(round(float(score), 4), 0.99))


def _get(obs, key: str, default=0.0):
    """Fetch a field from a dict or an object attribute."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _goal_for_task(obs, task_id: str) -> float:
    """
    Resolve the goal position.
    Priority:
      1. obs.goal  (set in AdaptiveDrivingObservation)
      2. obs.position + obs.distance_to_goal
      3. TASKS[task_id]["goal"]  (fallback from task config)
    """
    # Try obs.goal directly
    goal = _get(obs, "goal", None)
    if goal is not None and float(goal) > 0:
        return float(goal)

    # Try position + distance_to_goal
    position = float(_get(obs, "position", 0.0))
    dtg      = _get(obs, "distance_to_goal", None)
    if dtg is not None:
        return position + float(dtg)

    # Fall back to task config
    return float(TASKS.get(task_id, {}).get("goal", 50))


# ── Task 1: Easy ──────────────────────────────────────────────

def grade_easy(obs) -> float:
    """
    Score = progress toward goal (position / goal).
    Reaching goal → 0.99.
    """
    position = float(_get(obs, "position", 0.0))
    goal     = _goal_for_task(obs, "easy")

    if goal <= 0:
        return 0.01

    if position >= goal:
        return 0.99

    progress = position / goal
    return clamp(progress)


# ── Task 2: Medium ────────────────────────────────────────────

def grade_medium(obs) -> float:
    """
    Score = progress − safety penalties.
    Penalises low visibility (< 0.6) and high speed (> 10).
    """
    position   = float(_get(obs, "position",   0.0))
    speed      = float(_get(obs, "speed",      0.0))
    visibility = float(_get(obs, "visibility", 1.0))
    goal       = _goal_for_task(obs, "medium")

    if goal <= 0:
        return 0.01

    progress = position / goal

    penalty = 0.0
    if visibility < 0.6:
        penalty += 0.2
    if speed > 10:
        penalty += 0.2

    if position >= goal:
        # Reached goal — still apply safety penalty
        return clamp(0.99 - penalty)

    return clamp(progress - penalty)


# ── Task 3: Hard ──────────────────────────────────────────────

def grade_hard(obs) -> float:
    """
    Weighted score: 70% progress + 30% battery efficiency.
    Completing goal with remaining battery → bonus up to 0.99.
    """
    position = float(_get(obs, "position", 0.0))
    battery  = float(_get(obs, "battery",  0.0))
    goal     = _goal_for_task(obs, "hard")

    if goal <= 0:
        return 0.01

    progress       = position / goal
    battery_factor = battery / 100.0

    if position >= goal and battery > 0:
        # Scale reward by remaining battery: 0.85 – 0.99
        return clamp(0.85 + battery_factor * 0.14)

    score = progress * 0.7 + battery_factor * 0.3
    return clamp(score)


# ── Registry ─────────────────────────────────────────────────

GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}


def grade(task_id: str, obs) -> float:
    """
    Main entry point: grade(task_id, obs) → float in (0.01, 0.99).
    Used by openenv validator and inference.py.
    """
    fn = GRADERS.get(task_id)
    if fn is None:
        raise ValueError(
            f"Unknown task_id: {task_id!r}. Valid: {list(GRADERS)}"
        )
    return fn(obs)