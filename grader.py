"""
Graders for Adaptive Driving Environment.

Implements the OpenEnv Rubric interface:
    rubric = EasyDriveRubric()
    score = rubric(action, observation)   # calls forward()

Each forward() returns a float STRICTLY in (0.01, 0.99).
"""

# ── Try to use OpenEnv Rubric base class, fall back to plain ABC ──────────────
try:
    from openenv.core.rubrics import Rubric as _Base
except ImportError:
    from abc import ABC, abstractmethod
    class _Base(ABC):
        @abstractmethod
        def forward(self, action, observation) -> float:
            pass
        def __call__(self, action, observation) -> float:
            return self.forward(action, observation)


# ── Task goal positions ───────────────────────────────────────────────────────
_TASK_GOALS = {
    "easy":   50.0,
    "medium": 70.0,
    "hard":  100.0,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def clamp(score: float) -> float:
    """Return score strictly inside (0.01, 0.99)."""
    return max(0.01, min(round(float(score), 4), 0.99))


def _get(obs, key: str, default=0.0):
    """Fetch a field from a dict or object attribute."""
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _resolve_goal(obs, task_id: str) -> float:
    goal = _get(obs, "goal", None)
    if goal is not None:
        try:
            g = float(goal)
            if g > 0:
                return g
        except (TypeError, ValueError):
            pass
    position = float(_get(obs, "position", 0.0))
    dtg = _get(obs, "distance_to_goal", None)
    if dtg is not None:
        try:
            return position + float(dtg)
        except (TypeError, ValueError):
            pass
    return _TASK_GOALS.get(task_id, 50.0)


# ── Rubric classes (OpenEnv Rubric interface) ─────────────────────────────────

class EasyDriveRubric(_Base):
    """Score = position / goal. Reaching goal → 0.99."""

    def forward(self, action, observation) -> float:
        position = float(_get(observation, "position", 0.0))
        goal     = _resolve_goal(observation, "easy")
        if position >= goal:
            return 0.99
        return clamp(position / goal if goal > 0 else 0.01)


class MediumDriveRubric(_Base):
    """Score = progress − safety penalties (visibility, speed)."""

    def forward(self, action, observation) -> float:
        position   = float(_get(observation, "position",   0.0))
        speed      = float(_get(observation, "speed",      0.0))
        visibility = float(_get(observation, "visibility", 1.0))
        goal       = _resolve_goal(observation, "medium")

        penalty = 0.0
        if visibility < 0.6:
            penalty += 0.2
        if speed > 10:
            penalty += 0.2

        if position >= goal:
            return clamp(0.99 - penalty)

        progress = position / goal if goal > 0 else 0.0
        return clamp(progress - penalty)


class HardDriveRubric(_Base):
    """Score = 70% progress + 30% battery efficiency."""

    def forward(self, action, observation) -> float:
        position = float(_get(observation, "position", 0.0))
        battery  = float(_get(observation, "battery",  0.0))
        goal     = _resolve_goal(observation, "hard")

        battery_factor = max(0.0, min(battery / 100.0, 1.0))

        if position >= goal and battery > 0:
            return clamp(0.85 + battery_factor * 0.14)

        progress = position / goal if goal > 0 else 0.0
        return clamp(progress * 0.7 + battery_factor * 0.3)


# ── Plain function aliases (for openenv.yaml grader: field) ──────────────────

_easy_rubric   = EasyDriveRubric()
_medium_rubric = MediumDriveRubric()
_hard_rubric   = HardDriveRubric()


def grade_easy(observation) -> float:
    return _easy_rubric.forward(None, observation)


def grade_medium(observation) -> float:
    return _medium_rubric.forward(None, observation)


def grade_hard(observation) -> float:
    return _hard_rubric.forward(None, observation)


# ── Registry ──────────────────────────────────────────────────────────────────

GRADERS = {
    "easy":   grade_easy,
    "medium": grade_medium,
    "hard":   grade_hard,
}

RUBRICS = {
    "easy":   _easy_rubric,
    "medium": _medium_rubric,
    "hard":   _hard_rubric,
}


def grade(task_id: str, observation) -> float:
    fn = GRADERS.get(task_id)
    if fn is None:
        raise ValueError(f"Unknown task_id: {task_id!r}. Valid: {list(GRADERS)}")
    return fn(observation)