import math

def clamp(score: float) -> float:
    """Strictly enforces (0.0, 1.0) range with 3-decimal precision."""
    try:
        val = float(score)
        if not math.isfinite(val):
            return 0.01
        # Stay strictly away from 0.0 and 1.0
        return max(0.01, min(round(val, 3), 0.99))
    except (ValueError, TypeError):
        return 0.01

def _get_val(obj, key, default=0.0):
    """Safe extraction for both dicts and objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)

def grade_easy(obs):
    pos = float(_get_val(obs, "position", 0.0))
    goal = float(_get_val(obs, "goal", 50.0))
    progress = max(0.0, min(pos / goal, 1.0)) if goal > 0 else 0.0
    return 0.05 + (0.9 * progress)

def grade_medium(obs):
    pos = float(_get_val(obs, "position", 0.0))
    goal = float(_get_val(obs, "goal", 70.0))
    vis = float(_get_val(obs, "visibility", 1.0))
    spd = float(_get_val(obs, "speed", 0.0))
    
    progress = max(0.0, min(pos / goal, 1.0)) if goal > 0 else 0.0
    penalty = 0.0
    if vis < 0.6: penalty += 0.15
    if spd > 10:  penalty += 0.15
    
    return max(0.05, (0.05 + 0.9 * progress) - penalty)

def grade_hard(obs):
    pos = float(_get_val(obs, "position", 0.0))
    goal = float(_get_val(obs, "goal", 100.0))
    bat = float(_get_val(obs, "battery", 0.0))
    
    progress = max(0.0, min(pos / goal, 1.0)) if goal > 0 else 0.0
    bat_factor = max(0.0, min(bat / 100.0, 1.0))
    
    # 70% progress, 30% battery efficiency
    return 0.05 + (0.65 * progress) + (0.25 * bat_factor)

GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}

def grade(task_id: str, observation=None, **kwargs):
    """
    OpenEnv Entrypoint. 
    Handles 'obs' passed as positional or 'observation' as keyword.
    """
    # Standardize the task_id
    tid = str(task_id).lower()
    fn = GRADERS.get(tid)
    
    # Extract observation from various possible entry formats
    obs = observation if observation is not None else kwargs.get('obs')
    
    if not fn or obs is None:
        return 0.01

    try:
        raw_score = fn(obs)
        return clamp(raw_score)
    except Exception:
        return 0.02