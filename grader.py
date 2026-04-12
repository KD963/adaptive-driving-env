import math

def clamp(score: float) -> float:
    try:
        val = float(score)
        if not math.isfinite(val):
            return 0.01
        # Use a slightly higher floor to avoid rounding to 0.0
        if val <= 0.01: return 0.01
        if val >= 0.99: return 0.99
        return round(val, 4)
    except:
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

def grade(task_id: str, *args, **kwargs):
    """
    Ultra-robust entrypoint to ensure 'obs' is captured 
    regardless of how the platform passes it.
    """
    tid = str(task_id).lower()
    fn = GRADERS.get(tid)
    
    if not fn:
        return 0.01

    # Look for observation in positional args first, then keywords
    obs = None
    if len(args) > 0:
        obs = args[0]
    else:
        obs = kwargs.get('observation') or kwargs.get('obs')

    if obs is None:
        # If we still don't have it, the grader can't work.
        # But we return a valid float to avoid 'Out of Range' errors.
        return 0.01

    try:
        raw_score = fn(obs)
        return clamp(raw_score)
    except Exception:
        return 0.02