import math

def clamp(score: float) -> float:
    """
    THE VALIDATOR SHIELD:
    Forces the score to be strictly > 0 and < 1.
    Specifically avoids 0.0 and 1.0 which trigger the 'Out of Range' error.
    """
    try:
        val = float(score)
        if not math.isfinite(val): 
            return 0.01
        
        # Explicit floor at 0.01 and ceiling at 0.99
        # This prevents rounding errors (like 0.9999 -> 1.0)
        if val <= 0.01:
            return 0.01
        if val >= 0.99:
            return 0.99
            
        return round(val, 4)
    except (ValueError, TypeError):
        return 0.01
    

def _get_val(obj, key, default=0.0):
    """Safe extraction for both dicts and objects with internal error handling."""
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    except Exception:
        return default

def grade_easy(obs):
    try:
        # Priority: Use environment reward if already calculated
        env_reward = _get_val(obs, "reward", None)
        if env_reward is not None:
            return float(env_reward)

        pos = float(_get_val(obs, "position", 0.0))
        goal = float(_get_val(obs, "goal", 50.0))
        progress = max(0.0, min(pos / goal, 1.0)) if goal > 0 else 0.0
        return 0.05 + (0.9 * progress)
    except Exception:
        return 0.05

def grade_medium(obs):
    try:
        env_reward = _get_val(obs, "reward", None)
        if env_reward is not None:
            return float(env_reward)

        pos = float(_get_val(obs, "position", 0.0))
        goal = float(_get_val(obs, "goal", 75.0))
        vis = float(_get_val(obs, "visibility", 1.0))
        spd = float(_get_val(obs, "speed", 0.0))
        
        progress = max(0.0, min(pos / goal, 1.0)) if goal > 0 else 0.0
        penalty = 0.0
        if vis < 0.6: penalty += 0.15
        if spd > 10:  penalty += 0.15
        
        return max(0.05, (0.05 + 0.9 * progress) - penalty)
    except Exception:
        return 0.05

def grade_hard(obs):
    try:
        env_reward = _get_val(obs, "reward", None)
        if env_reward is not None:
            return float(env_reward)

        pos = float(_get_val(obs, "position", 0.0))
        goal = float(_get_val(obs, "goal", 110.0))
        bat = float(_get_val(obs, "battery", 100.0))
        
        progress = max(0.0, min(pos / goal, 1.0)) if goal > 0 else 0.0
        bat_factor = max(0.0, min(bat / 100.0, 1.0))
        
        # 70% progress, 30% battery efficiency
        return 0.05 + (0.65 * progress) + (0.25 * bat_factor)
    except Exception:
        return 0.05

GRADERS = {
    "easy": grade_easy,
    "medium": grade_medium,
    "hard": grade_hard,
}

def grade(task_id: str, *args, **kwargs):
    """
    Ultra-robust entrypoint.
    Returns a safe float if the task is unknown or data is missing.
    """
    try:
        tid = str(task_id).lower()
        fn = GRADERS.get(tid)
        
        if not fn:
            return 0.01

        # Extract observation positional or keyword args
        obs = args[0] if len(args) > 0 else (kwargs.get('observation') or kwargs.get('obs'))

        if obs is None:
            return 0.01

        raw_score = fn(obs)
        return clamp(raw_score)
        
    except Exception:
        # Ultimate fallback to ensure a valid score is ALWAYS returned
        return 0.02