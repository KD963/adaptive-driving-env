import math

def clamp(score: float) -> float:
    """
    STRICT VALIDATOR SHIELD:
    Ensures the score is NEVER 0.0 or 1.0.
    Using 0.02 and 0.98 to avoid floating-point precision rounding issues.
    """
    try:
        val = float(score)
        if not math.isfinite(val): 
            return 0.02
        
        # Explicit floor at 0.02 and ceiling at 0.98
        # This is the safest range to avoid 'strictly between 0 and 1' failures.
        if val <= 0.02:
            return 0.02
        if val >= 0.98:
            return 0.98
            
        return round(val, 4)
    except (ValueError, TypeError):
        return 0.02
    

def _get_val(obj, key, default=0.0):
    """Robust attribute extraction."""
    try:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
    except Exception:
        return default

def grade_easy(obs):
    try:
        # Check for pre-calculated reward from environment first
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
        
        progress = max(0.0, min(pos / goal, 1.0)) if goal > 0 else 0.0
        return 0.05 + (0.9 * progress)
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
    Ultimate entrypoint shield.
    Always returns a float strictly between 0 and 1.
    """
    try:
        tid = str(task_id).lower()
        fn = GRADERS.get(tid)
        
        if not fn:
            return 0.02

        # Capture observation from any possible argument position
        obs = args[0] if len(args) > 0 else (kwargs.get('observation') or kwargs.get('obs'))

        if obs is None:
            return 0.02

        raw_score = fn(obs)
        return clamp(raw_score)
        
    except Exception:
        # Ultimate fallback
        return 0.05