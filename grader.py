def clamp(score: float) -> float:
    """
    Ensure score is strictly between (0,1)
    """
    return max(0.01, min(score, 0.99))


def grade_easy(env):
    """
    Score based on reaching goal and efficiency
    """
    progress = env.position / env.goal if env.goal else 0

    # If reached goal → near perfect but not 1
    if env.position >= env.goal:
        return 0.99

    return clamp(round(progress, 2))


def grade_medium(env):
    """
    Penalize unsafe driving (speed + visibility issues)
    """
    progress = env.position / env.goal if env.goal else 0

    safety_penalty = 0
    if env.visibility < 0.6:
        safety_penalty += 0.2
    if env.speed > 10:
        safety_penalty += 0.2

    score = progress - safety_penalty
    return clamp(round(score, 2))


def grade_hard(env):
    """
    Focus on battery efficiency + goal completion
    """
    progress = env.position / env.goal if env.goal else 0
    battery_factor = env.battery / 100

    score = progress * 0.7 + battery_factor * 0.3

    # Even if perfect → don't return 1
    if env.position >= env.goal and env.battery > 0:
        return 0.99

    return clamp(round(score, 2))