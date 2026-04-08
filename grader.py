def grade_easy(env):
    """
    Score based on reaching goal and efficiency
    """
    if env.position >= env.goal:
        return 1.0

    progress = env.position / env.goal
    return round(progress, 2)


def grade_medium(env):
    """
    Penalize unsafe driving (speed + visibility issues)
    """
    if env.position >= env.goal:
        return 1.0

    progress = env.position / env.goal

    safety_penalty = 0
    if env.visibility < 0.6:
        safety_penalty += 0.2
    if env.speed > 10:
        safety_penalty += 0.2

    score = max(progress - safety_penalty, 0)
    return round(score, 2)


def grade_hard(env):
    """
    Focus on battery efficiency + goal completion
    """
    if env.position >= env.goal and env.battery > 0:
        return 1.0

    progress = env.position / env.goal
    battery_factor = env.battery / 100

    score = progress * 0.7 + battery_factor * 0.3
    return round(min(score, 1.0), 2)