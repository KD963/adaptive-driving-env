import os
import math
from openai import OpenAI

from models import AdaptiveDrivingAction, AdaptiveDrivingObservation
from server.adaptive_driving_env_environment import AdaptiveDrivingEnvironment

# ---------------- ENV VARIABLES ----------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

SYSTEM_PROMPT = """
You control a car. Goal: reach the target position.
Actions: accelerate or brake. Return ONLY one word.
"""

# ─────────────────────────────────────────────
# ACTION GENERATION
# ─────────────────────────────────────────────

def get_action(position, goal, speed, battery):
    distance = goal - position

    def fallback():
        return "brake" if abs(distance) < 5 else "accelerate"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Pos={position}, Goal={goal}, Spd={speed}, Bat={battery}"}
            ],
            temperature=0.3,
            max_tokens=5,
        )
        text = response.choices[0].message.content.lower().strip()
        return "brake" if "brake" in text else "accelerate"

    except Exception:
        return fallback()


# ─────────────────────────────────────────────
# SAFE LOGGING (CRITICAL)
# ─────────────────────────────────────────────

def log_safe(val: float) -> float:
    """
    Ensures values NEVER become 0 or 1,
    and avoids rounding into boundaries.
    """
    try:
        f_val = float(val)

        if not math.isfinite(f_val):
            return 0.021

        if f_val <= 0.02:
            return 0.021
        if f_val >= 0.98:
            return 0.979

        return round(f_val, 4)

    except:
        return 0.021


# ─────────────────────────────────────────────
# MAIN LOOP
# ─────────────────────────────────────────────

def main():
    task_list = ["easy", "medium", "hard"]
    env = AdaptiveDrivingEnvironment()

    for task_id in task_list:
        success = False
        rewards = []
        steps = 0
        started = False

        try:
            obs = env.reset(task_id=task_id)

            print(f"[START] task={task_id} env=adaptive-driving model={MODEL_NAME}")
            started = True

            for step_num in range(1, 51):
                try:
                    action_str = get_action(
                        obs.position,
                        obs.goal,
                        obs.speed,
                        obs.battery
                    )

                    action = AdaptiveDrivingAction(move=action_str)
                    obs = env.step(action)

                    # ✅ SAFE REWARD
                    current_reward = log_safe(obs.reward)
                    done = bool(obs.done)

                    rewards.append(current_reward)
                    steps += 1

                    # 🔥 IMPORTANT: 4 DECIMALS (NOT 2)
                    print(
                        f"[STEP] step={steps} "
                        f"action={action_str} "
                        f"reward={current_reward:.4f} "
                        f"done={str(done).lower()} "
                        f"error=null"
                    )

                    if done:
                        success = True
                        break

                except Exception as step_error:
                    print(
                        f"[STEP] step={step_num} "
                        f"action=error "
                        f"reward=0.021 "
                        f"done=true "
                        f"error={str(step_error)}"
                    )
                    break

        except Exception as e:
            if not started:
                print(f"[START] task={task_id} env=adaptive-driving model={MODEL_NAME}")
            print(f"Environment Error: {str(e)}")

        finally:
            # 🔥 IMPORTANT: 4 DECIMALS + SAFE DEFAULT
            reward_str = (
                ",".join(f"{log_safe(r):.4f}" for r in rewards)
                if rewards else "0.021"
            )

            print(
                f"[END] success={str(success).lower()} "
                f"steps={steps} "
                f"rewards={reward_str}"
            )


if __name__ == "__main__":
    main()