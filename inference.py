import os
from openai import OpenAI

from models import AdaptiveDrivingAction, AdaptiveDrivingObservation
from server.adaptive_driving_env_environment import AdaptiveDrivingEnvironment

# ---------------- ENV VARIABLES ----------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN     = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ---------------- OPENAI CLIENT ----------------
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN
)

# ---------------- PROMPT ----------------
SYSTEM_PROMPT = """
You control a car.

Goal: reach the target position.

Actions:
- accelerate → increase speed
- brake → decrease speed

Return ONLY one word: accelerate or brake
"""

# ---------------- POLICY ----------------
def get_action(position, goal, speed, battery):
    distance = goal - position

    # fallback (VERY IMPORTANT for stability)
    def fallback():
        return "brake" if abs(distance) < 5 else "accelerate"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Position={position}, Goal={goal}, "
                    f"Distance={distance}, Speed={speed}, Battery={battery}"
                )}
            ],
            temperature=0.3,
            max_tokens=5,
        )

        text = response.choices[0].message.content.lower().strip()

        if "brake" in text:
            return "brake"
        return "accelerate"

    except Exception:
        # NEVER fail → always fallback
        return fallback()


# ---------------- MAIN ----------------
def main():
    success = False
    rewards = []
    steps = 0
    last_error = None

    try:
        env = AdaptiveDrivingEnvironment()
        obs: AdaptiveDrivingObservation = env.reset()

        task_name = obs.metadata.get("task", "unknown") if obs.metadata else "unknown"

        # ✅ START
        print(f"[START] task={task_name} env=adaptive-driving model={MODEL_NAME}")

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

                reward = float(obs.reward)
                done   = bool(obs.done)

                rewards.append(reward)
                steps += 1

                print(
                    f"[STEP] step={steps} "
                    f"action={action_str} "
                    f"reward={reward:.2f} "
                    f"done={str(done).lower()} "
                    f"error=null"
                )

                if done:
                    success = True
                    break

            except Exception as step_error:
                last_error = str(step_error)

                print(
                    f"[STEP] step={step_num} "
                    f"action=error "
                    f"reward=0.00 "
                    f"done=true "
                    f"error={last_error}"
                )
                break

    except Exception as e:
        last_error = str(e)
        print(f"[START] task=unknown env=adaptive-driving model={MODEL_NAME}")

    finally:
        reward_str = ",".join(f"{r:.2f}" for r in rewards)

        # ✅ END (MANDATORY)
        print(
            f"[END] success={str(success).lower()} "
            f"steps={steps} "
            f"rewards={reward_str}"
        )


if __name__ == "__main__":
    main()