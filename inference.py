import os
from openai import OpenAI
from models import AdaptiveDrivingAction
from server.adaptive_driving_env_environment import AdaptiveDrivingEnvironment

# ---------------- ENV VARIABLES ----------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

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

Goal: reach target position.

Actions:
accelerate OR brake

Return ONLY one word: accelerate or brake
"""

# ---------------- LLM ACTION ----------------
def get_action(position, goal):
    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Position={position}, Goal={goal}"}
            ],
            temperature=0.3,
            max_tokens=10
        )

        text = res.choices[0].message.content.lower().strip()

        if "brake" in text:
            return "brake"
        return "accelerate"

    except Exception:
        # fallback (important for evaluation stability)
        return "accelerate"


# ---------------- MAIN ----------------
def main():
    success = False
    rewards = []
    steps_taken = 0
    last_error = None

    env = AdaptiveDrivingEnvironment()
    obs = env.reset()

    task_name = obs.metadata.get("task", "unknown")

    # -------- START LINE --------
    print(f"[START] task={task_name} env=adaptive-driving model={MODEL_NAME}")

    try:
        for step in range(50):
            action_str = get_action(obs.position, obs.distance_to_goal + obs.position)

            try:
                action = AdaptiveDrivingAction(move=action_str)
                obs = env.step(action)

                reward = float(obs.reward)
                done = bool(obs.done)

                rewards.append(reward)
                steps_taken += 1

                # -------- STEP LINE --------
                print(
                    f"[STEP] step={steps_taken} "
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
                    f"[STEP] step={steps_taken+1} "
                    f"action={action_str} "
                    f"reward=0.00 "
                    f"done=true "
                    f"error={last_error}"
                )
                break

    except Exception as e:
        last_error = str(e)

    finally:
        # -------- END LINE --------
        reward_str = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END] success={str(success).lower()} "
            f"steps={steps_taken} "
            f"rewards={reward_str}"
        )


if __name__ == "__main__":
    main()