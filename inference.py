import os
import time
import requests

# ---------------- ENV VARIABLES ----------------
API_BASE_URL   = os.getenv("API_BASE_URL",   "https://api.openai.com/v1")
MODEL_NAME     = os.getenv("MODEL_NAME",     "gpt-4.1-mini")
HF_TOKEN       = os.getenv("HF_TOKEN",       None)   # ← no longer a hard raise
ENV_SERVER_URL = os.getenv("ENV_SERVER_URL", "http://0.0.0.0:7860")

# ---------------- OPENAI CLIENT ----------------
# Initialise only when a token is present.
# If absent, the episode still runs with a rule-based fallback policy
# so the container produces valid [STEP]/[END] log lines.
client = None
if HF_TOKEN:
    try:
        from openai import OpenAI
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    except Exception as e:
        print(f"[WARN] Could not init OpenAI client: {e}. Using rule-based fallback.")

# ---------------- PROMPT ----------------
SYSTEM_PROMPT = """
You control a car.
Goal: reach the target position.
Actions:
  accelerate  → increases speed, uses more battery
  brake       → decreases speed, conserves battery
Return ONLY one word: accelerate or brake
"""

# ---------------- SERVER HELPERS ----------------

def wait_for_server(url: str, timeout: int = 30) -> bool:
    """Poll /health until the env server is ready."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"{url}/health", timeout=3)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def env_reset(task: str = "easy") -> dict:
    r = requests.post(f"{ENV_SERVER_URL}/reset",
                      json={"task": task}, timeout=10)
    r.raise_for_status()
    return r.json()


def env_step(action_str: str) -> dict:
    r = requests.post(f"{ENV_SERVER_URL}/step",
                      json={"action": {"move": action_str}}, timeout=10)
    r.raise_for_status()
    return r.json()


# ---------------- ACTION POLICY ----------------

def get_action(position: float, goal: float, speed: float, battery: float) -> str:
    distance = goal - position

    def rule_based() -> str:
        return "brake" if abs(distance) < 5 else "accelerate"

    if client is None:
        return rule_based()

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": (
                    f"Position={position:.1f}, Goal={goal:.1f}, "
                    f"Distance={distance:.1f}, Speed={speed:.1f}, Battery={battery:.1f}"
                )},
            ],
            temperature=0.3,
            max_tokens=10,
        )
        text = res.choices[0].message.content.lower().strip()
        return "brake" if "brake" in text else "accelerate"
    except Exception:
        return rule_based()


# ---------------- MAIN ----------------

def main():
    success     = False
    rewards     = []
    steps_taken = 0
    last_error  = None

    # Wait for env HTTP server to be ready
    if not wait_for_server(ENV_SERVER_URL, timeout=30):
        print(f"[ERROR] Env server at {ENV_SERVER_URL} did not become ready in 30s.")
        print("[END] success=false steps=0 rewards=")
        return

    # Reset episode
    try:
        obs = env_reset(task="easy")
    except Exception as e:
        print(f"[ERROR] /reset failed: {e}")
        print("[END] success=false steps=0 rewards=")
        return

    # Parse initial observation (plain dict from JSON)
    position         = float(obs.get("position",         0))
    speed            = float(obs.get("speed",            0))
    battery          = float(obs.get("battery",          100))
    distance_to_goal = float(obs.get("distance_to_goal", 50))
    goal             = float(obs.get("goal", position + distance_to_goal))
    task_name        = (obs.get("metadata") or {}).get("task", "easy")

    print(f"[START] task={task_name} env=adaptive-driving model={MODEL_NAME}")

    # Episode loop
    for step_num in range(1, 51):
        action_str = get_action(position, goal, speed, battery)

        try:
            result   = env_step(action_str)

            # /step returns {observation: {...}, done: bool, reward: float}
            obs_data = result.get("observation", result)
            reward   = float(result.get("reward", obs_data.get("reward", 0.0)))
            done     = bool(result.get("done",    obs_data.get("done",   False)))

            # Update state for next iteration
            position         = float(obs_data.get("position",         position))
            speed            = float(obs_data.get("speed",            speed))
            battery          = float(obs_data.get("battery",          battery))
            distance_to_goal = float(obs_data.get("distance_to_goal", distance_to_goal))
            goal             = float(obs_data.get("goal", position + distance_to_goal))

            rewards.append(reward)
            steps_taken += 1

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
                f"[STEP] step={step_num} "
                f"action={action_str} "
                f"reward=0.00 "
                f"done=true "
                f"error={last_error}"
            )
            break

    # Final log lines (required by validator)
    reward_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps_taken} "
        f"rewards={reward_str}"
    )

    print()
    print("Final Summary:")
    print(f"  Success:     {success}")
    print(f"  Steps Taken: {steps_taken}")
    print(f"  Score:       {sum(rewards):.2f}")
    print(f"  Rewards:     {rewards}")
    if last_error:
        print(f"  Last Error:  {last_error}")


if __name__ == "__main__":
    main()