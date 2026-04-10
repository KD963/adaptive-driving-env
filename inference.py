import os

from models import AdaptiveDrivingAction, AdaptiveDrivingObservation
from server.adaptive_driving_env_environment import AdaptiveDrivingEnvironment

# ---------------- ENV ----------------
MODEL_NAME = os.getenv("MODEL_NAME", "rule-based")


# ---------------- POLICY ----------------
def get_action(position: float, goal: float, speed: float, battery: float) -> str:
    distance = goal - position
    return "brake" if abs(distance) < 5 else "accelerate"


# ---------------- MAIN ----------------
def main():
    success = False
    rewards = []
    steps_taken = 0
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
                done = bool(obs.done)

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

    except Exception as e:
        last_error = str(e)
        print(f"[START] task=unknown env=adaptive-driving model={MODEL_NAME}")

    finally:
        reward_str = ",".join(f"{r:.2f}" for r in rewards)

        # ✅ END (MANDATORY)
        print(
            f"[END] success={str(success).lower()} "
            f"steps={steps_taken} "
            f"rewards={reward_str}"
        )


if __name__ == "__main__":
    main()