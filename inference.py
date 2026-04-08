import asyncio
from dotenv import load_dotenv
import os
from openai import OpenAI
from models import AdaptiveDrivingAction
from server.adaptive_driving_env_environment import AdaptiveDrivingEnvironment

# Load environment variables
load_dotenv()

# OpenAI client
client = OpenAI(
    base_url=os.getenv("API_BASE_URL"),
    api_key=os.getenv("HF_TOKEN")
)
MODEL_NAME = os.getenv("MODEL_NAME")

# System prompt for the LLM
SYSTEM_PROMPT = """
You control a car.

Goal: reach target position.

Actions:
accelerate OR brake

Return ONLY one word: accelerate or brake
"""

# Function to get LLM action
def get_action(position, goal):
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


async def main():
    success = False
    steps_taken = 0
    score = 0
    rewards = []

    env = AdaptiveDrivingEnvironment()
    obs = env.reset()
    
    try:
        for step in range(50):  # run up to 50 steps
            action_str = get_action(obs.position, obs.distance_to_goal + obs.position)
            action = AdaptiveDrivingAction(move=action_str)
            obs = env.step(action)

            rewards.append(obs.reward)
            steps_taken += 1

            print(f"Step {steps_taken}: Action={action.move}, Position={obs.position}, Reward={obs.reward}")

            if getattr(obs, "done", False):
                success = True
                break

        score = obs.position  # you can define scoring differently
        print(f"Simulation ended. Success={success}, Steps={steps_taken}, Score={score}")

    except Exception as e:
        print("Simulation failed:", e)

    finally:
        # logging at the end
        print("Final Summary:")
        print(f"Success: {success}")
        print(f"Steps Taken: {steps_taken}")
        print(f"Score: {score}")
        print(f"Rewards: {rewards}")


if __name__ == "__main__":
    asyncio.run(main())