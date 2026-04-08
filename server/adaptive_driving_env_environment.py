import random
from typing import Optional


from ..models import AdaptiveDrivingAction, AdaptiveDrivingObservation
from tasks import TASKS


class AdaptiveDrivingEnvironment:
    """
    Real-world Adaptive Driving Environment

    Simulates an electric car driving under:
    - Weather conditions (rain, fog, heat)
    - Road slope
    - Battery constraints

    Goal: Reach destination efficiently and safely.
    """

    def __init__(self):
        self.step_count = 0
        self.episode_id = str(random.randint(1000, 9999))
        self.current_task = "easy"
        self.reset(self.current_task)

    # ---------------- RESET ----------------
    # def reset(self, task_name: str = "easy", **kwargs):
    def reset(self, task_name: str = None):
        if task_name is None:
            task_name = random.choice(["easy", "medium", "hard"])
        task = TASKS.get(task_name, TASKS["easy"])
        self.current_task = task_name

        # Core state
        self.position = 0
        self.speed = 0
        self.battery = task["battery"]
        self.goal = task["goal"]

        # Environment conditions
        self.weather = task["weather"]
        self.slope = task["slope"]

        # Derived conditions
        self.visibility = 1.0
        self.traction = 1.0

        # Step tracking
        self.step_count = 0
        self.episode_id = str(random.randint(1000, 9999))

        return self._get_obs(reward=0.0, done=False)

    # ---------------- STEP ----------------
    def step(self, action: AdaptiveDrivingAction):
        self.step_count += 1

        prev_distance = self.goal - self.position

        # -------- APPLY ACTION --------
        if action.move == "accelerate":
            self.speed += 2
            self.battery -= 1.5
        elif action.move == "brake":
            self.speed -= 2
            self.battery -= 0.5

        # Prevent negative speed
        self.speed = max(self.speed, 0)

        # -------- WEATHER EFFECTS --------
        if self.weather == "rain":
            self.traction = 0.7
            self.speed *= 0.9

        elif self.weather == "fog":
            self.visibility = 0.5

        elif self.weather == "heat":
            self.battery -= 0.3

        # -------- SLOPE EFFECT --------
        self.speed -= self.slope * 0.1

        # -------- UPDATE POSITION --------
        self.position += self.speed

        # Clamp
        self.position = max(self.position, 0)

        new_distance = self.goal - self.position

        # -------- REWARD FUNCTION --------
        reward = (prev_distance - new_distance) * 0.5  # progress reward

        # Penalties
        if self.speed > 12:
            reward -= 3  # unsafe speed

        if self.visibility < 0.6:
            reward -= 1.5  # poor visibility risk

        if self.battery < 10:
            reward -= 2  # low battery anxiety

        # -------- TERMINATION --------
        done = False

        if self.battery <= 0:
            reward -= 10
            done = True

        elif self.position >= self.goal:
            reward += 50
            done = True

        elif self.step_count >= 50:
            done = True

        return self._get_obs(reward=reward, done=done)

    # ---------------- OBSERVATION ----------------
    def _get_obs(self, reward: float, done: bool):
        return AdaptiveDrivingObservation(
            position=round(self.position, 2),
            speed=round(self.speed, 2),
            battery=round(self.battery, 2),
            slope=round(self.slope, 2),
            weather=self.weather,
            visibility=round(self.visibility, 2),
            traction=round(self.traction, 2),
            distance_to_goal=round(self.goal - self.position, 2),
            reward=round(reward, 2),
            done=done,
            metadata={
                "step": self.step_count,
                "task": self.current_task,
            },
        )

    # # ---------------- STATE ----------------
    # @property
    # def state(self) -> State:
    #     return State(
    #         episode_id=self.episode_id,
    #         step_count=self.step_count,
    #     )