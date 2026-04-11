import random
from models import AdaptiveDrivingAction, AdaptiveDrivingObservation
from tasks import TASKS
from grader import grade


class AdaptiveDrivingEnvironment:

    def __init__(self):
        self._task_id = "easy"
        self._reset_state()

    def _reset_state(self):
        self._position = 0.0
        self._speed = 0.0
        self._battery = 100.0
        self._slope = 0.0
        self._weather = "clear"
        self._visibility = 1.0
        self._traction = 1.0
        self._goal = 50.0
        self._step = 0
        self._done = False

    # ─────────────────────────────────────────

    def reset(self, task_id: str = None):
        if task_id not in TASKS:
            task_id = random.choice(list(TASKS.keys()))

        cfg = TASKS[task_id]

        self._task_id = task_id
        self._reset_state()

        self._battery = float(cfg["battery"])
        self._slope = float(cfg["slope"])
        self._weather = cfg["weather"]
        self._goal = float(cfg["goal"])

        self._visibility = self._compute_visibility()
        self._traction = self._compute_traction()

        return self._make_obs(0.05)

    # ─────────────────────────────────────────

    def step(self, action: AdaptiveDrivingAction):
        if self._done:
            return self._make_obs(0.05)

        self._step += 1
        move = action.move.lower().strip()

        if move == "accelerate":
            accel = max(0.5, 2.0 - self._slope * 0.3) * self._traction
            self._speed = min(self._speed + accel, 20.0)
            self._battery = max(0.0, self._battery - 2.0)

        elif move == "brake":
            self._speed = max(0.0, self._speed - 3.0)
            self._battery = max(0.0, self._battery - 0.5)

        self._position += self._speed
        self._position = min(self._position, self._goal)

        reached = self._position >= self._goal
        out_of_battery = self._battery <= 0
        timed_out = self._step >= 50

        self._done = reached or out_of_battery or timed_out

        obs_snapshot = {
            "position": self._position,
            "speed": self._speed,
            "battery": self._battery,
            "visibility": self._visibility,
            "goal": self._goal,
        }

        reward = grade(self._task_id, obs_snapshot)

        return self._make_obs(reward)

    # ─────────────────────────────────────────

    def _compute_visibility(self):
        return {"clear": 1.0, "rain": 0.5, "heat": 0.8}.get(self._weather, 1.0)

    def _compute_traction(self):
        return {"clear": 1.0, "rain": 0.6, "heat": 0.9}.get(self._weather, 1.0)

    def _make_obs(self, reward: float):
        safe_reward = max(0.02, min(round(float(reward), 4), 0.98))

        return AdaptiveDrivingObservation(
            position=self._position,
            speed=self._speed,
            battery=self._battery,
            slope=self._slope,
            weather=self._weather,
            visibility=self._visibility,
            traction=self._traction,
            distance_to_goal=max(0.0, self._goal - self._position),
            goal=self._goal,
            reward=safe_reward,
            done=self._done,
            metadata={"task": self._task_id, "step": self._step},
        )