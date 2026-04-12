"""
AdaptiveDrivingEnvironment

FINAL VERSION (Validator Safe)

- No Python grader logic
- Reward is progress-based (non-constant)
- Always strictly between (0, 1)
"""

import random
from models import AdaptiveDrivingAction, AdaptiveDrivingObservation
from tasks import TASKS

VALID_TASKS = list(TASKS.keys())


class AdaptiveDrivingEnvironment:

    def __init__(self):
        self._task_id    = "easy"
        self._position   = 0.0
        self._speed      = 0.0
        self._battery    = 100.0
        self._slope      = 0.0
        self._weather    = "clear"
        self._visibility = 1.0
        self._traction   = 1.0
        self._goal       = 50.0
        self._step       = 0
        self._done       = False

    # ─────────────────────────────────────────────

    def reset(self, task_id: str = None) -> AdaptiveDrivingObservation:
        if task_id in VALID_TASKS:
            self._task_id = task_id
        else:
            self._task_id = "easy"

        cfg = TASKS[self._task_id]

        self._position   = 0.0
        self._speed      = 0.0
        self._battery    = float(cfg["battery"])
        self._slope      = float(cfg["slope"])
        self._weather    = cfg["weather"]
        self._goal       = float(cfg["goal"])
        self._visibility = self._compute_visibility()
        self._traction   = self._compute_traction()
        self._step       = 0
        self._done       = False

        return self._make_obs()

    # ─────────────────────────────────────────────

    def step(self, action: AdaptiveDrivingAction) -> AdaptiveDrivingObservation:
        if self._done:
            return self._make_obs()

        self._step += 1
        move = action.move.lower().strip()

        if move == "accelerate":
            accel         = max(0.5, 2.0 - self._slope * 0.3) * self._traction
            self._speed   = min(self._speed + accel, 20.0)
            self._battery = max(0.0, self._battery - 2.0)

        elif move == "brake":
            self._speed   = max(0.0, self._speed - 3.0)
            self._battery = max(0.0, self._battery - 0.5)

        # Update position
        self._position = min(self._position + self._speed, self._goal)

        reached        = self._position >= self._goal
        out_of_battery = self._battery <= 0
        timed_out      = self._step >= 50

        self._done = reached or out_of_battery or timed_out

        return self._make_obs()

    # ─────────────────────────────────────────────

    def state(self) -> dict:
        return {
            "task_id":  self._task_id,
            "position": self._position,
            "speed":    self._speed,
            "battery":  self._battery,
            "goal":     self._goal,
            "step":     self._step,
            "done":     self._done,
        }

    # ─────────────────────────────────────────────

    def _compute_visibility(self) -> float:
        return {"clear": 1.0, "rain": 0.5, "heat": 0.8}.get(self._weather, 1.0)

    def _compute_traction(self) -> float:
        return {"clear": 1.0, "rain": 0.6, "heat": 0.9}.get(self._weather, 1.0)

    # ✅ KEY FIX: dynamic reward
    def _compute_reward(self) -> float:
        progress = self._position / max(self._goal, 1.0)

        # small penalty if wasting battery
        efficiency = self._battery / 100.0

        reward = 0.7 * progress + 0.3 * efficiency

        # clamp strictly between (0,1)
        return max(0.02, min(round(reward, 4), 0.98))

    def _make_obs(self) -> AdaptiveDrivingObservation:
        return AdaptiveDrivingObservation(
            position         = float(round(self._position, 4)),
            speed            = float(round(self._speed, 4)),
            battery          = float(round(self._battery, 4)),
            slope            = float(self._slope),
            weather          = str(self._weather),
            visibility       = float(self._visibility),
            traction         = float(self._traction),
            distance_to_goal = float(round(max(0.0, self._goal - self._position), 4)),
            goal             = float(self._goal if self._goal > 0 else 1.0),
            reward           = self._compute_reward(),  # ✅ dynamic reward
            done             = bool(self._done),
            metadata         = {"task": self._task_id, "step": self._step},
        )