"""
AdaptiveDrivingEnvironment

Key fix: reset(task_id) ALWAYS uses the given task_id.
The validator calls reset("easy"), reset("medium"), reset("hard") separately
and checks that each returns a different, correctly-configured episode.
Falling back to random breaks task isolation.
"""

import random
from models import AdaptiveDrivingAction, AdaptiveDrivingObservation
from tasks import TASKS
from grader import grade

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

    def reset(self, task_id: str = None) -> AdaptiveDrivingObservation:
        # ── KEY FIX: never fall through to random if a valid task is given ──
        if task_id in VALID_TASKS:
            self._task_id = task_id
        elif task_id is None:
            self._task_id = "easy"          # default, not random
        else:
            # Unknown task → default to easy (don't silently randomise)
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

        # Initial reward: use grader on the start state (always > 0)
        reward = self._compute_reward()
        return self._make_obs(reward)

    def step(self, action: AdaptiveDrivingAction) -> AdaptiveDrivingObservation:
        if self._done:
            return self._make_obs(self._compute_reward())

        self._step += 1
        move = action.move.lower().strip()

        if move == "accelerate":
            accel         = max(0.5, 2.0 - self._slope * 0.3) * self._traction
            self._speed   = min(self._speed + accel, 20.0)
            self._battery = max(0.0, self._battery - 2.0)
        elif move == "brake":
            self._speed   = max(0.0, self._speed - 3.0)
            self._battery = max(0.0, self._battery - 0.5)

        self._position = min(self._position + self._speed, self._goal)

        reached          = self._position >= self._goal
        out_of_battery   = self._battery <= 0
        timed_out        = self._step >= 50
        self._done       = reached or out_of_battery or timed_out

        reward = self._compute_reward()
        return self._make_obs(reward)

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

    # ── helpers ───────────────────────────────────────────────

    def _compute_reward(self) -> float:
        obs_snapshot = {
            "position":   self._position,
            "speed":      self._speed,
            "battery":    self._battery,
            "visibility": self._visibility,
            "goal":       self._goal,
        }
        reward = grade(self._task_id, obs_snapshot)
        # Final guard: must be strictly in (0, 1)
        return max(0.02, min(float(reward), 0.98))

    def _compute_visibility(self) -> float:
        return {"clear": 1.0, "rain": 0.5, "heat": 0.8}.get(self._weather, 1.0)

    def _compute_traction(self) -> float:
        return {"clear": 1.0, "rain": 0.6, "heat": 0.9}.get(self._weather, 1.0)

    def _make_obs(self, reward: float) -> AdaptiveDrivingObservation:
        return AdaptiveDrivingObservation(
            position         = round(self._position, 4),
            speed            = round(self._speed, 4),
            battery          = round(self._battery, 4),
            slope            = self._slope,
            weather          = self._weather,
            visibility       = self._visibility,
            traction         = self._traction,
            distance_to_goal = round(max(0.0, self._goal - self._position), 4),
            goal             = self._goal,
            reward           = round(reward, 4),
            done             = self._done,
            metadata         = {"task": self._task_id, "step": self._step},
        )