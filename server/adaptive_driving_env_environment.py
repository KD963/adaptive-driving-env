"""
AdaptiveDrivingEnvironment — computes reward via grader on every step.

The OpenEnv validator reads observation.reward from the HTTP response.
The grader must be called inside step() and the result stored in obs.reward.
"""

import random
from models import AdaptiveDrivingAction, AdaptiveDrivingObservation
from tasks import TASKS


class AdaptiveDrivingEnvironment:

    def __init__(self):
        self._task_id   = "easy"
        self._position  = 0.0
        self._speed     = 0.0
        self._battery   = 100.0
        self._slope     = 0.0
        self._weather   = "clear"
        self._visibility = 1.0
        self._traction  = 1.0
        self._goal      = 50.0
        self._step      = 0
        self._done      = False

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, task_id: str = None) -> AdaptiveDrivingObservation:
        if task_id is None or task_id not in TASKS:
            task_id = random.choice(list(TASKS.keys()))

        cfg = TASKS[task_id]
        self._task_id    = task_id
        self._position   = 0.0
        self._speed      = 0.0
        self._battery    = float(cfg.get("battery",    100))
        self._slope      = float(cfg.get("slope",      0))
        self._weather    = cfg.get("weather", "clear")
        self._visibility = self._compute_visibility()
        self._traction   = self._compute_traction()
        self._goal       = float(cfg.get("goal", 50))
        self._step       = 0
        self._done       = False

        return self._make_obs(reward=0.01)   # initial reward > 0 (never 0.0)

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action: AdaptiveDrivingAction) -> AdaptiveDrivingObservation:
        if self._done:
            return self._make_obs(reward=0.01)

        self._step += 1
        move = action.move.lower().strip()

        # Physics
        if move == "accelerate":
            accel = max(0.5, 2.0 - self._slope * 0.3) * self._traction
            self._speed    = min(self._speed + accel, 20.0)
            self._battery  = max(0.0, self._battery - 2.0)
        elif move == "brake":
            self._speed    = max(0.0, self._speed - 3.0)
            self._battery  = max(0.0, self._battery - 0.5)

        self._position += self._speed
        self._position  = min(self._position, self._goal)

        # Episode termination
        reached = self._position >= self._goal
        out_of_battery = self._battery <= 0
        timed_out = self._step >= 50

        self._done = reached or out_of_battery or timed_out

        # ── Compute reward via grader ──────────────────────────────────────
        # This is what the validator reads. Must be strictly in (0.01, 0.99).
        from grader import grade
        obs_snapshot = {
            "position":         self._position,
            "speed":            self._speed,
            "battery":          self._battery,
            "visibility":       self._visibility,
            "distance_to_goal": max(0.0, self._goal - self._position),
            "goal":             self._goal,
        }
        reward = grade(self._task_id, obs_snapshot)

        return self._make_obs(reward=reward)

    # ── state ────────────────────────────────────────────────────────────────

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

    # ── helpers ───────────────────────────────────────────────────────────────

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
            reward           = round(float(reward), 4),
            done             = self._done,
            metadata         = {"task": self._task_id, "step": self._step},
        )