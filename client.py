"""Adaptive Driving Environment Client (FINAL - Validator Safe)."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import AdaptiveDrivingAction, AdaptiveDrivingObservation


class AdaptiveDrivingEnv(
    EnvClient[AdaptiveDrivingAction, AdaptiveDrivingObservation, State]
):
    """
    Client for Adaptive Driving Environment.
    """

    # ✅ FIX 1: Match server contract
    def _step_payload(self, action: AdaptiveDrivingAction) -> Dict:
        """
        Convert action into API payload.
        Server expects: {"move": "accelerate" | "brake"}
        """
        return {
            "move": action.move  # ✅ aligned with server
        }

    # ✅ FIX 2: Parse full observation correctly
    def _parse_result(self, payload: Dict) -> StepResult[AdaptiveDrivingObservation]:
        obs_data = payload.get("observation", {})

        observation = AdaptiveDrivingObservation(
            position=obs_data.get("position", 0.0),
            speed=obs_data.get("speed", 0.0),
            battery=obs_data.get("battery", 0.0),
            slope=obs_data.get("slope", 0.0),
            weather=obs_data.get("weather", "clear"),
            visibility=obs_data.get("visibility", 1.0),
            traction=obs_data.get("traction", 1.0),
            distance_to_goal=obs_data.get("distance_to_goal", 0.0),
            goal=obs_data.get("goal", 1.0),  # ✅ CRITICAL FIX
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=float(payload.get("reward", 0.0)),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )