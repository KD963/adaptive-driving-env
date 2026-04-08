"""Adaptive Driving Environment Client (Real-world EV simulation)."""

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

    Supports real-world EV simulation with weather, slope, and battery constraints.
    """

    def _step_payload(self, action: AdaptiveDrivingAction) -> Dict:
        """
        Convert action into API payload.
        """
        return {
            "acceleration": action.acceleration,
            "brake": action.brake,
        }

    def _parse_result(self, payload: Dict) -> StepResult[AdaptiveDrivingObservation]:
        """
        Parse server response.
        """
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
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse environment state.
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )