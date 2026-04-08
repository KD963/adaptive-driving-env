from pydantic import BaseModel, Field
from typing import Optional, Dict


class AdaptiveDrivingAction(BaseModel):
    move: str = Field(
        ...,
        description="Action: 'accelerate' or 'brake'"
    )


class AdaptiveDrivingObservation(BaseModel):
    position:         float
    speed:            float
    battery:          float
    slope:            float
    weather:          str
    visibility:       float
    traction:         float
    distance_to_goal: float

    # ── FIX: goal was missing — inference.py needs it ──
    # goal is the absolute target position = position + distance_to_goal
    # We compute it as a property AND expose it as a plain field so that
    # both obs.goal and serialisation work without changes to inference.py.
    goal: float = Field(
        default=0.0,
        description="Absolute target position (position + distance_to_goal)"
    )

    reward: float = 0.0
    done:   bool  = False

    metadata: Optional[Dict] = None

    def model_post_init(self, __context):
        """Auto-compute goal if not explicitly provided."""
        if self.goal == 0.0 and self.distance_to_goal != 0.0:
            # Use object.__setattr__ because Pydantic models are frozen-ish
            object.__setattr__(self, "goal", self.position + self.distance_to_goal)