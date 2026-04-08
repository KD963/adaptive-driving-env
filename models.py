from pydantic import BaseModel, Field
from typing import Optional, Dict


class AdaptiveDrivingAction(BaseModel):
    move: str = Field(
        ...,
        description="Action: 'accelerate' or 'brake'"
    )


class AdaptiveDrivingObservation(BaseModel):
    position: float
    speed: float
    battery: float
    slope: float
    weather: str
    visibility: float
    traction: float
    distance_to_goal: float

    reward: float = 0.0
    done: bool = False

    metadata: Optional[Dict] = None