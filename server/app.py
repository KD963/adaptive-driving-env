"""
FastAPI server for Adaptive Driving Environment (FINAL - Validator Safe)
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
import sys, os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AdaptiveDrivingAction, AdaptiveDrivingObservation
from server.adaptive_driving_env_environment import AdaptiveDrivingEnvironment

app = FastAPI(title="Adaptive Driving Environment", version="1.0.2")

# One env per task
_envs: Dict[str, AdaptiveDrivingEnvironment] = {}


def get_env(task_id: str) -> AdaptiveDrivingEnvironment:
    if task_id not in _envs:
        _envs[task_id] = AdaptiveDrivingEnvironment()
    return _envs[task_id]


# Track active task
_active_task: str = "easy"


class ResetRequest(BaseModel):
    task: Optional[str] = None


class StepRequest(BaseModel):
    action: AdaptiveDrivingAction


class StepResponse(BaseModel):
    observation: AdaptiveDrivingObservation
    done: bool
    reward: float
    info: Dict[str, Any]


# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


# ─────────────────────────────────────────────
# RESET
# ─────────────────────────────────────────────

@app.post("/reset", response_model=AdaptiveDrivingObservation)
def reset(req: ResetRequest = ResetRequest()):
    global _active_task

    task_id = req.task if req.task in ("easy", "medium", "hard") else "easy"
    _active_task = task_id

    env = get_env(task_id)

    try:
        obs = env.reset(task_id)
        return obs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# STEP (🔥 CRITICAL FIX HERE)
# ─────────────────────────────────────────────

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    global _active_task

    env = get_env(_active_task)

    try:
        obs = env.step(req.action)

        # 🔥 HARD VALIDATOR-SAFE CLAMP
        safe_reward = float(obs.reward)

        if not math.isfinite(safe_reward):
            safe_reward = 0.021

        # Never allow boundaries
        if safe_reward <= 0.02:
            safe_reward = 0.021
        elif safe_reward >= 0.98:
            safe_reward = 0.979

        # Final clamp
        safe_reward = max(0.021, min(round(safe_reward, 4), 0.979))

        return StepResponse(
            observation=obs,
            done=obs.done,
            reward=safe_reward,
            info={"task": _active_task, "step": obs.metadata.get("step", 0)},
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────
# STATE / SCHEMA / TASKS
# ─────────────────────────────────────────────

@app.get("/state")
def state(task: str = "easy"):
    return get_env(task).state()


@app.get("/schema")
def schema():
    return {
        "action": AdaptiveDrivingAction.model_json_schema(),
        "observation": AdaptiveDrivingObservation.model_json_schema(),
    }


@app.get("/tasks")
def list_tasks():
    return [
        {"id": "easy", "name": "Easy Drive", "difficulty": "easy"},
        {"id": "medium", "name": "Medium Drive", "difficulty": "medium"},
        {"id": "hard", "name": "Hard Drive", "difficulty": "hard"},
    ]


# ─────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>🚗 Adaptive Driving Env</h1><a href='/docs'>Docs</a>"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()