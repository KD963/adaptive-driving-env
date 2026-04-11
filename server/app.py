"""
FastAPI server for Adaptive Driving Environment.

The validator calls:
    POST /reset  {"task": "easy"}   → must reset to that exact task
    POST /step   {"action": {...}}  → must return reward strictly in (0, 1)
    GET  /health                   → must return 200
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Any, Dict, Optional
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AdaptiveDrivingAction, AdaptiveDrivingObservation
from server.adaptive_driving_env_environment import AdaptiveDrivingEnvironment

app = FastAPI(title="Adaptive Driving Environment", version="1.0.0")

# One env instance per task
_envs: Dict[str, AdaptiveDrivingEnvironment] = {}
_last_task: str = "easy"   # tracks which task was last reset


def get_env(task_id: str) -> AdaptiveDrivingEnvironment:
    if task_id not in _envs:
        _envs[task_id] = AdaptiveDrivingEnvironment()
    return _envs[task_id]


class ResetRequest(BaseModel):
    task: Optional[str] = None


class StepRequest(BaseModel):
    action: AdaptiveDrivingAction


class StepResponse(BaseModel):
    observation: AdaptiveDrivingObservation
    done: bool
    reward: float
    info: Dict[str, Any] = {}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=AdaptiveDrivingObservation)
def reset(req: ResetRequest = ResetRequest()):
    global _last_task
    task_id = req.task if req.task in ("easy", "medium", "hard") else "easy"
    _last_task = task_id
    env = get_env(task_id)
    try:
        obs = env.reset(task_id)   # always pass task explicitly
        return obs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    global _last_task
    env = get_env(_last_task)
    try:
        obs = env.step(req.action)
        return StepResponse(observation=obs, done=obs.done, reward=obs.reward)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state(task: str = "easy"):
    return get_env(task).state()


@app.get("/schema")
def schema():
    return {
        "action":      AdaptiveDrivingAction.model_json_schema(),
        "observation": AdaptiveDrivingObservation.model_json_schema(),
    }


@app.get("/tasks")
def list_tasks():
    return [
        {"id": "easy",   "name": "Easy Drive",   "difficulty": "easy"},
        {"id": "medium", "name": "Medium Drive",  "difficulty": "medium"},
        {"id": "hard",   "name": "Hard Drive",    "difficulty": "hard"},
    ]


@app.get("/", response_class=HTMLResponse)
def home():
    return "<h1>🚗 Adaptive Driving Env</h1><a href='/docs'>Docs</a>"


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()