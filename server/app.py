"""
FastAPI application for Adaptive Driving Environment (OpenEnv compliant)
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import sys
import os

# Ensure imports work from root or server/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import AdaptiveDrivingAction, AdaptiveDrivingObservation
from server.adaptive_driving_env_environment import AdaptiveDrivingEnvironment

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="Adaptive Driving Environment",
    description="OpenEnv-compliant RL environment for adaptive driving simulation.",
    version="1.0.0",
)

# Single shared environment instance
_env: Optional[AdaptiveDrivingEnvironment] = None

def get_env() -> AdaptiveDrivingEnvironment:
    global _env
    if _env is None:
        _env = AdaptiveDrivingEnvironment()
    return _env


# ── Request/Response models ───────────────────────────────────

class ResetRequest(BaseModel):
    task: Optional[str] = None


class StepRequest(BaseModel):
    action: AdaptiveDrivingAction


class StepResponse(BaseModel):
    observation: AdaptiveDrivingObservation
    done: bool
    reward: float
    info: Dict[str, Any] = Field(default_factory=dict)  # ✅ FIXED


# ── Inference execution state ─────────────────────────────────

has_run = False
cached_output = ""


@app.get("/run")
def run_inference():
    """
    Runs inference.py exactly once and caches output.
    Subsequent calls return same output (validator-safe).
    """
    global has_run, cached_output

    if has_run:
        return {"output": cached_output}

    import io

    old_stdout = sys.stdout
    buffer = io.StringIO()
    sys.stdout = buffer

    try:
        from inference import main
        main()
        has_run = True

    except Exception as e:
        # Always return valid format
        print(f"[START] task=error env=adaptive-driving model=error")
        print(f"[STEP] step=1 action=error reward=0.00 done=true error={str(e)}")
        print(f"[END] success=false steps=1 rewards=0.00 error={str(e)}")

    finally:
        sys.stdout = old_stdout

    cached_output = buffer.getvalue()
    return {"output": cached_output}


# ── Core Endpoints ────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <h1>🚗 Adaptive Driving Env</h1>
    <p>API is running successfully ✅</p>
    <ul>
        <li><a href="/docs">Swagger Docs</a></li>
        <li><a href="/schema">Schema</a></li>
    </ul>
    """


@app.post("/reset", response_model=StepResponse)
def reset(req: ResetRequest = ResetRequest()):
    env = get_env()
    try:
        obs = env.reset(req.task)
        return StepResponse(
            observation=obs,
            done=obs.done,
            reward=obs.reward,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    env = get_env()
    try:
        obs = env.step(req.action)
        return StepResponse(
            observation=obs,
            done=obs.done,
            reward=obs.reward,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    env = get_env()
    try:
        return env.state() if hasattr(env, "state") else {"status": "no state() method"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/schema")
def schema():
    return {
        "action": AdaptiveDrivingAction.model_json_schema(),
        "observation": AdaptiveDrivingObservation.model_json_schema(),
    }


# ── Demo UI ───────────────────────────────────────────────────

@app.get("/demo", response_class=HTMLResponse)
def demo_ui():
    return """
    <html>
    <head><title>🚗 Adaptive Driving AI</title></head>
    <body style="text-align:center;font-family:sans-serif;padding:40px">
        <h1>🚗 Adaptive Driving Simulation</h1>
        <button onclick="run()" style="padding:10px 24px;font-size:16px">▶ Run Simulation</button>
        <pre id="output" style="margin-top:20px;text-align:left;display:inline-block;
             background:#f4f4f4;padding:20px;border-radius:8px;min-width:400px"></pre>
        <script>
        async function run() {
            document.getElementById("output").innerText = "Starting...";
            await fetch("/reset", {
                method: "POST",
                headers: {"Content-Type":"application/json"},
                body: JSON.stringify({task: "easy"})
            });
            let output = "";
            for (let i = 0; i < 20; i++) {
                const res = await fetch("/step", {
                    method: "POST",
                    headers: {"Content-Type":"application/json"},
                    body: JSON.stringify({ action: { move: "accelerate" } })
                });
                const data = await res.json();
                const obs = data.observation;
                output += `Step ${i+1}: pos=${obs.position.toFixed(1)} speed=${obs.speed.toFixed(1)} battery=${obs.battery.toFixed(1)} reward=${obs.reward.toFixed(2)}\\n`;
                document.getElementById("output").innerText = output;
                if (data.done) { output += "\\n✅ Done!"; break; }
            }
            document.getElementById("output").innerText = output;
        }
        </script>
    </body>
    </html>
    """


# ── Entry point ───────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()