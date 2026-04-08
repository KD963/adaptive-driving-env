import gradio as gr
import traceback
from server.adaptive_driving_env_environment import AdaptiveDrivingEnvironment

# Global environment instance
env = None

def format_obs(obs):
    """Safely extract values from observation object."""
    try:
        return (
            "✅ OK",
            f"Task: {obs.metadata.get('task', 'N/A')} | Step: {obs.metadata.get('step', 0)}",
            str(obs.position),
            str(obs.speed),
            str(obs.battery),
            str(obs.slope),
            str(obs.weather),
            str(obs.traction),
            str(obs.distance_to_goal),
        )
    except Exception as e:
        err = f"❌ {str(e)}"
        return (err,) * 9


def reset_env(task_name):
    global env
    try:
        env = AdaptiveDrivingEnvironment()
        obs = env.reset(task_name if task_name != "random" else None)
        return format_obs(obs)
    except Exception:
        tb = traceback.format_exc()
        print(tb)  # Print to terminal for debugging
        err = f"❌ Reset failed: {tb.splitlines()[-1]}"
        return (err,) * 9


def take_step(action):
    global env
    try:
        if env is None:
            raise ValueError("Environment not initialized. Click 'Reset' first.")

        from models import AdaptiveDrivingAction
        act = AdaptiveDrivingAction(move=action)
        obs = env.step(act)
        return format_obs(obs)
    except Exception:
        tb = traceback.format_exc()
        print(tb)  # Print to terminal for debugging
        err = f"❌ Step failed: {tb.splitlines()[-1]}"
        return (err,) * 9


# ── UI Layout ──────────────────────────────────────────────
with gr.Blocks(title="Adaptive Driving Environment 🚗") as demo:
    gr.Markdown("# Adaptive Driving Environment 🚗💨")

    with gr.Row():
        task_dropdown = gr.Dropdown(
            choices=["random", "easy", "medium", "hard"],
            value="easy",
            label="Task Difficulty"
        )
        reset_btn = gr.Button("🔄 Reset Environment", variant="primary")

    with gr.Row():
        action_radio = gr.Radio(
            choices=["accelerate", "brake"],
            value="accelerate",
            label="Action"
        )
        step_btn = gr.Button("▶ Take Step", variant="secondary")

    gr.Markdown("### 📊 Environment State")

    with gr.Row():
        out_status = gr.Textbox(label="Status", interactive=False)
        out_track = gr.Textbox(label="Track Info", interactive=False)

    with gr.Row():
        out_position = gr.Textbox(label="Position", interactive=False)
        out_speed = gr.Textbox(label="Speed", interactive=False)
        out_battery = gr.Textbox(label="Battery", interactive=False)

    with gr.Row():
        out_slope = gr.Textbox(label="Slope", interactive=False)
        out_weather = gr.Textbox(label="Weather", interactive=False)
        out_traction = gr.Textbox(label="Traction", interactive=False)
        out_distance = gr.Textbox(label="Distance to Goal", interactive=False)

    # Output list must match format_obs() return order
    outputs = [
        out_status, out_track,
        out_position, out_speed, out_battery,
        out_slope, out_weather, out_traction, out_distance
    ]

    reset_btn.click(fn=reset_env, inputs=[task_dropdown], outputs=outputs)
    step_btn.click(fn=take_step, inputs=[action_radio], outputs=outputs)

if __name__ == "__main__":
    demo.launch(share=True)