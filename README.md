---
title: "Adaptive Driving AI"
emoji: "🚗"
colorFrom: "blue"
colorTo: "green"
sdk: "gradio"
sdk_version: "4.0.0"
python_version: "3.10"
app_file: "app.py"
pinned: false
---

# 🚗 Adaptive Driving AI Environment

An AI-powered driving simulation where an LLM controls a car to reach a goal under constraints like:

- Weather (rain, fog, heat)
- Road slope
- Battery limitations

## 🎮 Features

- Real-time simulation
- Reward-based learning
- LLM decision-making (accelerate / brake)
- Visual track rendering in UI

## 🚀 How it works

1. Environment resets with random conditions
2. LLM decides next action
3. Environment updates:
   - position
   - speed
   - battery
4. Reward is calculated
5. Stops when goal reached or failure

## 🧠 Tech Stack

- OpenAI Client
- Gradio UI
- Custom RL Environment

## ▶️ Run Locally

```bash
pip install -r requirements.txt
python app.py