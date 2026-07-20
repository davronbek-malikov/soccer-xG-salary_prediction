"""
Gradio demo — interactive soccer salary predictor.

Run from project root:
    python app/gradio_demo.py

Loads the saved best model from models/best_model/.
"""

import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# ── make src importable ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineer import FeatureEngineer
from src.preprocessor     import Preprocessor
from src.logger           import get_logger
import gradio as gr

log = get_logger("gradio_demo")

MODEL_DIR = Path(__file__).parent.parent / "models" / "best_model"

# ── load artefacts ───────────────────────────────────────────────────
# Safe: loading our own sklearn objects produced by run_pipeline.py
model = joblib.load(MODEL_DIR / "model.pkl")
prep  = Preprocessor.load(MODEL_DIR / "preprocessor.pkl")

fe = FeatureEngineer()
fe._fitted = True   # stateless transforms — no fit needed

log.info("Model and preprocessor loaded for Gradio demo")

LEAGUES   = ["Bundesliga", "La Liga", "Serie A"]
POSITIONS = ["F S", "M S", "D", "GK", "F M S", "M", "D M", "F"]


# ── prediction function ──────────────────────────────────────────────
def predict_salary(
    games, time_min, goals, xG, assists, xA, shots, key_passes,
    yellow_cards, red_cards, npg, npxG, xGChain, xGBuildup,
    age, league, position,
):
    row = {
        "games": games, "time": time_min, "goals": goals, "xG": xG,
        "assists": assists, "xA": xA, "shots": shots, "key_passes": key_passes,
        "yellow_cards": yellow_cards, "red_cards": red_cards,
        "npg": npg, "npxG": npxG, "xGChain": xGChain, "xGBuildup": xGBuildup,
        "age": age, "league": league, "position": position,
        # non-feature cols needed by FE but will be dropped
        "player": "demo", "team": "demo",
    }
    df = pd.DataFrame([row])
    df = fe.transform(df)

    # align columns to what the preprocessor expects
    X = prep.transform(df)
    log_pred = model.predict(X)[0]
    salary_yr = np.expm1(log_pred)

    return (
        f"£{salary_yr:>12,.0f} / year\n"
        f"£{salary_yr/52:>12,.0f} / week\n"
        f"£{salary_yr/12:>12,.0f} / month"
    )


# ── Gradio interface ─────────────────────────────────────────────────
with gr.Blocks(title="⚽ Soccer Salary Predictor") as demo:
    gr.Markdown(
        "# ⚽ Soccer Player Salary Predictor\n"
        "Input a player's season statistics to estimate their annual gross salary (GBP).\n"
        "_Model trained on Bundesliga · La Liga · Serie A — seasons 2014/15 to 2021/22._"
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Playing Time")
            games        = gr.Slider(1,  38,  value=28,   step=1,   label="Games played")
            time_min     = gr.Slider(1,  3600, value=2300, step=50,  label="Minutes played")
            age          = gr.Slider(16, 40,  value=25,   step=1,   label="Age")

        with gr.Column():
            gr.Markdown("### Attacking Stats")
            goals        = gr.Slider(0,  50,  value=10,   step=1,   label="Goals")
            xG           = gr.Slider(0.0, 40.0, value=9.0, step=0.5, label="xG (Expected Goals)")
            npg          = gr.Slider(0,  50,  value=9,    step=1,   label="Non-Penalty Goals")
            npxG         = gr.Slider(0.0, 40.0, value=8.5, step=0.5, label="npxG")
            shots        = gr.Slider(0,  200, value=60,   step=5,   label="Shots")

        with gr.Column():
            gr.Markdown("### Creativity & Build-up")
            assists      = gr.Slider(0,  30,  value=5,    step=1,   label="Assists")
            xA           = gr.Slider(0.0, 20.0, value=4.5, step=0.5, label="xA (Expected Assists)")
            key_passes   = gr.Slider(0,  150, value=40,   step=5,   label="Key Passes")
            xGChain      = gr.Slider(0.0, 30.0, value=15.0, step=0.5, label="xGChain")
            xGBuildup    = gr.Slider(0.0, 20.0, value=5.0,  step=0.5, label="xGBuildup")

        with gr.Column():
            gr.Markdown("### Discipline & Context")
            yellow_cards = gr.Slider(0, 15, value=3, step=1, label="Yellow Cards")
            red_cards    = gr.Slider(0, 5,  value=0, step=1, label="Red Cards")
            league       = gr.Dropdown(choices=LEAGUES,   value="Bundesliga", label="League")
            position     = gr.Dropdown(choices=POSITIONS, value="F S",        label="Position")

    predict_btn = gr.Button("Predict Salary", variant="primary")
    output      = gr.Textbox(label="Predicted Annual Salary", lines=3)

    predict_btn.click(
        fn=predict_salary,
        inputs=[
            games, time_min, goals, xG, assists, xA, shots, key_passes,
            yellow_cards, red_cards, npg, npxG, xGChain, xGBuildup,
            age, league, position,
        ],
        outputs=output,
    )

    gr.Markdown(
        "---\n"
        "**Note:** Predictions are in log-space internally then back-transformed to £/year. "
        "Model: best ensemble from run_pipeline.py."
    )

if __name__ == "__main__":
    demo.launch(share=True)
