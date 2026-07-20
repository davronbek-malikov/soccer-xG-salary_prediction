"""
Render.com / Gradio app entry point.

On cold start:
  1. Loads raw CSV data from data/raw/
  2. Engineers features (no target used)
  3. Trains XGBoost (lightweight — fits in Render free 512 MB RAM)
  4. Caches model so UptimeRobot wake-ups after sleep skip retraining
  5. Serves Gradio UI on PORT env var (set by Render)

UptimeRobot pings / every 5 min → keeps Render free tier awake 24/7.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.logger           import get_logger
from src.data_loader      import DataLoader
from src.feature_engineer import FeatureEngineer
from src.preprocessor     import Preprocessor

import gradio as gr

log = get_logger("render_app")

# Render provides PORT; fall back to 7860 for local runs
PORT = int(os.environ.get("PORT", 7860))

CACHE_DIR   = Path(".cache")
MODEL_PATH  = CACHE_DIR / "model.pkl"
PREP_PATH   = CACHE_DIR / "preprocessor.pkl"
RAW_DIR     = "data/raw"
SALARY_PATH = "data/raw/Salary/capology_big5_latest.xls"

LEAGUES   = ["Bundesliga", "La Liga", "Serie A"]
POSITIONS = ["F S", "M S", "D", "GK", "F M S", "M", "D M", "F"]


# ── train once, cache for restarts ──────────────────────────────────
def train_and_cache():
    log.info("Cold start — training XGBoost (~40 s on Render free CPU)...")
    CACHE_DIR.mkdir(exist_ok=True)

    loader = DataLoader(RAW_DIR, SALARY_PATH)
    df     = loader.load_all()

    fe = FeatureEngineer()
    y  = np.log1p(df["salary"].values)
    X  = fe.fit_transform(df)

    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    prep = Preprocessor()
    Xp   = prep.fit_transform(X_tr)

    # Lightweight config: fits comfortably inside Render's 512 MB RAM
    model = XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0, n_jobs=1,
    )
    model.fit(Xp, y_tr)

    # Safe: saving our own XGBRegressor and sklearn Pipeline only.
    joblib.dump(model, MODEL_PATH)
    prep.save(str(PREP_PATH))
    log.info("Model cached → .cache/")
    return model, prep, fe


def load_or_train():
    if MODEL_PATH.exists() and PREP_PATH.exists():
        log.info("Loading cached model (skipping retraining)...")
        # Safe: loading files written by train_and_cache() above — our own XGBRegressor.
        model = joblib.load(MODEL_PATH)
        prep  = Preprocessor.load(str(PREP_PATH))
        fe    = FeatureEngineer()
        fe._fitted = True
        return model, prep, fe
    return train_and_cache()


model, prep, fe = load_or_train()
log.info("Ready — serving on port %d", PORT)


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
        "player": "demo", "team": "demo",
    }
    df  = fe.transform(pd.DataFrame([row]))
    X   = prep.transform(df)
    sal = np.expm1(model.predict(X)[0])
    return (
        f"£ {sal:>12,.0f}  /  year\n"
        f"£ {sal/52:>12,.0f}  /  week\n"
        f"£ {sal/12:>12,.0f}  /  month"
    )


# ── Gradio UI ────────────────────────────────────────────────────────
css = """
.gr-button-primary { background: #2563eb !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="⚽ Soccer Salary Predictor", css=css, theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """# ⚽ Soccer Player Salary Predictor
Estimate annual gross salary (£) from one season of statistics.
_Trained on Bundesliga · La Liga · Serie A · seasons 2014/15 – 2021/22_
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Playing Time")
            games    = gr.Slider(1,  38,   value=28,   step=1,  label="Games played")
            time_min = gr.Slider(1,  3600, value=2300, step=50, label="Minutes played")
            age      = gr.Slider(16, 40,   value=25,   step=1,  label="Age")

        with gr.Column(scale=1):
            gr.Markdown("### Attacking")
            goals = gr.Slider(0,  50,   value=10,  step=1,   label="Goals")
            xG    = gr.Slider(0., 40.,  value=9.0, step=0.5, label="xG")
            npg   = gr.Slider(0,  50,   value=9,   step=1,   label="Non-Penalty Goals")
            npxG  = gr.Slider(0., 40.,  value=8.5, step=0.5, label="npxG")
            shots = gr.Slider(0,  200,  value=60,  step=5,   label="Shots")

        with gr.Column(scale=1):
            gr.Markdown("### Creativity")
            assists    = gr.Slider(0,  30,  value=5,    step=1,   label="Assists")
            xA         = gr.Slider(0., 20., value=4.5,  step=0.5, label="xA")
            key_passes = gr.Slider(0,  150, value=40,   step=5,   label="Key Passes")
            xGChain    = gr.Slider(0., 30., value=15.0, step=0.5, label="xGChain")
            xGBuildup  = gr.Slider(0., 20., value=5.0,  step=0.5, label="xGBuildup")

        with gr.Column(scale=1):
            gr.Markdown("### Context")
            yellow_cards = gr.Slider(0, 15, value=3, step=1, label="Yellow Cards")
            red_cards    = gr.Slider(0, 5,  value=0, step=1, label="Red Cards")
            league   = gr.Dropdown(LEAGUES,   value="Bundesliga", label="League")
            position = gr.Dropdown(POSITIONS, value="F S",        label="Position")

    predict_btn = gr.Button("Predict Salary", variant="primary", size="lg")

    output = gr.Textbox(label="Predicted Annual Salary", lines=3, interactive=False)

    predict_btn.click(
        fn=predict_salary,
        inputs=[
            games, time_min, goals, xG, assists, xA, shots, key_passes,
            yellow_cards, red_cards, npg, npxG, xGChain, xGBuildup,
            age, league, position,
        ],
        outputs=output,
    )

    gr.Examples(
        examples=[
            [34, 2900, 27, 25.1, 8, 7.2, 130, 55, 2, 0, 26, 23.8, 32.0, 8.0, 24, "La Liga",    "F S"],
            [30, 2600,  8,  9.5, 10, 9.8,  60, 72, 5, 1,  8,  9.1, 20.0, 6.0, 28, "Bundesliga", "M S"],
            [32, 2700,  2,  2.4,  4, 4.1,  20, 38, 4, 0,  2,  2.4, 12.0, 7.0, 30, "Serie A",    "D"],
        ],
        inputs=[
            games, time_min, goals, xG, assists, xA, shots, key_passes,
            yellow_cards, red_cards, npg, npxG, xGChain, xGBuildup,
            age, league, position,
        ],
        label="Example Players",
    )

    gr.Markdown(
        "---\n"
        "Built by **Davronbek Malikov** · "
        "[GitHub](https://github.com/davronbek-malikov/soccer-xG-salary_prediction)"
    )

if __name__ == "__main__":
    # server_name="0.0.0.0" required so Render's router can reach the app
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False)
