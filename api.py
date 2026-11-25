import os
import datetime

import requests
from fastapi import FastAPI, HTTPException
from fastapi import Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predictor import load_model, predict_game

app = FastAPI(
    title="Premier League Predictor API",
    description="Match predictions from the model used by the Streamlit app.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def require_formly_access(authorization: str | None, sport: str):
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Sign in to access this model.")
    website_url = os.getenv("FORMLY_API_URL", "").rstrip("/")
    if not website_url:
        raise HTTPException(status_code=503, detail="Model access service is not configured.")
    try:
        response = requests.post(
            f"{website_url}/api/access/check",
            headers={"Authorization": authorization},
            json={"sport": sport},
            timeout=10,
        )
    except requests.RequestException as error:
        raise HTTPException(status_code=503, detail="Could not verify subscription access.") from error
    if response.status_code == 401:
        raise HTTPException(status_code=401, detail="Sign in again.")
    if not response.ok:
        raise HTTPException(status_code=503, detail="Could not verify subscription access.")
    try:
        access = response.json()
    except requests.exceptions.JSONDecodeError as error:
        raise HTTPException(status_code=503, detail="Could not verify subscription access.") from error
    if not isinstance(access, dict) or access.get("active") is not True:
        raise HTTPException(status_code=403, detail="An active Football subscription is required.")

class MatchRequest(BaseModel):
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)

class PredictionResponse(BaseModel):
    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float
    home_win_probability_pct: float
    draw_probability_pct: float
    away_win_probability_pct: float
    over_2_5_probability_pct: float
    under_2_5_probability_pct: float
    most_likely_score: str
    most_likely_score_probability_pct: float

# --- Helper: build a clean prediction - FIXES WRONG RESULTS ---
def build_prediction(home_input: str, away_input: str):
    if home_input.lower().strip() == away_input.lower().strip():
        raise HTTPException(status_code=422, detail="Home and away teams must be different.")

    result, home, away = predict_game(home_input.strip(), away_input.strip())
    if result is None:
        attack, _, _ = load_model()
        missing = [t for t in (home, away) if t not in attack]
        raise HTTPException(status_code=404, detail=f"Team(s) not found in model: {', '.join(missing)}")

    (exp_home, exp_away, home_win, draw, away_win, over, correct_score, correct_score_prob) = result

    # FIX: Clamp crazy values from stale model
    exp_home = max(0.0, min(5.0, float(exp_home)))
    exp_away = max(0.0, min(5.0, float(exp_away)))

    # FIX: Normalise 1X2 so it ALWAYS sums to 100% - this was causing wrong %
    total = float(home_win) + float(draw) + float(away_win)
    if total <= 0:
        home_win, draw, away_win = 33.3, 33.3, 33.4
    else:
        home_win = round(float(home_win) / total * 100, 1)
        draw = round(float(draw) / total * 100, 1)
        away_win = round(100 - home_win - draw, 1)

    over = max(0.0, min(100.0, float(over)))
    return PredictionResponse(
        home_team=home,
        away_team=away,
        expected_home_goals=round(exp_home, 2),
        expected_away_goals=round(exp_away, 2),
        home_win_probability_pct=home_win,
        draw_probability_pct=draw,
        away_win_probability_pct=away_win,
        over_2_5_probability_pct=round(over, 1),
        under_2_5_probability_pct=round(100 - over, 1),
        most_likely_score=correct_score,
        most_likely_score_probability_pct=round(float(correct_score_prob), 1),
    )

@app.get("/")
@app.get("/health")
@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.get("/teams", response_model=list[str])
@app.get("/api/teams", response_model=list[str])
def teams(authorization: str | None = Header(default=None)):
    require_formly_access(authorization, "Football")
    attack, _, _ = load_model()
    return sorted(attack)

@app.post("/predict", response_model=PredictionResponse)
@app.post("/api/predict", response_model=PredictionResponse)
def predict_match(match: MatchRequest, authorization: str | None = Header(default=None)):
    require_formly_access(authorization, "Football")
    return build_prediction(match.home_team, match.away_team)

# --- NEW: NEXT 24 HOURS GAMES + AUTO PREDICTIONS ---
@app.get("/api/fixtures")
@app.get("/api/games")
@app.get("/api/games/next-24h")
@app.get("/fixtures")
def get_next_24h_games(authorization: str | None = Header(default=None)):
    require_formly_access(authorization, "Football")

    api_key = os.getenv("FOOTBALL_DATA_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="FOOTBALL_DATA_API_KEY not set on Render. Add it in Environment Variables.")

    now = datetime.datetime.now(datetime.timezone.utc)
    tomorrow = now + datetime.timedelta(hours=24)

    # football-data.org expects YYYY-MM-DD
    date_from = now.strftime("%Y-%m-%d")
    date_to = tomorrow.strftime("%Y-%m-%d")

    try:
        resp = requests.get(
            f"https://api.football-data.org/v4/competitions/PL/matches?dateFrom={date_from}&dateTo={date_to}",
            headers={"X-Auth-Token": api_key},
            timeout=15,
        )
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Could not fetch fixtures: {e}") from e

    if not resp.ok:
        raise HTTPException(status_code=503, detail=f"Fixtures API failed: {resp.status_code}")

    data = resp.json()
    games = []
    for m in data.get("matches", []):
        kickoff_str = m.get("utcDate", "")
        try:
            kickoff = datetime.datetime.fromisoformat(kickoff_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        # Only keep games in next 24h, not past games
        if not (now - datetime.timedelta(minutes=5) <= kickoff <= tomorrow):
            continue

        home_name = m["homeTeam"]["name"]
        away_name = m["awayTeam"]["name"]

        # Auto-predict this game with your model
        try:
            pred = build_prediction(home_name, away_name)
            pred_dict = pred.model_dump()
        except Exception:
            pred_dict = None

        games.append({
            "home_team": home_name,
            "away_team": away_name,
            "kickoff": kickoff_str,
            "status": m.get("status"),
            "competition": "Premier League",
            "prediction": pred_dict,
        })

    # Sort by kickoff time
    games.sort(key=lambda x: x["kickoff"])
    return {"count": len(games), "fixtures": games, "games": games}