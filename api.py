import io
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
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
    home_team: str = Field(min_length=1, description="Home team name")
    away_team: str = Field(min_length=1, description="Away team name")

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

def _build_prediction(home_raw: str, away_raw: str):
    result, home, away = predict_game(home_raw, away_raw)
    if result is None:
        return None, home, away
    (
        expected_home,
        expected_away,
        home_win,
        draw,
        away_win,
        over,
        correct_score,
        correct_score_probability,
    ) = result
    return PredictionResponse(
        home_team=home,
        away_team=away,
        expected_home_goals=expected_home,
        expected_away_goals=expected_away,
        home_win_probability_pct=home_win,
        draw_probability_pct=draw,
        away_win_probability_pct=away_win,
        over_2_5_probability_pct=over,
        under_2_5_probability_pct=100 - over,
        most_likely_score=correct_score,
        most_likely_score_probability_pct=correct_score_probability,
    ), home, away

@app.get("/health")
@app.get("/api/health")
@app.get("/")
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
    home_input = match.home_team.strip()
    away_input = match.away_team.strip()
    if not home_input or not away_input:
        raise HTTPException(status_code=422, detail="Team names must not be blank.")

    pred, home, away = _build_prediction(home_input, away_input)
    if pred is None:
        attack, _, _ = load_model()
        missing_teams = [team for team in (home, away) if team not in attack]
        raise HTTPException(
            status_code=404,
            detail=f"Team(s) not found in model: {', '.join(missing_teams)}",
        )
    return pred

# ===== NEW: Same as Streamlit Tab 1 - Next 24h, but with FRESH fixtures =====
@app.get("/api/fixtures")
@app.get("/api/games/next-24h")
def get_next_24h_games(authorization: str | None = Header(default=None)):
    # Keep your subscription check - website must send token
    require_formly_access(authorization, "Football")

    uk = ZoneInfo("Europe/London")
    now = datetime.now(uk)
    next_24h = now + timedelta(hours=24)

    # SAME CSV your Streamlit uses for retrain / real scores
    # This has ALL 2026/27 E0 fixtures + results, with Date + Time
    url = "https://www.football-data.co.uk/mmz4281/2627/E0.csv"
    try:
        r = requests.get(url, timeout=20)
        df = pd.read_csv(io.BytesIO(r.content))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Could not fetch fixtures: {e}")

    games = []
    for _, row in df.iterrows():
        try:
            # E0.csv: Date = 21/11/2026, Time = 16:00
            kickoff = datetime.strptime(f"{row['Date']} {row['Time']}", "%d/%m/%Y %H:%M").replace(tzinfo=uk)
        except Exception:
            continue

        # SAME filter as your Streamlit: now <= kickoff <= now + 24h
        if not (now <= kickoff <= next_24h):
            continue

        home_raw = str(row['HomeTeam'])
        away_raw = str(row['AwayTeam'])

        pred, home, away = _build_prediction(home_raw, away_raw)
        if pred is None:
            # Team not in model - skip like Streamlit does with warning
            continue

        games.append({
            "home_team": home_raw,
            "away_team": away_raw,
            "kickoff": kickoff.strftime("%Y-%m-%d %H:%M"),
            "status": "SCHEDULED",
            "prediction": pred.model_dump(),
        })

    games.sort(key=lambda x: x["kickoff"])
    return {"count": len(games), "fixtures": games, "games": games}