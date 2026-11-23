# EPL Match Predictor

Context-aware ML model for Premier League: Over/Under 2.5 + 1X2 + Correct Score (90 mins only).

## Features
- Form Last 5 (Goals, Points)
- xG Form (Expected Goals from Shots + Shots on Target + Corners)
- Rest Days + Rest Advantage
- Stakes: Matchweek, Season Progress, Points, Position, Title Race, Relegation Battle, Must Win, Dead Rubber
- H2H Over Rate Last 5
- Derby, Top6 Clash, Winter, Weekend
- Weather (Open-Meteo) - Live adjustment
- Correct Score: Poisson + Direct Classifier

## No Lineup
Lineup comes 1 hour before kickoff, too late. Model uses form + xG which already capture missing players effect.

## Auto Retrain
Every Monday 10am with latest results.

## Fixtures
The Streamlit Predict tab uses the fixture list in `app.py` and shows only
kickoffs within the next 24 hours, using UK local time. Update that list when
the fixture schedule changes.

## Run
Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI service used by the FORMLY website:

```bash
python -m uvicorn api:app --reload
```

The API and interactive API docs are at `http://localhost:8000` and
`http://localhost:8000/docs`. Set `FORMLY_API_URL` to the website URL before
starting the API. Team and prediction endpoints require an
`Authorization: Bearer <token>` header with a valid token and active Football
subscription.

### API endpoints

- `GET /health` checks that the API is running.
- `GET /teams` lists teams supported by the model.
- `POST /predict` predicts a match. Send JSON such as:

```json
{
  "home_team": "Arsenal",
  "away_team": "Chelsea"
}
```

Predictions include expected goals, 1X2 and Over/Under 2.5 probabilities,
and the most likely score. Unknown teams return `404`.
