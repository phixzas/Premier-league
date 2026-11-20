# EPL Match Predictor - V5

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

## Run
pip install -r requirements.txt
python train_v5.py
python live_predict_v5.py