import math
import pickle
from functools import lru_cache
from pathlib import Path


TEAM_MAP = {
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Leeds United": "Leeds",
    "Coventry City": "Coventry",
    "Tottenham Hotspur": "Tottenham",
    "Ipswich Town": "Ipswich",
    "Hull City": "Hull",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Nottingham Forest": "Nott'm Forest",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
    "Wolverhampton": "Wolves",
    "Leicester City": "Leicester",
    "Sheffield United": "Sheffield Utd",
}


def fix_name(name):
    name = name.strip()
    return TEAM_MAP.get(name, name)


@lru_cache(maxsize=1)
def load_model():
    model_dir = Path(__file__).resolve().parent / "Notebook"
    with (model_dir / "attack.pkl").open("rb") as model_file:
        attack = pickle.load(model_file)
    with (model_dir / "defence.pkl").open("rb") as model_file:
        defence = pickle.load(model_file)
    with (model_dir / "league_avg.pkl").open("rb") as model_file:
        league = pickle.load(model_file)
    return attack, defence, league


def poisson(lam, k):
    return (math.exp(-lam) * (lam**k)) / math.factorial(k)


def predict_game(home_raw, away_raw):
    attack, defence, league = load_model()
    home = fix_name(home_raw)
    away = fix_name(away_raw)
    if home not in attack or away not in attack:
        return None, home, away

    league_home = league.get("home_avg", 1.53)
    league_away = league.get("away_avg", 1.20)
    ha = sum(attack[home][-10:]) / 10
    hd = sum(defence[home][-10:]) / 10
    aa = sum(attack[away][-10:]) / 10
    ad = sum(defence[away][-10:]) / 10
    ph = (ha / league_home) * (ad / league_away) * league_home
    pa = (aa / league_away) * (hd / league_home) * league_away
    ph = max(0.3, min(3.0, 0.8 * ph + 0.2 * league_home))
    pa = max(0.2, min(2.6, 0.8 * pa + 0.2 * league_away))

    p_h = sum(
        poisson(ph, h) * poisson(pa, a)
        for h in range(6)
        for a in range(6)
        if h > a
    )
    p_d = sum(
        poisson(ph, h) * poisson(pa, a)
        for h in range(6)
        for a in range(6)
        if h == a
    )
    p_a = sum(
        poisson(ph, h) * poisson(pa, a)
        for h in range(6)
        for a in range(6)
        if h < a
    )
    total = p_h + p_d + p_a
    p_h, p_d, p_a = p_h / total * 100, p_d / total * 100, p_a / total * 100

    p0 = poisson(ph, 0) * poisson(pa, 0)
    p1 = poisson(ph, 1) * poisson(pa, 0) + poisson(ph, 0) * poisson(pa, 1)
    p2 = (
        poisson(ph, 2) * poisson(pa, 0)
        + poisson(ph, 1) * poisson(pa, 1)
        + poisson(ph, 0) * poisson(pa, 2)
    )
    over = (1 - p0 - p1 - p2) * 100
    scores = {
        f"{h}-{a}": poisson(ph, h) * poisson(pa, a) * 100
        for h in range(5)
        for a in range(5)
    }
    best_cs = max(scores, key=scores.get)
    return (ph, pa, p_h, p_d, p_a, over, best_cs, scores[best_cs]), home, away
