import streamlit as st
import pandas as pd, pickle, math, os, requests, subprocess
from datetime import datetime, timedelta

st.set_page_config(page_title="Premier League Predictor", layout="wide")
st.title("⚽ Complete - Premier League Predictor")

# --- FIX TEAM NAMES ---
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

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    try:
        attack = pickle.load(open('attack.pkl','rb'))
        defence = pickle.load(open('defence.pkl','rb'))
        league = pickle.load(open('league_avg.pkl','rb'))
        return attack, defence, league
    except Exception as e:
        st.error(f"Model files not found: {e} - Run train_v9_complete.py first")
        return {}, {}, {'home_avg': 1.53, 'away_avg': 1.20}

attack, defence, league = load_model()
LEAGUE_HOME = league.get('home_avg', 1.53)
LEAGUE_AWAY = league.get('away_avg', 1.20)

with st.expander("🔍 See Teams Model Knows (click to open)"):
    st.write(f"Total teams in model: {len(attack)}")
    st.write(sorted(list(attack.keys())))

def poisson(lam, k):
    return (math.exp(-lam) * (lam**k)) / math.factorial(k)

def predict_game(home_raw, away_raw):
    home = fix_name(home_raw)
    away = fix_name(away_raw)
    if home not in attack or away not in attack:
        return None, home, away
    ha = sum(attack[home][-10:]) / 10; hd = sum(defence[home][-10:]) / 10
    aa = sum(attack[away][-10:]) / 10; ad = sum(defence[away][-10:]) / 10
    ph = (ha / LEAGUE_HOME) * (ad / LEAGUE_AWAY) * LEAGUE_HOME
    pa = (aa / LEAGUE_AWAY) * (hd / LEAGUE_HOME) * LEAGUE_AWAY
    ph = 0.8*ph + 0.2*LEAGUE_HOME; pa = 0.8*pa + 0.2*LEAGUE_AWAY
    ph = max(0.3, min(3.0, ph)); pa = max(0.2, min(2.6, pa))
    p_h = sum(poisson(ph,h)*poisson(pa,a) for h in range(6) for a in range(6) if h>a)
    p_d = sum(poisson(ph,h)*poisson(pa,a) for h in range(6) for a in range(6) if h==a)
    p_a = sum(poisson(ph,h)*poisson(pa,a) for h in range(6) for a in range(6) if h<a)
    tot = p_h+p_d+p_a; p_h,p_d,p_a = p_h/tot*100, p_d/tot*100, p_a/tot*100
    p0 = poisson(ph,0)*poisson(pa,0)
    p1 = poisson(ph,1)*poisson(pa,0) + poisson(ph,0)*poisson(pa,1)
    p2 = poisson(ph,2)*poisson(pa,0) + poisson(ph,1)*poisson(pa,1) + poisson(ph,0)*poisson(pa,2)
    over = (1-p0-p1-p2)*100
    scores = {f"{h}-{a}": poisson(ph,h)*poisson(pa,a)*100 for h in range(5) for a in range(5)}
    best_cs = max(scores, key=scores.get)
    return (ph, pa, p_h, p_d, p_a, over, best_cs, scores[best_cs]), home, away

# Helper to check duplicate - Home + Away + Match_Date
def is_already_saved(home, away, match_date):
    f = "logs/predictions_vs_real.csv"
    if not os.path.exists(f):
        return False
    try:
        df = pd.read_csv(f)
        if df.empty or 'HomeTeam' not in df.columns:
            return False
        return ((df['HomeTeam'] == home) & (df['AwayTeam'] == away) & (df['Match_Date'] == match_date)).any()
    except:
        return False

# ===== TABS =====
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Next 24h", "✍️ Save Score", "📊 Tracker (Model vs Real)", "🔄 Retrain"])

# --- TAB 1 ---
with tab1:
    st.subheader("Fixtures for Next 24 Hours")
    st.write(f"League Avg Home {LEAGUE_HOME:.2f} - Away {LEAGUE_AWAY:.2f}")

    fixtures = [
        ("2026-11-21 13:30", "Man City", "Fulham"),
        ("2026-11-21 16:00", "Bournemouth", "Nottingham Forest"),
        ("2026-11-21 16:00", "Aston Villa", "Sunderland"),
        ("2026-11-21 16:00", "Chelsea", "Leeds United"),
        ("2026-11-21 16:00", "Coventry City", "Crystal Palace"),
        ("2026-11-21 16:00", "Tottenham Hotspur", "Ipswich Town"),
        ("2026-11-21 18:30", "Newcastle United", "Arsenal"),
        ("2026-11-22 15:00", "Hull City", "Brighton & Hove Albion"),
        ("2026-11-22 17:30", "Liverpool", "Manchester United"),
        ("2026-11-23 21:00", "Brentford", "Everton"),
    ]

    for dt, home_raw, away_raw in fixtures:
        res, home, away = predict_game(home_raw, away_raw)
        if res is None:
            st.warning(f"**{dt} - {home_raw} vs {away_raw}** -> After fix: `{home}` vs `{away}` - NOT in model. Check expander above for correct names.")
            continue
        ph, pa, p_h, p_d, p_a, over, best_cs, cs_prob = res
        decision = "❌ SKIP"
        if p_h > 60: decision = "✅ BET HOME WIN"
        elif over < 40: decision = "✅ BET UNDER 2.5"
        elif p_h > 55: decision = "⚠️ LEAN HOME"
        st.markdown(f"### {home_raw} vs {away_raw}")
        st.write(f"📅 {dt} | Expected: {ph:.2f} - {pa:.2f} | **{decision}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Home Win", f"{p_h:.1f}%")
        c2.metric("Draw", f"{p_d:.1f}%")
        c3.metric("Away Win", f"{p_a:.1f}%")
        c4, c5, c6 = st.columns(3)
        c4.metric("Over 2.5", f"{over:.1f}%")
        c5.metric("Under 2.5", f"{100-over:.1f}%")
        c6.metric("Correct Score", f"{best_cs}", f"{cs_prob:.1f}% chance")
        # FIX: NO DUPLICATE SAVE - Check before showing button
        match_date_only = dt.split()[0]
        if is_already_saved(home, away, match_date_only):
            st.info(f"✅ Already saved: {home} vs {away} on {match_date_only} - No need save again")
        else:
            if st.button(f"Save Prediction - {home_raw} vs {away_raw}", key=f"save_{home_raw}_{away_raw}"):
                # Check again inside button - for double-click fast
                if is_already_saved(home, away, match_date_only):
                    st.warning(f"⚠️ Already saved before: {home} vs {away} - Not saved twice")
                else:
                    os.makedirs("logs", exist_ok=True)
                    row = {'Prediction_Date': datetime.now().strftime('%Y-%m-%d'), 'Prediction_Time': datetime.now().strftime('%H:%M:%S'),
                           'Match_Date': match_date_only, 'Match_Time': dt.split()[1], 'HomeTeam': home, 'AwayTeam': away,
                           'Expected_Home': round(ph,2), 'Expected_Away': round(pa,2),
                           'Pred_HomeWin_%': round(p_h,1), 'Pred_Draw_%': round(p_d,1), 'Pred_AwayWin_%': round(p_a,1),
                           'Pred_Over_%': round(over,1), 'Pred_Under_%': round(100-over,1),
                           'Pred_CorrectScore': best_cs, 'Pred_Decision': decision,
                           'Real_FTHG': None, 'Real_FTAG': None, 'Real_Result': None, 'Win_Correct': None, 'OverUnder_Correct': None, 'CS_Correct': None}
                    f = "logs/predictions_vs_real.csv"
                    df = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
                    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
                    df.to_csv(f, index=False)
                    st.success(f"Saved {home} vs {away} at {row['Prediction_Time']}")
                    st.rerun()

# --- TAB 2 ---
with tab2:
    st.subheader("Save Real Score - After Game (3 hours after kickoff)")
    st.write("Game don finish? Press button below. E go fetch real score automatically. No need wait Monday.")

    if st.button("🔄 Fetch Real Results Now (For Finished Games)"):
        try:
            url = "https://www.football-data.co.uk/mmz4281/2627/E0.csv"
            r = requests.get(url, timeout=20)
            df_results = pd.read_csv(pd.io.common.BytesIO(r.content))
            st.write(f"Downloaded {len(df_results)} games from football-data")
        except Exception as e:
            st.error(f"Download failed: {e}")
            df_results = pd.DataFrame()

        f = "logs/predictions_vs_real.csv"
        if os.path.exists(f) and not df_results.empty:
            df = pd.read_csv(f)
            updated = 0
            now = datetime.now()
            for idx, row in df.iterrows():
                if pd.notna(row.get('Real_FTHG')):
                    continue
                try:
                    kickoff = datetime.strptime(f"{row['Match_Date']} {row['Match_Time']}", "%Y-%m-%d %H:%M")
                except:
                    continue
                finish_time = kickoff + timedelta(hours=3)
                if now < finish_time:
                    st.write(f"⏳ {row['HomeTeam']} vs {row['AwayTeam']} - Not finished yet. Finish time: {finish_time.strftime('%Y-%m-%d %H:%M')}")
                    continue
                match = df_results[(df_results['HomeTeam'] == row['HomeTeam']) & (df_results['AwayTeam'] == row['AwayTeam'])]
                if match.empty:
                    st.write(f"❓ {row['HomeTeam']} vs {row['AwayTeam']} - No result found yet in CSV")
                    continue
                fthg = int(match.iloc[0]['FTHG'])
                ftag = int(match.iloc[0]['FTAG'])
                real_result = 'H' if fthg > ftag else 'D' if fthg == ftag else 'A'
                real_over = 1 if (fthg + ftag) > 2.5 else 0
                real_cs = f"{fthg}-{ftag}"
                pred_result = 'H' if row['Pred_HomeWin_%'] == max(row['Pred_HomeWin_%'], row['Pred_Draw_%'], row['Pred_AwayWin_%']) else 'D' if row['Pred_Draw_%'] == max(row['Pred_HomeWin_%'], row['Pred_Draw_%'], row['Pred_AwayWin_%']) else 'A'
                df.loc[idx, 'Real_FTHG'] = fthg
                df.loc[idx, 'Real_FTAG'] = ftag
                df.loc[idx, 'Real_Result'] = real_result
                df.loc[idx, 'Win_Correct'] = 'YES' if pred_result == real_result else 'NO'
                df.loc[idx, 'OverUnder_Correct'] = 'YES' if (1 if row['Pred_Over_%'] > 50 else 0) == real_over else 'NO'
                df.loc[idx, 'CS_Correct'] = 'YES' if row['Pred_CorrectScore'] == real_cs else 'NO'
                updated += 1
                st.success(f"✅ {row['HomeTeam']} {fthg}-{ftag} {row['AwayTeam']} | Win: {df.loc[idx,'Win_Correct']} | O/U: {df.loc[idx,'OverUnder_Correct']} | CS: {df.loc[idx,'CS_Correct']}")
            df.to_csv(f, index=False)
            if updated == 0:
                st.info("No new finished games to update.")
            else:
                st.success(f"Updated {updated} games!")
        else:
            st.warning("No predictions yet. Go Tab 1 save prediction first.")

    st.divider()
    st.write("**Manual - If auto fetch fail, type am yourself:**")
    if attack:
        home_m = st.selectbox("Home Team (manual)", sorted(list(attack.keys())), key="manual_home")
        away_m = st.selectbox("Away Team (manual)", sorted(list(attack.keys())), key="manual_away")
    else:
        home_m = st.text_input("Home Team"); away_m = st.text_input("Away Team")
    c1, c2 = st.columns(2)
    fthg_m = c1.number_input("Home Goals", 0, 10, 0, key="manual_fthg")
    ftag_m = c2.number_input("Away Goals", 0, 10, 0, key="manual_ftag")
    match_date_m = st.date_input("Match Date", datetime.now(), key="manual_date")
    if st.button("Save Manual Result"):
        f = "logs/predictions_vs_real.csv"
        if os.path.exists(f):
            df = pd.read_csv(f)
            mask = (df['HomeTeam'] == home_m) & (df['AwayTeam'] == away_m) & (df['Match_Date'] == str(match_date_m))
            if mask.any():
                idx = df[mask].index[0]
                real_result = 'H' if fthg_m > ftag_m else 'D' if fthg_m == ftag_m else 'A'
                real_over = 1 if (fthg_m + ftag_m) > 2.5 else 0
                real_cs = f"{fthg_m}-{ftag_m}"
                pred_row = df.loc[idx]
                pred_result = 'H' if pred_row['Pred_HomeWin_%'] == max(pred_row['Pred_HomeWin_%'], pred_row['Pred_Draw_%'], pred_row['Pred_AwayWin_%']) else 'D' if pred_row['Pred_Draw_%'] == max(pred_row['Pred_HomeWin_%'], pred_row['Pred_Draw_%'], pred_row['Pred_AwayWin_%']) else 'A'
                df.loc[idx, 'Real_FTHG'] = fthg_m; df.loc[idx, 'Real_FTAG'] = ftag_m; df.loc[idx, 'Real_Result'] = real_result
                df.loc[idx, 'Win_Correct'] = 'YES' if pred_result == real_result else 'NO'
                df.loc[idx, 'OverUnder_Correct'] = 'YES' if (1 if pred_row['Pred_Over_%'] > 50 else 0) == real_over else 'NO'
                df.loc[idx, 'CS_Correct'] = 'YES' if pred_row['Pred_CorrectScore'] == real_cs else 'NO'
                df.to_csv(f, index=False)
                st.success(f"Updated manually! {home_m} {fthg_m}-{ftag_m} {away_m}")
            else:
                st.warning("No prediction found for that game/date.")

# --- TAB 3 ---
with tab3:
    st.subheader("Model vs Real")
    f = "logs/predictions_vs_real.csv"
    st.divider()
    st.write("**Reset Tracker**")
    confirm_reset = st.checkbox("Clear all saved predictions and real results", key="confirm_tracker_reset")
    if st.button("🗑️ Clear Tracker History", disabled=not confirm_reset):
        columns = [
            'Prediction_Date', 'Prediction_Time', 'Match_Date', 'Match_Time',
            'HomeTeam', 'AwayTeam', 'Expected_Home', 'Expected_Away',
            'Pred_HomeWin_%', 'Pred_Draw_%', 'Pred_AwayWin_%',
            'Pred_Over_%', 'Pred_Under_%', 'Pred_CorrectScore', 'Pred_Decision',
            'Real_FTHG', 'Real_FTAG', 'Real_Result', 'Win_Correct',
            'OverUnder_Correct', 'CS_Correct'
        ]
        os.makedirs("logs", exist_ok=True)
        pd.DataFrame(columns=columns).to_csv(f, index=False)
        st.success("Tracker history cleared. You can start fresh.")
        st.rerun()

    if os.path.exists(f):
        df = pd.read_csv(f)
        st.dataframe(df, use_container_width=True)
        if len(df)>0 and 'Win_Correct' in df:
            done = df.dropna(subset=['Win_Correct'])
            if len(done)>0:
                st.write(f"Games Done: {len(done)} | Win: {(done['Win_Correct']=='YES').mean()*100:.1f}% | O/U: {(done['OverUnder_Correct']=='YES').mean()*100:.1f}% | CS: {(done['CS_Correct']=='YES').mean()*100:.1f}%")
    else:
        st.write("No predictions yet.")

# --- TAB 4 ---
with tab4:
    st.subheader("Retrain V9 - ONLY Monday 10am")
    st.write("Do NOT retrain after every game. Retrain only Monday 10am, after all weekend games don finish.")
    today = datetime.now()
    day_name = today.strftime("%A")
    st.write(f"Today na: **{day_name} {today.strftime('%Y-%m-%d %H:%M')}**")
    is_monday = today.weekday() == 0
    if not is_monday:
        st.warning(f"⛔ Today na {day_name}, no be Monday. Retrain go wait till Monday 10am.")
        st.write("Why? If you retrain now, model go learn half-weekend only, e go spoil your accuracy.")
        st.write("Real results: Go Tab 2 after game + 3 hours. Retrain: Monday only.")
        allow_override = st.checkbox("I know wetin I dey do - Force retrain today (not recommended)")
    else:
        st.success("✅ Today na Monday - You fit retrain now!")
        allow_override = True
    if st.button("🔄 Retrain Now"):
        if not is_monday and not allow_override:
            st.error("Blocked: Today no be Monday. Tick the checkbox if you really want force am.")
        else:
            with st.spinner("Downloading + Retraining..."):
                try:
                    os.makedirs("Data", exist_ok=True)
                    url = "https://www.football-data.co.uk/mmz4281/2627/E0.csv"
                    r = requests.get(url, timeout=20)
                    open("Data/2627.csv", "wb").write(r.content)
                    st.write("Downloaded 2627.csv")
                    result = subprocess.run(["python", "train_v9_complete.py"], capture_output=True, text=True)
                    st.code(result.stdout[-2000:])
                    if result.stderr:
                        st.code(result.stderr[-1000:])
                    st.success("Retrain Done! Press R or reload app to see new model.")
                    st.write(f"Retrained on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Monday")
                except Exception as e:
                    st.error(f"Error: {e}")
    st.divider()
    st.write("**For Auto Monday 10am even if PC off:**")
    st.write("Use GitHub Actions - `.github/workflows/retrain.yml` with cron `0 9 * * 1` (09:00 UTC = 10:00 WAT). GitHub go retrain for you even if your laptop off.")