import streamlit as st
import pandas as pd, pickle, math, os, requests, subprocess
from datetime import datetime, timedelta

st.set_page_config(page_title="V9 Premier League Predictor", layout="wide")
st.title("⚽ V9 Complete - Premier League Predictor")

# Load model
@st.cache_resource
def load_model():
    try:
        attack = pickle.load(open('attack.pkl','rb'))
        defence = pickle.load(open('defence.pkl','rb'))
        league = pickle.load(open('league_avg.pkl','rb'))
        return attack, defence, league
    except:
        return {}, {}, {'home_avg': 1.53, 'away_avg': 1.20}

attack, defence, league = load_model()
LEAGUE_HOME = league['home_avg']; LEAGUE_AWAY = league['away_avg']

def poisson(lam,k): return (math.exp(-lam)*(lam**k))/math.factorial(k)

def predict_game(home, away):
    if home not in attack or away not in attack:
        return None
    ha = sum(attack[home][-10:])/10; hd = sum(defence[home][-10:])/10
    aa = sum(attack[away][-10:])/10; ad = sum(defence[away][-10:])/10
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
    return ph, pa, p_h, p_d, p_a, over, best_cs, scores[best_cs]

# ===== TABS =====
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Next 24h", "✍️ Save Score", "📊 Tracker (Model vs Real)", "🔄 Retrain"])

# --- TAB 1: PREDICT ---
with tab1:
    st.subheader("Fixtures for Next 24 Hours")
    st.write(f"Model Last Retrained: League Avg Home {LEAGUE_HOME:.2f} - Away {LEAGUE_AWAY:.2f}")
    # Manual fixtures for this weekend - you fit change am, or we go connect Odds API later
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
    # Note: Team names must match your Data files - e.g. "Man City" vs "Manchester City"
    # If e no match, app go show "Team not in model"
    for dt, home, away in fixtures:
        res = predict_game(home, away)
        if res is None:
            st.write(f"**{dt} - {home} vs {away}** - Team name no match training data, check name")
            continue
        ph, pa, p_h, p_d, p_a, over, best_cs, cs_prob = res
        decision = "❌ SKIP"
        if p_h > 60: decision = "✅ BET HOME WIN"
        elif over < 40: decision = "✅ BET UNDER 2.5"
        elif p_h > 55: decision = "⚠️ LEAN HOME"
        st.markdown(f"### {home} vs {away}")
        st.write(f"📅 {dt} | Expected: {ph:.2f} - {pa:.2f} | CS: {best_cs} ({cs_prob:.1f}%) | **{decision}**")
        col1, col2, col3 = st.columns(3)
        col1.metric("Home Win", f"{p_h:.1f}%")
        col2.metric("Draw", f"{p_d:.1f}%")
        col3.metric("Away Win", f"{p_a:.1f}%")
        col1.metric("Over 2.5", f"{over:.1f}%")
        col2.metric("Under 2.5", f"{100-over:.1f}%")
        # Save prediction button
        if st.button(f"Save Prediction - {home} vs {away}", key=f"save_{home}_{away}"):
            os.makedirs("logs", exist_ok=True)
            row = {'Prediction_Date': datetime.now().strftime('%Y-%m-%d'), 'Prediction_Time': datetime.now().strftime('%H:%M:%S'),
                   'Match_Date': dt.split()[0], 'Match_Time': dt.split()[1], 'HomeTeam': home, 'AwayTeam': away,
                   'Expected_Home': round(ph,2), 'Expected_Away': round(pa,2),
                   'Pred_HomeWin_%': round(p_h,1), 'Pred_Draw_%': round(p_d,1), 'Pred_AwayWin_%': round(p_a,1),
                   'Pred_Over_%': round(over,1), 'Pred_Under_%': round(100-over,1),
                   'Pred_CorrectScore': best_cs, 'Pred_Decision': decision,
                   'Real_FTHG': None, 'Real_FTAG': None, 'Real_Result': None, 'Win_Correct': None, 'OverUnder_Correct': None, 'CS_Correct': None}
            f = "logs/predictions_vs_real.csv"
            df = pd.read_csv(f) if os.path.exists(f) else pd.DataFrame()
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(f, index=False)
            st.success(f"Saved {home} vs {away} with time {row['Prediction_Time']}")

# --- TAB 2: SAVE SCORE ---
with tab2:
    st.subheader("Save Real Score - E Go Compare With Model")
    home = st.selectbox("Home Team", list(attack.keys()) if attack else ["Arsenal"])
    away = st.selectbox("Away Team", list(attack.keys()) if attack else ["Chelsea"])
    col1, col2 = st.columns(2)
    fthg = col1.number_input("Home Goals", 0, 10, 0)
    ftag = col2.number_input("Away Goals", 0, 10, 0)
    match_date = st.date_input("Match Date", datetime.now())
    if st.button("Save Real Result"):
        f = "logs/predictions_vs_real.csv"
        if os.path.exists(f):
            df = pd.read_csv(f)
            # Find matching prediction
            mask = (df['HomeTeam']==home) & (df['AwayTeam']==away) & (df['Match_Date']==str(match_date))
            if mask.any():
                idx = df[mask].index[0]
                real_result = 'H' if fthg>ftag else 'D' if fthg==ftag else 'A'
                real_over = 1 if (fthg+ftag)>2.5 else 0
                real_cs = f"{min(fthg,4)}-{min(ftag,4)}"
                # Check correctness
                pred_row = df.loc[idx]
                pred_result = 'H' if pred_row['Pred_HomeWin_%']==max(pred_row['Pred_HomeWin_%'],pred_row['Pred_Draw_%'],pred_row['Pred_AwayWin_%']) else 'D' if pred_row['Pred_Draw_%']==max(pred_row['Pred_HomeWin_%'],pred_row['Pred_Draw_%'],pred_row['Pred_AwayWin_%']) else 'A'
                df.loc[idx,'Real_FTHG']=fthg; df.loc[idx,'Real_FTAG']=ftag; df.loc[idx,'Real_Result']=real_result
                df.loc[idx,'Win_Correct']='YES' if pred_result==real_result else 'NO'
                df.loc[idx,'OverUnder_Correct']='YES' if (1 if pred_row['Pred_Over_%']>50 else 0)==real_over else 'NO'
                df.loc[idx,'CS_Correct']='YES' if pred_row['Pred_CorrectScore']==real_cs else 'NO'
                df.to_csv(f, index=False)
                st.success(f"Updated! {home} {fthg}-{ftag} {away} | Win: {df.loc[idx,'Win_Correct']} | O/U: {df.loc[idx,'OverUnder_Correct']} | CS: {df.loc[idx,'CS_Correct']}")
            else:
                st.warning("No prediction found for this game. Save prediction first for Tab 1.")
        else:
            st.warning("No predictions yet. Go Tab 1 save prediction first.")

# --- TAB 3: TRACKER ---
with tab3:
    st.subheader("Model vs Real - With Time and Date")
    f = "logs/predictions_vs_real.csv"
    if os.path.exists(f):
        df = pd.read_csv(f)
        st.dataframe(df, use_container_width=True)
        if len(df)>0 and 'Win_Correct' in df:
            done = df.dropna(subset=['Win_Correct'])
            if len(done)>0:
                st.write(f"Games Done: {len(done)} | Win Accuracy: {(done['Win_Correct']=='YES').mean()*100:.1f}% | O/U Accuracy: {(done['OverUnder_Correct']=='YES').mean()*100:.1f}% | CS Accuracy: {(done['CS_Correct']=='YES').mean()*100:.1f}%")
    else:
        st.write("No predictions yet.")

# --- TAB 4: RETRAIN ---
with tab4:
    st.subheader("Retrain V9")
    st.write("This go download latest games + retrain. For auto Monday 10am even if PC off, use GitHub Actions we talk before.")
    if st.button("🔄 Retrain Now"):
        with st.spinner("Downloading + Retraining..."):
            try:
                os.makedirs("Data", exist_ok=True)
                url = "https://www.football-data.co.uk/mmz4281/2627/E0.csv"
                r = requests.get(url, timeout=20)
                open("Data/2627.csv","wb").write(r.content)
                st.write("Downloaded 2627.csv")
                result = subprocess.run(["python", "train_v9_complete.py"], capture_output=True, text=True)
                st.code(result.stdout[-2000:])
                st.success("Retrain Done! Reload app to see new model.")
            except Exception as e:
                st.error(f"Error: {e}")