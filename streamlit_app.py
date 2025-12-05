import streamlit as st
import pandas as pd
import requests
from ift6758.ift6758.client.game_client import GameClient
from ift6758.ift6758.client.serving_client import ServingClient


st.set_page_config(page_title="Hockey Visualization App", layout="wide")

API_URL = "http://127.0.0.1:8000"   


def payload_to_meta(payload, game_id: str):
    return {
        "game_id": game_id,
        "home_team": payload.get("homeTeam", {}).get("commonName", {}).get("default"),
        "away_team": payload.get("awayTeam", {}).get("commonName", {}).get("default"),
        "home_id":   payload.get("homeTeam", {}).get("id"),
        "away_id":   payload.get("awayTeam", {}).get("id"),
        "home_score": payload.get("homeTeam", {}).get("score"),
        "away_score": payload.get("awayTeam", {}).get("score"),
        "period": payload.get("periodDescriptor", {}).get("number"),
        "time_remaining": payload.get("clock", {}).get("timeRemaining", "—"),
    }

def event_period_and_time(evt: dict):
    # The period number 
    pd_num = (evt.get("periodDescriptor") or {}).get("number")
    # The time remaining in that period
    t_rem = evt.get("timeRemaining") 
    return pd_num, t_rem


def compute_xg(df: pd.DataFrame | None):
    # update the home and away xG based on df
    if df is None or df.empty or "goal_prob" not in df.columns or "team" not in df.columns:
        return 0.0, 0.0
    h = df.loc[df["team"] == "home", "goal_prob"].sum()
    a = df.loc[df["team"] == "away", "goal_prob"].sum()
    return float(h), float(a)

def fetch_live_payload(game_id: str):
    url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.json()

def call_predict(df_feats: pd.DataFrame):
    r = requests.post(f"{API_URL}/predict",
                      json=df_feats.to_dict(orient="records"),
                      timeout=60)
    return r.json()
st.title("NHL Shot Goal Probability Predictor")


with st.sidebar:
    workspace = st.selectbox("Workspace", ["IFT6758-2025-B1"])
    model_name = st.selectbox("Model", ["model3-distance-angle", "model1-distance"])
    version = st.selectbox("Version", ["latest"])

    if st.button("Get model"):
        payload = {
            "entity": workspace,
            "project": "IFT6758-2025-B01",
            "model_name": model_name,
            "version": version
        }

        try:
            res = requests.post(f"{API_URL}/download_registry_model", json=payload)
            st.success("Model loaded successfully.")
        except Exception as e:
            st.error(f"Error loading model: {e}")

# Initialize session state variables
if "serving" not in st.session_state:
    st.session_state.serving = ServingClient(API_URL)
if "events_df" not in st.session_state:
    st.session_state.events_df = pd.DataFrame()
if "meta" not in st.session_state:
    st.session_state.meta = None      
if "play_payload" not in st.session_state:
    st.session_state.play_payload = None       
if "play_idx" not in st.session_state:
    st.session_state.play_idx = 0 
if "home_xg" not in st.session_state:
    st.session_state.home_xg = 0.0
if "away_xg" not in st.session_state:
    st.session_state.away_xg = 0.0  
if "prev_gid" not in st.session_state:
    st.session_state.prev_gid = 2021020329
  
with st.container():
    st.subheader("Game ID")
    game_id = st.text_input("Enter Game ID", value=st.session_state.prev_gid, key="game_id")
    if int(game_id) != int(st.session_state.prev_gid):
        print(f"Game ID changed from {st.session_state.prev_gid} to {game_id}, resetting state.")
        st.session_state.last_seen = -1
        st.session_state.events_df = None
        st.session_state.meta = None
        st.session_state.prev_gid = int(game_id)
        st.session_state.home_xg = 0.0                   
        st.session_state.away_xg = 0.0
        st.session_state.play_payload = None
        st.session_state.play_idx = 0
        st.session_state.events_df = pd.DataFrame()             

    if not game_id.strip():
        st.warning("Please enter a game ID.")
    elif not game_id.isdigit() or len(game_id) != 10:
        st.info("Please enter a ten digit ID, ex: 2021020329.")

    if st.button("Ping game"):
        try:
            payload = fetch_live_payload(game_id)
            if st.session_state.play_payload is None or st.session_state.game_id != game_id:
                st.session_state.play_payload = payload
                st.session_state.play_idx = 0
                prev_gid = game_id
              
            pl = st.session_state.play_payload.get("plays", []) or []
            
            # Extract shot events from batch
            shot_types = {"shot-on-goal","missed-shot","goal","failed-shot-attempt"}
            start = st.session_state.play_idx  
            n = len(pl)
            shot_idx = None 
            # each ping fetches up to the next shot event
            for i in range(start, n):
                e = pl[i]
                if e.get("typeDescKey") in shot_types:
                    shot_idx = i
                    break
            if shot_idx is  None:
                batch = []
            else:
                end = shot_idx + 1
                batch = pl[start:end]
                st.session_state.play_idx = end
            if batch:
                last_evt = batch[-1]
                period, time_left = event_period_and_time(last_evt)
            else:
                period, time_left = 1, "20:00"

            meta = payload_to_meta(st.session_state.play_payload, game_id)
            meta["period"] = period
            meta["time_remaining"] = time_left
            st.session_state.meta = meta
            
            shot_batch = [e for e in batch if e.get("typeDescKey") in shot_types]
            gc = GameClient(game_id=game_id, serving_client=None)
            feats = gc.events_to_features(shot_batch)
            owner_map = {}
            for e in shot_batch:
                eid = int(e.get("eventId"))
                # map event ID to owning team ID
                oid = (e.get("details") or {}).get("eventOwnerTeamId")

                owner_map[eid] = oid

            feats["team"] = feats["event_id"].map(lambda eid:
                "home" if owner_map.get(eid) == meta["home_id"] else
                ("away" if owner_map.get(eid) == meta["away_id"] else pd.NA)
            )
            
            if not feats.empty:
                feature_cols = ["distance_to_net"] if model_name == "model1-distance" else ["distance_to_net","shot_angle"]
                to_pred = feats[feature_cols]
                out = call_predict(to_pred)
                if "predictions" in out:
                    probs = [p[1] for p in out["predictions"]]
                    feats = feats.copy()
                    feats["goal_prob"] = probs
                    if st.session_state.events_df is None or st.session_state.events_df.empty:
                        st.session_state.events_df = feats
                    else:
                        st.session_state.events_df = pd.concat(
                            [st.session_state.events_df, feats], ignore_index=True
                        )

            if st.session_state.play_idx >= len(pl):
                st.info("No more events in this game.")
            else:
                st.success(f" events loaded successfully.")

        except Exception as e:
            st.error(f"Game ended — no more events to fetch.")

with st.container():
    meta = st.session_state.meta
    df = st.session_state.events_df
    if meta:
        st.subheader(f"**Game {meta['game_id']}: {meta['home_team']} @ {meta['away_team']}**")
        colA, colB = st.columns(2)
        with colA:
            st.caption(f"Period {meta.get('period','?')} • Time left {meta.get('time_remaining','—')}")
        st.session_state.home_xg, st.session_state.away_xg = compute_xg(df)
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label=f"{meta['home_team']} xG (actual)",
                value=f"{st.session_state.home_xg:.2f} ({meta.get('home_score','?')})",
                delta=f"{st.session_state.home_xg - float(meta.get('home_score', 0)):+.2f}"
            )
        with col2:
            st.metric(
                label=f"{meta['away_team']} xG (actual)",
                value=f"{st.session_state.away_xg:.2f} ({meta.get('away_score','?')})",
                delta=f"{st.session_state.away_xg - float(meta.get('away_score', 0)):+.2f}"
            )
    else:
        st.info("Select a game and click Ping game to fetch info.")

with st.container():
    st.subheader("Data used for predictions (and predictions)")
    Xshow = st.session_state.get("events_df")
    if Xshow is not None and not Xshow.empty:
        if model_name == "model1-distance":
            cols = [ c for c in ['distance_to_net','goal_prob','team'] if c in Xshow.columns]
        else:
            cols = [ c for c in ['shot_angle','distance_to_net','team','goal_prob'] if c in Xshow.columns]
        view = Xshow.loc[:, cols].rename(columns={"goal_prob": "Model output"}).reset_index(drop=True)
        if 'team' in view.columns and meta:
            name_map = {'home': meta.get('home_team'), 'away': meta.get('away_team')}
            view['team'] = view['team'].map(name_map).fillna(view['team'])

        view.index = pd.Index([f"Event {i+1}" for i in range(len(view))], name="")
        st.dataframe(view, width='stretch')
       







