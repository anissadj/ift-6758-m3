import streamlit as st
import pandas as pd
import requests
from ift6758.ift6758.client.game_client import GameClient
from ift6758.ift6758.client.serving_client import ServingClient


st.set_page_config(page_title="Hockey Visualization App", layout="wide")

API_URL = "http://serving:8000"   


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
if "gc" not in st.session_state:
    st.session_state.gc = GameClient(game_id=st.session_state.prev_gid, serving_client=None)

  
with st.container():
    st.subheader("Game ID")
    game_id = st.text_input("Enter Game ID", value=st.session_state.prev_gid, key="game_id")
    if int(game_id) != int(st.session_state.prev_gid):
        st.session_state.last_seen = -1
        st.session_state.events_df = None
        st.session_state.meta = None
        st.session_state.prev_gid = int(game_id)
        st.session_state.home_xg = 0.0                   
        st.session_state.away_xg = 0.0
        st.session_state.play_payload = None
        st.session_state.play_idx = 0
        st.session_state.events_df = pd.DataFrame()
        st.session_state.gc = GameClient(game_id=game_id, serving_client=None)         

    if not game_id.strip():
        st.warning("Please enter a game ID.")
    elif not game_id.isdigit() or len(game_id) != 10:
        st.info("Please enter a ten digit ID, ex: 2021020329.")

    if st.button("Ping game"):
        try:
            gc = st.session_state.gc
            payload = gc.fetch_live_game()
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
                period, time_left = gc.event_period_and_time(last_evt)
            else:
                period, time_left = 1, "20:00"

            meta = gc.payload_to_meta(st.session_state.play_payload, game_id)
            meta["period"] = period
            meta["time_remaining"] = time_left
            st.session_state.meta = meta
            
            shot_batch = [e for e in batch if e.get("typeDescKey") in shot_types]
            feats = gc.events_to_features(shot_batch, e.get('situationCode'), str( e.get("eventId")) == str ((e.get("details") or {}).get("eventOwnerTeamId")) )
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
            st.info("Game ended — no more events to fetch.")
            st.info(e)

with st.container():
    gc = st.session_state.gc
    meta = st.session_state.meta
    df = st.session_state.events_df
    if meta:
        st.subheader(f"**Game {meta['game_id']}: {meta['home_team']} @ {meta['away_team']}**")
        colA, colB = st.columns(2)
        with colA:
            st.caption(f"Period {meta.get('period','?')} • Time left {meta.get('time_remaining','—')}")
        st.session_state.home_xg, st.session_state.away_xg = gc.compute_xg(df)
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
            cols = [ c for c in ['distance_to_net','goal_prob','team', 'empty_net', 'is_goal'] if c in Xshow.columns]
        else:
            cols = [ c for c in ['shot_angle','distance_to_net','team','goal_prob', 'empty_net', 'is_goal'] if c in Xshow.columns]
        view = Xshow.loc[:, cols].rename(columns={"goal_prob": "Model output"}).reset_index(drop=True)
        if 'team' in view.columns and meta:
            name_map = {'home': meta.get('home_team'), 'away': meta.get('away_team')}
            view['team'] = view['team'].map(name_map).fillna(view['team'])

        view.index = pd.Index([f"Event {i+1}" for i in range(len(view))], name="")
        st.dataframe(view, width='stretch')
       







