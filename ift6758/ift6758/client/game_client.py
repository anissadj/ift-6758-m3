import requests
import time 
import pandas as pd 
import numpy as np 
import logging

logger = logging.getLogger(__name__)

class GameClient:
    
    SHOT_TYPES = {"shot-on-goal", "missed-shot", "goal", "failed-shot-attempt"}
    
    def __init__(self, game_id, serving_client):
        """
        Store game_id, serving client, and initialize tracker.
        """
        self.game_id = game_id
        self.serving = serving_client
        self.last_seen = -1
        self.base_url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}"
        self.goal_probabilities = {}

    def fetch_live_game(self):
        """
        Query the NHL API to get the current game state.
        Return the list of 'plays'.
        """
        url = f"{self.base_url}/play-by-play"
        try:
            r = requests.get(url)
        except requests.RequestException:
            return "fail"

        if r.status_code == 200:
            payload = r.json()
            return payload

    def get_new_events(self, payload):
        """
        Compare the fetched events with last_seen,
        return only the events that have not yet been processed.
        """
        
        if payload is None:
            return []
        
        plays = payload.get("plays", [])
        if self.last_seen == -1:
            new_events = plays
        else:
            new_events = plays[self.last_seen + 1 :]

        self.last_seen = len(plays) - 1 # len starts at 1, index at 0
        return new_events

    def _event_to_features_row(self, event: dict) -> dict | None:
        """
        Private helper function:
        - Takes a single play/event dict
        - If it's a valid shot event with coordinates, return a feature row (dict)
        """

        type_key = event.get("typeDescKey")
        if type_key not in self.SHOT_TYPES:
            return None

        event_id = event.get("eventId")
        details = event.get("details") or {}
        x = details.get("xCoord")
        y = details.get("yCoord")

        if x is None or y is None:
            return None

        row = {
            "event_id": event_id,
            "typeDescKey": type_key,
            "coordinates_x": x,
            "coordinates_y": y,
        }

        return row

    def events_to_features(self, events: list[dict] | dict) -> pd.DataFrame:
        """
        - Take a list of events (or a single event dict)
        - For each event, extract features if it's a valid shot event
        - Return a DataFrame of features
        """
        
        # If a single event dict is passed, wrap it into a list
        if isinstance(events, dict):
            events = [events]

        if not events:
            return pd.DataFrame()

        rows = []
        for e in events:
            row = self._event_to_features_row(e)
            if row is not None:
                rows.append(row)

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        
        net_x1, net_y1 = 89, 0
        net_x2, net_y2 = -89, 0

        df["shot_angle"] = np.degrees(np.arctan2(-df["coordinates_y"], 89 - df["coordinates_x"]))

        df["distance1"] = np.sqrt((df["coordinates_x"] - net_x1) ** 2 + (df["coordinates_y"] - net_y1) ** 2)
        df["distance2"] = np.sqrt((df["coordinates_x"] - net_x2) ** 2 + (df["coordinates_y"] - net_y2) ** 2)

        df["distance_to_net"] = df[["distance1", "distance2"]].min(axis=1)
        df.drop(columns=["distance1", "distance2"], inplace=True)

        return df

    def process_new_events(self, start=None):
        """
        - Fetch live data
        - Get new events only
        - Convert them into features
        - Send to ServingClient.predict()
        - Update last_seen
        - Return predictions
        """
        
        payload = self.fetch_live_game()
        
        if not payload:
            logger.warning("No payload returned from NHL API.")
            return None
        
        events  = self.get_new_events(payload)
        
        if not events:
            logger.info("No new events to process.")
            return None
        
        features_df  = self.events_to_features(events)
        
        if features_df.empty:
            logger.info("No valid shot events found in new events.")
            return None
        
        try:
            response = self.serving.predict(features_df[["distance_to_net"]])
        except Exception as e:
            logger.error(f"Error during prediction request: {e}")
            return None
        
        try:
            data = response.json()
        except:
            logger.error("Failed to parse JSON from prediction response.")
            return None
        
        
        preds = data.get("predictions", [])
        batch_probs = []
        
        for event_id, probs in zip(features_df["event_id"], preds):
            
            p_no_goal, p_goal = probs[0], probs[1]
            
            self.goal_probabilities[event_id] = float(p_goal)
            
            record = {
                "eventId": int(event_id),
                "prob_goal": float(p_goal),
                "prob_no_goal": float(p_no_goal)
            }
            
            batch_probs.append(record)
        
        return batch_probs
    def payload_to_meta(self,payload, game_id: str):
        """
        Extract metadata from the payload.
        """
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
    
    def event_period_and_time(self,evt: dict):
        # Extract period number and time remaining from an event.
        pd_num = (evt.get("periodDescriptor") or {}).get("number")
        # The time remaining in that period
        t_rem = evt.get("timeRemaining") 
        return pd_num, t_rem
    
    def compute_xg(self,df: pd.DataFrame | None):
        """
        Update home and away xG from the events DataFrame.
        """
        if df is None or df.empty or "goal_prob" not in df.columns or "team" not in df.columns:
            return 0.0, 0.0
        h = df.loc[df["team"] == "home", "goal_prob"].sum()
        a = df.loc[df["team"] == "away", "goal_prob"].sum()
        return float(h), float(a)