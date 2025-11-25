import requests
import time 
import pandas as pd 
import numpy as np 
import logging

logger = logging.getLogger(__name__)

class GameClient:
    def __init__(self, game_id, serving_client):
        """
        Store game_id, serving client, and initialize tracker.
        """
        self.game_id = game_id
        self.serving = serving_client
        self.last_seen = -1
        self.base_url = f"https://api-web.nhle.com/v1/gamecenter/{game_id}"

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
        payload  = self.fetch_live_game() #fetch everything
        
        plays = payload.get("plays", [])
        if self.last_seen == -1:
            new_events = plays
            self.last_seen = len(plays) - 1
            return new_events

        new_events = plays[self.last_seen + 1 :]
        self.last_seen = len(plays) - 1
        return new_events

  

    def events_to_features(self, event):
        """
        Convert a single NHL event dict into a model-ready feature dict.
        (You will plug in your actual event feature logic here.)
        """
        df = pd.DataFrame(event)
        df = df[(df['typeDescKey'] == 'missed-shot') | (df['typeDescKey'] == 'shot')]
        df['shot_angle'] = np.degrees(np.arctan2(- df['coordinates_y'], 89 - df['coordinates_x']))
        net_x1, net_y1 = 89, 0
        net_x2, net_y2 = -89, 0

        df["distance1"] = np.sqrt(
            (df["coordinates_x"] - net_x1)**2 +
            (df["coordinates_y"] - net_y1)**2
        )

        df["distance2"] = np.sqrt(
            (df["coordinates_x"] - net_x2)**2 +
            (df["coordinates_y"] - net_y2)**2
        )

        df["distance_to_net"] = df[["distance1", "distance2"]].min(axis=1)

        df.drop(columns=["distance1", "distance2"], inplace=True)

        return df 

    def process_new_events(self, start):
        """
        - Fetch live data
        - Get new events only
        - Convert them into features
        - Send to ServingClient.predict()
        - Update last_seen
        - Return predictions
        """
        payload = self.fetch_live_game()
        events  = self.get_new_events(payload)
        finalized  = self.events_to_features(events)
        return finalized
