import os
import json
import requests
import pandas as pd
from tqdm import tqdm   # ← FIX 1

BASE_URL = "https://api-web.nhle.com/v1/gamecenter"
GLOBAL_TRACKER = 0 

def __processeventtype_(situationcode, isHome):
    """
    Determine empty-net  
    """
    if situationcode: 
        goalie_away = int(situationcode[0])
        goalie_home = int(situationcode[3])

        #Check if the goalie was present or not
        if (isHome and goalie_away == 0) or (not isHome and goalie_home == 0):
            empty_net = True 
        else:
            empty_net = False

        return empty_net
    return None

def __process_distance(x_cord, y_cord):
    pass 


def _process_angle(x_cord, y_cord):
    pass 


def events_to_dataframe(all_games_events):
    records = []

    for game_id, game_data in tqdm(all_games_events.items()):
        home_team = game_data.get("homeTeam", [])
        away_team = game_data.get("awayTeam", [])
        season = game_data.get("season", [])

        id_h = home_team.get("id", [])
        name_h = home_team.get("commonName", {}).get("default") if home_team else []

        id_a = away_team.get("id")
        name_a = away_team.get("commonName", {}).get("default") if away_team else []

        plays = game_data.get("plays", [])
        players = game_data.get("rosterSpots", [])
        game_time = game_data.get("gameTime")

        for ev in plays:
            ev_type = ev.get("typeDescKey")
            if ev_type not in ["shot-on-goal", "goal"]:
                continue

            details = ev.get("details", {})
            empty_net = __processeventtype_(ev.get("situationCode"), str(id_h) == str(details.get("eventOwnerTeamId")))
            shooter_player_id = details.get("shootingPlayerId")
            scoring_player_id = details.get("scoringPlayerId")
            goalie_in_net_id = details.get("goalieInNetId")
            ev_zone_code = ev.get("details", {}).get('zoneCode')

            record = {
                "event_type": "goal" if ev_type == "goal" else "shot",
                "team_id": details.get("eventOwnerTeamId"),
                "team_name": name_h if str(id_h) == str(details.get("eventOwnerTeamId")) else name_a,
                "coordinates_x": details.get("xCoord"),
                "coordinates_y": details.get("yCoord"),
                "shooter": shooter_player_id or scoring_player_id,
                "goalie": goalie_in_net_id,
                "shot_type": details.get("shotType"),
                "empty_net": empty_net,
                "situation_code": ev.get("situationCode"),
                "zone_code": ev_zone_code
            }

            for player in players:
                player_id = player.get("playerId")

                first = player.get("firstName", {}).get("default", "")
                last = player.get("lastName", {}).get("default", "")
                fullname = f"{first} {last}".strip()

                if player_id == shooter_player_id or player_id == scoring_player_id:
                    record["shooter"] = fullname
                elif player_id == goalie_in_net_id:
                    record["goalie"] = fullname

            records.append(record)

    return pd.DataFrame(records)


def write_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f)


def read_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def get_game(game_id, out_dir):
    if out_dir is None:    
        out_dir = "./games"

    file_path = os.path.join(out_dir, f"{game_id}.json")

    if os.path.exists(file_path):
        return read_json(file_path)

    url = f"{BASE_URL}/{game_id}/play-by-play"
    try:
        r = requests.get(url)
    except requests.RequestException:
        return "fail"

    if r.status_code == 200:
        payload = r.json()
        write_json(payload, file_path)
        return payload

    return None


def controller(gameid):
    payload = get_game(gameid, None)
    df = events_to_dataframe(payload)
    return df
