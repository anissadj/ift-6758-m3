from game_client import GameClient
from serving_client import ServingClient

def _init_test_clients():
    serving_client = ServingClient(ip="localhost", port=5000)
    game_id = "2025020100"  # Test game ID
    game_client = GameClient(game_id, serving_client)
    return game_client

def test_fetch_live_games():
  print("Testing fetch_live_game...")
  
  game_client = _init_test_clients()
  
  _ = game_client.fetch_live_game()
  
  try:
    print("Payload fetched successfully:")
    return True
  except Exception as e:
    print("Error fetching payload:", e)
    return False


def test_get_new_events():
  print("Testing get_new_events...")
  
  game_client = _init_test_clients()
  
  payload = game_client.fetch_live_game()
  _ = game_client.get_new_events(payload)
  
  try:
    print("New events fetched successfully:")
    return True
  except Exception as e:
    print("Error fetching new events:", e)
    return False
  
def test_events_to_features():
  print("Testing events_to_features...")

  game_client = _init_test_clients()

  payload = game_client.fetch_live_game()
  new_events = game_client.get_new_events(payload)

  if not new_events:
    print("No new events to test.")
    return False

  features = game_client.events_to_features(new_events)

  try:
    print("Features extracted successfully:")
    print(features.head())
    return True
  except Exception as e:
    print("Error extracting features:", e)
    return False
  
def test_process_new_events():
  print("Testing process_new_events...")

  game_client = _init_test_clients()
  
  batch_probs = game_client.process_new_events()
  
  try:
    print("Processed new events successfully:")
    print(game_client.goal_probabilities)
    return True 
  except Exception as e:
    print("Error processing new events:", e)
    return False


if __name__ == "__main__":
  print("====== RUNNING GAME_CLIENT TESTS ======")

  fetch_live  = test_fetch_live_games()  
  new_events  = test_get_new_events()  
  features  = test_events_to_features()
  batch_probs = test_process_new_events()
  

  if (fetch_live and new_events and features and batch_probs): 
    print("\n====== ALL TESTS COMPLETED WITH SUCCESS ======")
  else:
    print("\n====== There was an error in one of the tests ======")