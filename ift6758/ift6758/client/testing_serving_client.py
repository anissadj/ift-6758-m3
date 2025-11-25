import pandas as pd
from serving_client import ServingClient

def test_predict():
    print("Testing the prediction")
    client = ServingClient(ip="localhost", port=5000)
    df = pd.DataFrame([
        {"distance_to_net": 1},
        {"distance_to_net": 25},
        {"distance_to_net": 40}
    ])
    response = client.predict(df)
    print("Status Code:", response.status_code)
    try:
        print("Response JSON:", response.json())
    except:
        print("Raw Response:", response.text)


def test_logs():
    print("Testing the logs")
    client = ServingClient(ip="localhost", port=5000)
    try:
        result = client.logs()
        print("Logs have been loaded")
    except Exception as e:
        print("Hitting an error calling logs")
        return False 


def test_download_registry_model():
    print('testing downloaiding the registy')

    client = ServingClient(ip="localhost", port=5000)

    # Use dummy values unless your server really connects to Comet
    entity =  "IFT6758-2025-B1",
    project = "IFT6758-2025-B01",
    model_name =  "model1-distance",
    version =  "latest"

    try:
        result = client.download_registry_model(entity=entity, project=project, model_name=model_name, version=version)
        print("Registry Download Response:", result)
    except Exception as e:
        print("Error calling /download_registry_model:", e)


if __name__ == "__main__":
    print("====== RUNNING SERVING_CLIENT TESTS ======")

    prediction  = test_predict() == True 
    logs = test_logs()  
    download  = test_download_registry_model()  

    if (prediction and logs and download): 
        print("\n====== ALL TESTS COMPLETED WITH SUCCESS ======")
    else:
        print("\n====== There was an error in one of our test ======")


