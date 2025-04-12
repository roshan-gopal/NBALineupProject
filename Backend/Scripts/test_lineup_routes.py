import requests
import json

BASE_URL = "http://localhost:8000/api"

def test_optimize_endpoint():
    print("\nTesting /optimize endpoint...")
    data = {
        "opponent_players": [
            "Jayson Tatum",
            "Jaylen Brown",
            "Kristaps Porzingis",
            "Derrick White",
            "Jrue Holiday"
        ],
        "team": "NYK"
    }
    
    response = requests.post(f"{BASE_URL}/optimize", json=data)
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))

def test_team_endpoint():
    print("\nTesting /team/NYK endpoint...")
    response = requests.get(f"{BASE_URL}/team/NYK")
    print(f"Status Code: {response.status_code}")
    print(f"Number of players returned: {len(response.json())}")
    print("\nFirst 5 players:")
    for player in response.json()[:5]:
        print(f"- {player['PLAYER']}")

def test_test_endpoint():
    print("\nTesting /test endpoint...")
    response = requests.get(f"{BASE_URL}/test")
    print(f"Status Code: {response.status_code}")
    print("Response:")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    print("Starting lineup routes tests...")
    
    try:
        test_optimize_endpoint()
        test_team_endpoint()
        test_test_endpoint()
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the server. Make sure the server is running!")
        print("Run this command to start the server:")
        print("uvicorn Backend.server.main:app --reload") 