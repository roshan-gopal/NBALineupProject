# from fastapi import APIRouter, HTTPException
# from typing import List, Dict, Any
# import torch
# import random
# from supabase import create_client
# import os
# import sys

# # Add the project root to the Python path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
# sys.path.append(project_root)

# from config import SUPABASE_URL, SUPABASE_KEY

# router = APIRouter()

# # Initialize Supabase client
# supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# # Define state columns
# STATE_COLUMNS = ['PIE', 'NETRTG', 'MIN', 'USG%', 'TS%', 'AST%']

# # Define the Q-Network
# class QNetwork(torch.nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super().__init__()
#         self.net = torch.nn.Sequential(
#             torch.nn.Linear(30, 128),  # Input layer: 30 features (6 stats × 5 players)
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 64),  # Hidden layer: 128 → 64 neurons
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 9)     # Output layer: 64 → 9 actions
#         )
#     def forward(self, x):
#         return self.net(x)

# class LineupOptimizer:
#     def __init__(self):
#         # Initialize the model
#         state_dim = 30  # 6 stats × 5 players
#         action_dim = 9  # Number of possible actions
#         self.model = QNetwork(state_dim, action_dim)
        
#         # Load the trained model
#         model_path = os.path.join(project_root, "Models", "q_learning_lineup_model_knicks_final.pth")
#         checkpoint = torch.load(model_path)
#         self.model.load_state_dict(checkpoint['q_network_state_dict'])
#         self.model.eval()

#     def get_team_players(self, team: str) -> List[Dict[str, Any]]:
#         """Get all players from a specific team."""
#         try:
#             response = supabase.table("AdvancedStats24New").select("*").eq("TEAM", team).execute()
#             return response.data
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error fetching team players: {str(e)}")

#     def optimize_lineup(self, opponent_players: List[str], team: str) -> Dict[str, Any]:
#         """Optimize lineup using RL model."""
#         try:
#             # Get team roster
#             team_roster = self.get_team_players(team)
#             if not team_roster:
#                 raise HTTPException(status_code=404, detail=f"No players found for team {team}")

#             # Get stats for opponent lineup
#             opponent_stats = []
#             missing_players = []
#             for name in opponent_players:
#                 res = supabase.table("AdvancedStats24New").select("*").eq("PLAYER", name).execute()
#                 if res.data:
#                     player = res.data[0]
#                     opponent_stats.append(player)
#                 else:
#                     missing_players.append(name)
#                     opponent_stats.append({})

#             # Check if we have enough opponent players
#             valid_players = [p for p in opponent_stats if p]
#             if len(valid_players) < 5:
#                 raise HTTPException(status_code=400, detail=f"Not enough valid opponent players. Missing: {', '.join(missing_players)}")

#             # Track best lineup and score
#             best_lineup = random.sample(team_roster, 5)  # Start with random lineup
            
#             # Get initial score
#             state = []
#             for p in best_lineup:
#                 state.extend([float(p.get(stat, 0)) for stat in STATE_COLUMNS])
#             state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
#             with torch.no_grad():
#                 q_values = self.model(state_tensor)
#                 best_score = q_values.max().item()
            
#             # Try to improve the lineup systematically
#             for _ in range(200):  # 200 iterations
#                 current_lineup = best_lineup.copy()
                
#                 # Get model's suggestion
#                 state = []
#                 for p in current_lineup:
#                     state.extend([float(p.get(stat, 0)) for stat in STATE_COLUMNS])
#                 state_tensor = torch.FloatTensor(state).unsqueeze(0)
                
#                 with torch.no_grad():
#                     q_values = self.model(state_tensor)
#                     best_action = q_values.argmax().item()
                
#                 # Get available players (excluding current lineup)
#                 available_players = [p for p in team_roster if p not in current_lineup]
                
#                 if available_players:
#                     suggested_player = available_players[best_action % len(available_players)]
                    
#                     # Try replacing each current player with the suggested player
#                     for i in range(5):
#                         temp_lineup = current_lineup.copy()
#                         temp_lineup[i] = suggested_player
                        
#                         # Evaluate this potential lineup
#                         temp_state = []
#                         for p in temp_lineup:
#                             temp_state.extend([float(p.get(stat, 0)) for stat in STATE_COLUMNS])
#                         temp_tensor = torch.FloatTensor(temp_state).unsqueeze(0)
                        
#                         with torch.no_grad():
#                             temp_q_values = self.model(temp_tensor)
#                             temp_score = temp_q_values.max().item()
                            
#                             # Update best lineup if this is better
#                             if temp_score > best_score:
#                                 best_score = temp_score
#                                 best_lineup = temp_lineup.copy()

#             return {
#                 "lineup": [player["PLAYER"] for player in best_lineup],
#                 "score": best_score,
#                 "opponent": opponent_players
#             }
            
#         except Exception as e:
#             raise HTTPException(status_code=500, detail=f"Error optimizing lineup: {str(e)}")

#     def test_optimization(self) -> Dict[str, Any]:
#         """Test the optimization with a sample Knicks lineup."""
#         try:
#             # Sample opponent lineup
#             sample_opponent = ["Jayson Tatum", "Jaylen Brown", "Kristaps Porzingis", "Derrick White", "Jrue Holiday"]
            
#             # Optimize against this lineup
#             result = self.optimize_lineup(sample_opponent, "NYK")
            
#             return {
#                 "test_result": "success",
#                 "sample_opponent": sample_opponent,
#                 "optimized_lineup": result["lineup"],
#                 "score": result["score"]
#             }
#         except Exception as e:
#             return {
#                 "test_result": "failure",
#                 "error": str(e)
#             }

# # Create a single instance of the optimizer
# lineup_optimizer = LineupOptimizer()

# @router.post("/optimize")
# async def optimize_lineup(opponent_players: List[str], team: str):
#     """Optimize lineup against opponent players."""
#     print("\nBackend received request:")
#     print(f"Opponent players: {opponent_players}")
#     print(f"Team: {team}")
    
#     result = lineup_optimizer.optimize_lineup(opponent_players, team)
    
#     print("\nBackend sending response:")
#     print(f"Result: {result}")
    
#     return result

# @router.get("/team/{team}")
# async def get_team_players(team: str):
#     """Get all players from a specific team."""
#     return lineup_optimizer.get_team_players(team)

# @router.get("/test")
# async def test_optimization():
#     """Test the lineup optimization with a sample case."""
#     return lineup_optimizer.test_optimization() 