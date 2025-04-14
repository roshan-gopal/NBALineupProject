from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
import torch
import random
from supabase import create_client
import os
import sys
from collections import Counter

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(project_root)

from config import SUPABASE_URL, SUPABASE_KEY

router = APIRouter()

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Define state columns
STATE_COLUMNS = ['PIE', 'NETRTG', 'MIN', 'USG%', 'TS%', 'AST%']

# Define the Q-Network
class QNetwork(torch.nn.Module):
    def __init__(self, state_dim, action_dim, output_dim=None):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(30, 128),  # Input layer: 30 features (6 stats × 5 players)
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),  # Hidden layer: 128 → 64 neurons
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim or action_dim)  # Output layer: 64 → output_dim actions
        )
    def forward(self, x):
        return self.net(x)

class LineupOptimizer:
    def __init__(self):
        # Initialize the model
        state_dim = 30  # 6 stats × 5 players
        action_dim = 9  # Default action dim for Knicks model
        self.model = QNetwork(state_dim, action_dim)
        
        # Base path for models
        self.model_base_path = os.path.join(project_root, "Models")
        
        # Load the default model (Knicks)
        model_path = os.path.join(self.model_base_path, "q_learning_lineup_model_knicks_final.pth")
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['q_network_state_dict'])
        self.model.eval()

    def load_model_for_team(self, team: str):
        """Load the appropriate model based on team."""
        if team == "GSW":
            # Warriors model has 10 output dimensions
            print("Loading Warriors model")
            self.model = QNetwork(30, 9, output_dim=10)
            model_name = "q_learning_lineup_model_warriors_final.pth"
        elif team == "LAL":
            # Lakers model has 12 output dimensions
            self.model = QNetwork(30, 9, output_dim=12)
            model_name = "q_learning_lineup_model_lakers_final.pth"
        else:
            # Knicks model has 9 output dimensions
            print("Loading Knicks model")
            self.model = QNetwork(30, 9)
            model_name = "q_learning_lineup_model_knicks_final.pth"
        
        model_path = os.path.join(self.model_base_path, model_name)
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['q_network_state_dict'])
        self.model.eval()

    def get_team_players(self, team: str) -> List[Dict[str, Any]]:
        """Get all players from a specific team."""
        try:
            response = supabase.table("AdvancedStats24New").select("*").eq("TEAM", team).execute()
            return response.data
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error fetching team players: {str(e)}")

    def optimize_lineup(self, opponent_players: List[str], team: str) -> Dict[str, Any]:
        """Optimize lineup using RL model."""
        try:
            # Load the appropriate model
            self.load_model_for_team(team)

            # Get team roster
            team_roster = self.get_team_players(team)
            if not team_roster:
                raise HTTPException(status_code=404, detail=f"No players found for team {team}")

            # Get stats for opponent lineup
            opponent_stats = []
            missing_players = []
            for name in opponent_players:
                res = supabase.table("AdvancedStats24New").select("*").eq("PLAYER", name).execute()
                if res.data:
                    player = res.data[0]
                    opponent_stats.append(player)
                else:
                    missing_players.append(name)
                    opponent_stats.append({})

            # Check if we have enough opponent players
            valid_players = [p for p in opponent_stats if p]
            if len(valid_players) < 5:
                raise HTTPException(status_code=400, detail=f"Not enough valid opponent players. Missing: {', '.join(missing_players)}")

            # Track all lineups and their scores
            lineup_counter = Counter()
            all_scores = {}

            # Run optimization with exploration
            num_attempts = 20
            iterations_per_attempt = 150
            exploration_rate = 0.25 if team == "GSW" else 0.1  # Higher exploration for Warriors

            for _ in range(num_attempts):
                current_lineup = random.sample(team_roster, 5)
                
                for _ in range(iterations_per_attempt):
                    state = []
                    for p in current_lineup:
                        state.extend([float(p.get(stat, 0)) for stat in STATE_COLUMNS])
                    state_tensor = torch.FloatTensor(state).unsqueeze(0)
                    
                    with torch.no_grad():
                        q_values = self.model(state_tensor)
                        
                        # Use epsilon-greedy selection
                        if random.random() < exploration_rate:
                            best_action = random.randint(0, q_values.size(1)-1)
                            current_score = q_values[0, best_action].item()
                        else:
                            best_action = q_values.argmax().item()
                            current_score = q_values.max().item()
                    
                    available_players = [p for p in team_roster if p not in current_lineup]
                    
                    if available_players:
                        # Special handling for Warriors model
                        if team == "GSW" and len(q_values[0]) == 10:
                            player_index = int((best_action / 10) * len(available_players))
                            suggested_player = available_players[min(player_index, len(available_players)-1)]
                        else:
                            suggested_player = available_players[best_action % len(available_players)]
                        
                        # Try replacing each current player
                        best_new_score = current_score
                        best_new_lineup = current_lineup
                        
                        for i in range(5):
                            temp_lineup = current_lineup.copy()
                            temp_lineup[i] = suggested_player
                            
                            temp_state = []
                            for p in temp_lineup:
                                temp_state.extend([float(p.get(stat, 0)) for stat in STATE_COLUMNS])
                            temp_tensor = torch.FloatTensor(temp_state).unsqueeze(0)
                            
                            with torch.no_grad():
                                temp_q_values = self.model(temp_tensor)
                                temp_score = temp_q_values.max().item()
                                
                                if temp_score > best_new_score:
                                    best_new_score = temp_score
                                    best_new_lineup = temp_lineup.copy()
                        
                        if best_new_score > current_score:
                            current_lineup = best_new_lineup
                
                # Track lineup and score
                lineup_tuple = tuple(sorted(player["PLAYER"] for player in current_lineup))
                lineup_counter[lineup_tuple] += 1
                if lineup_tuple not in all_scores or current_score > all_scores[lineup_tuple]:
                    all_scores[lineup_tuple] = current_score

            # Get top 3 lineups
            top_lineups = []
            for lineup_tuple, count in lineup_counter.most_common(3):
                top_lineups.append({
                    "lineup": list(lineup_tuple),
                    "score": all_scores[lineup_tuple],
                    "frequency": count
                })

            return {
                "lineups": top_lineups,
                "opponent": opponent_players
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error optimizing lineup: {str(e)}")

    def test_optimization(self) -> Dict[str, Any]:
        """Test the optimization with a sample Knicks lineup."""
        try:
            # Sample opponent lineup
            sample_opponent = ["Jayson Tatum", "Jaylen Brown", "Kristaps Porzingis", "Derrick White", "Jrue Holiday"]
            
            # Optimize against this lineup
            result = self.optimize_lineup(sample_opponent, "NYK")
            
            return {
                "test_result": "success",
                "sample_opponent": sample_opponent,
                "optimized_lineup": result["lineups"][0]["lineup"],
                "score": result["lineups"][0]["score"]
            }
        except Exception as e:
            return {
                "test_result": "failure",
                "error": str(e)
            }

# Create a single instance of the optimizer
lineup_optimizer = LineupOptimizer()

@router.post("/optimize")
async def optimize_lineup(opponent_players: List[str], team: str):
    """Optimize lineup against opponent players."""
    print("\nBackend received request:")
    print(f"Opponent players: {opponent_players}")
    print(f"Team: {team}")
    
    result = lineup_optimizer.optimize_lineup(opponent_players, team)
    
    print("\nBackend sending response:")
    print(f"Result: {result}")
    
    return result

@router.get("/team/{team}")
async def get_team_players(team: str):
    """Get all players from a specific team."""
    return lineup_optimizer.get_team_players(team)

@router.get("/test")
async def test_optimization():
    """Test the lineup optimization with a sample case."""
    return lineup_optimizer.test_optimization() 