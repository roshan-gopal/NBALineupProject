import torch
import numpy as np
from supabase import create_client
import random
import sys
import os
from collections import Counter
from typing import List

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# Supabase setup
SUPABASE_URL = "https://gljggtstugjvekcnncys.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdsamdndHN0dWdqdmVrY25uY3lzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDI4NjA5MzIsImV4cCI6MjA1ODQzNjkzMn0.QHy7TUnFxGTaP5VkByyomYhju-FAoCcX8O6gfREbCo4"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Columns used in state
STATE_COLUMNS = ['PIE', 'NETRTG', 'MIN', 'USG%', 'TS%', 'AST%']

# Define the Q-Network with correct output dimension
class QNetwork(torch.nn.Module):
    def __init__(self, state_dim=30, output_dim=10):  # Set output_dim to 10 for compatibility
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(30, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def main():
    print("=== Knicks Lineup Optimizer using RL Model ===")
    
    # Step 1: Load the saved model
    print("\n1. Loading saved model...")
    try:
        state_dim = 30  # 6 stats × 5 players
        output_dim = 10  # Match the saved model's output dimension
        model = QNetwork(state_dim=state_dim, output_dim=output_dim)
        model_path = "/Users/roshangopal/Desktop/NBALineupOptimizer/Backend/Models/q_learning_lineup_model_lakers_episode_50.pth"
        
        # Load the checkpoint
        checkpoint = torch.load(model_path)
        # Load the Q-network state dict
        model.load_state_dict(checkpoint['q_network_state_dict'])
        model.eval()
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Step 2: Load Knicks roster
    print("\n2. Loading Knicks roster...")
    try:
        res = supabase.table("AdvancedStats24New").select("*").eq("TEAM", "LAL").execute()
        knicks_roster = res.data
        print(f"✅ Loaded {len(knicks_roster)} LAL players.")
    except Exception as e:
        print(f"❌ Error loading Knicks roster: {e}")
        return
    
    # Step 3: Load opponent lineups from 2024 only
    print("\n3. Loading opponent lineups from 2024...")
    try:
        matchups = supabase.table("lineup_matchups").select("lineup2").eq("year", "2024").execute()
        opponent_lineups = matchups.data[:100]  # Take first 100 lineups
        print(f"✅ Loaded {len(opponent_lineups)} opponent lineups from 2024.")
    except Exception as e:
        print(f"❌ Error loading opponent lineups: {e}")
        return
    
    # Step 4: Find counter lineups
    print("\n4. Finding counter lineups...")
    counter_lineups = []
    lineup_distribution = {}
    player_frequency = {}  # Track how many times each player appears

    for i, matchup in enumerate(opponent_lineups):
        # Split the lineup string into individual player names
        opponent_names = matchup["lineup2"].split(" - ")
        
        # Get stats for opponent lineup
        opponent_stats = []
        missing_players = []
        for name in opponent_names:
            res = supabase.table("AdvancedStats24New").select("*").eq("PLAYER", name).execute()
            if res.data:
                player = res.data[0]
                opponent_stats.append(player)
            else:
                missing_players.append(name)
                # Add an empty dict as a placeholder
                opponent_stats.append({})
        
        # Check if we have enough players
        valid_players = [p for p in opponent_stats if p]
        if len(valid_players) < 5:
            print(f"Skipping opponent {i} - only found {len(valid_players)} players out of 5")
            if missing_players:
                print(f"Missing players: {', '.join(missing_players)}")
            continue
        
        # Find best Knicks lineup against this opponent
        best_lineup = None
        best_score = -float("inf")
        
        # Try some random lineups
        for _ in range(20):
            # Start with a random lineup
            current_lineup = random.sample(knicks_roster, 5)
            
            # Create the state vector
            state = []
            for p in current_lineup:
                state.extend([float(p.get(stat, 0)) for stat in STATE_COLUMNS])
            
            # Convert to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get Q-values from the model
            with torch.no_grad():
                try:
                    q_values = model(state_tensor)
                    best_action = q_values.argmax().item()
                    
                    # Get available players (excluding current lineup)
                    available_players = [p for p in knicks_roster if p not in current_lineup]
                    if available_players:  # Only replace if there are available players
                        replace_idx = random.randint(0, 4)
                        current_lineup[replace_idx] = available_players[best_action % len(available_players)]
                    
                    # Calculate the score for this lineup
                    score = q_values.max().item()
                except Exception as e:
                    print(f"Error evaluating lineup: {e}")
                    continue
            
            if score > best_score:
                best_score = score
                best_lineup = current_lineup
        
        if best_lineup:
            # Store the counter lineup
            counter_lineups.append({
                "opponent": opponent_names,
                "counter": [p["PLAYER"] for p in best_lineup],
                "score": best_score
            })
            
            # Track lineup distribution (sort players to handle different orderings)
            lineup_key = " - ".join(sorted([p["PLAYER"] for p in best_lineup]))
            if lineup_key in lineup_distribution:
                lineup_distribution[lineup_key] += 1
            else:
                lineup_distribution[lineup_key] = 1
                
            # Track player frequency
            for player in best_lineup:
                player_name = player["PLAYER"]
                if player_name in player_frequency:
                    player_frequency[player_name] += 1
                else:
                    player_frequency[player_name] = 1
        
        # Print progress every 10 opponents
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{len(opponent_lineups)} opponents")
    
    # Step 5: Display results
    print("\n5. Results:")
    print(f"Found {len(counter_lineups)} counter lineups.")
    
    # Sort by score
    counter_lineups.sort(key=lambda x: x["score"], reverse=True)
    
    # Display top 10 counter lineups
    print("\nTop 10 Counter Lineups:")
    for i, lineup in enumerate(counter_lineups[:10]):
        print(f"\n{i+1}. Score: {lineup['score']:.2f}")
        print("   Opponent: " + ", ".join(lineup["opponent"]))
        print("   Counter: " + ", ".join(lineup["counter"]))

    # Display lineup distribution
    print("\nLineup Distribution:")
    for lineup, count in sorted(lineup_distribution.items(), key=lambda item: item[1], reverse=True):
        print(f"{lineup}: {count} times")
        
    # Display player frequency
    print("\nPlayer Frequency in Optimal Lineups:")
    for player, count in sorted(player_frequency.items(), key=lambda item: item[1], reverse=True):
        print(f"{player}: {count} appearances")

if __name__ == "__main__":
    main()
