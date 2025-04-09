import torch
import numpy as np
from supabase import create_client
import random
import sys
import os

# Add the directory containing the RL model to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Define a model class that matches the saved model architecture
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(30, 128),  # Changed from 64 to 128
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),  # Changed from 32 to 64
            torch.nn.ReLU(),
            torch.nn.Linear(64, 9)     # Changed from 1 to 9
        )
    def forward(self, x):
        return self.net(x).squeeze()

# Supabase setup
SUPABASE_URL = "https://gljggtstugjvekcnncys.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdsamdndHN0dWdqdmVrY25uY3lzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDI4NjA5MzIsImV4cCI6MjA1ODQzNjkzMn0.QHy7TUnFxGTaP5VkByyomYhju-FAoCcX8O6gfREbCo4"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Columns used in state
STATE_COLUMNS = ['PIE', 'NETRTG', 'MIN', 'USG%', 'TS%', 'AST%']

def main():
    print("=== Knicks Lineup Optimizer using RL Model ===")
    
    # Step 1: Load the RL model
    print("\n1. Loading RL model...")
    try:
        # Create a model instance
        model = SimpleModel()
        
        # Load the state dict
        model_path = "/Users/roshangopal/Desktop/RLForFlirting/q_learning_lineup_model_knicks_final.pth"
        saved_data = torch.load(model_path)
        
        # Check if it's a state dict or a complete model
        if isinstance(saved_data, dict):
            # It's a state dict, check if it contains Q-network
            if "q_network_state_dict" in saved_data:
                # Extract the Q-network state dict
                q_network_state_dict = saved_data["q_network_state_dict"]
                
                # Try to load it into our model
                try:
                    model.load_state_dict(q_network_state_dict)
                    print("✅ Loaded Q-network state dict successfully!")
                except Exception as e:
                    print(f"❌ Error loading Q-network state dict: {e}")
                    print("Trying to create a new model with the state dict...")
                    
                    # If loading fails, create a new model with the state dict
                    model = SimpleModel()
                    model.load_state_dict(q_network_state_dict)
            else:
                # Try to load the state dict directly
                try:
                    model.load_state_dict(saved_data)
                    print("✅ Loaded state dict successfully!")
                except Exception as e:
                    print(f"❌ Error loading state dict: {e}")
                    print("Creating a new model with the state dict...")
                    model = SimpleModel()
                    model.load_state_dict(saved_data)
        else:
            # It's a complete model, use it directly
            model = saved_data
            
        # Set to evaluation mode
        model.eval()
        print("✅ RL model loaded successfully!")
        
        # Verify the model is callable
        test_input = torch.randn(1, 30)
        with torch.no_grad():
            test_output = model(test_input)
        print(f"✅ Model test successful! Output shape: {test_output.shape}")
        
    except Exception as e:
        print(f"❌ Error loading RL model: {e}")
        return
    
    # Step 2: Load Knicks roster
    print("\n2. Loading Knicks roster...")
    try:
        res = supabase.table("AdvancedStats24New").select("*").eq("TEAM", "NYK").execute()
        knicks_roster = res.data
        print(f"✅ Loaded {len(knicks_roster)} Knicks players.")
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
        
        # Convert opponent lineup to state tensor
        opponent_state = []
        for p in opponent_stats:
            opponent_state.extend([float(p.get(stat, 0)) for stat in STATE_COLUMNS])
        opponent_tensor = torch.FloatTensor(opponent_state).unsqueeze(0)
        
        # Find best Knicks lineup against this opponent
        best_lineup = None
        best_score = -float("inf")
        
        # Try some random lineups
        for _ in range(20):
            random_lineup = random.sample(knicks_roster, 5)
            
            # Convert lineup to state tensor
            lineup_state = []
            for p in random_lineup:
                lineup_state.extend([float(p.get(stat, 0)) for stat in STATE_COLUMNS])
            lineup_tensor = torch.FloatTensor(lineup_state).unsqueeze(0)
            
            # Get score from the model
            with torch.no_grad():
                try:
                    # The model outputs 9 values, we'll use the first one as the score
                    score = model(lineup_tensor)[0].item()
                except Exception as e:
                    print(f"Error evaluating lineup: {e}")
                    continue
            
            if score > best_score:
                best_score = score
                best_lineup = random_lineup
        
        # Store the counter lineup
        if best_lineup:
            print(f"Best Lineup: {best_lineup}")
            print(f"Opponent Stats: {opponent_stats}")
            
            # Create a safe version of the opponent lineup with placeholders for missing players
            opponent_lineup = []
            for j, p in enumerate(opponent_stats):
                if p and "PLAYER" in p:
                    opponent_lineup.append(p["PLAYER"])
                else:
                    opponent_lineup.append(f"Unknown Player {j+1}")
            
            counter_lineups.append({
                "opponent": opponent_lineup,
                "counter": [p["PLAYER"] for p in best_lineup],
                "score": best_score
            })
            
            # Track lineup distribution
            lineup_key = " - ".join([p["PLAYER"] for p in best_lineup])
            if lineup_key in lineup_distribution:
                lineup_distribution[lineup_key] += 1
            else:
                lineup_distribution[lineup_key] = 1
        
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

if __name__ == "__main__":
    main()
