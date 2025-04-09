import torch
import itertools
import numpy as np
from supabase import create_client
from collections import Counter
import random

# Supabase setup
SUPABASE_URL = "https://gljggtstugjvekcnncys.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdsamdndHN0dWdqdmVrY25uY3lzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDI4NjA5MzIsImV4cCI6MjA1ODQzNjkzMn0.QHy7TUnFxGTaP5VkByyomYhju-FAoCcX8O6gfREbCo4"
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Columns used in state
STATE_COLUMNS = ['PIE', 'NETRTG', 'MIN', 'USG%', 'TS%', 'AST%']

# Load critic model
class CriticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(30, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze()

critic = CriticModel()
critic.load_state_dict(torch.load("lineup_critic_modelNN.pth"))
critic.eval()

# Helper to convert a 5-player lineup to input vector
def encode_lineup(players):
    vec = []
    for p in players:
        vec.extend([float(p.get(stat, 0)) for stat in STATE_COLUMNS])
    return torch.FloatTensor(vec).unsqueeze(0)

# Load Knicks roster
res = supabase.table("AdvancedStats24New").select("*").eq("TEAM", "NYK").execute()
knicks_roster = res.data
print(f"\nLoaded {len(knicks_roster)} Knicks players.")

# Define the preferred lineup (the one we want to prioritize)
preferred_lineup_names = ["Karl-Anthony Towns", "Jalen Brunson", "Mitchell Robinson", "Josh Hart", "OG Anunoby"]
preferred_lineup = []

# Find the player objects for the preferred lineup
for name in preferred_lineup_names:
    for player in knicks_roster:
        if player["PLAYER"] == name:
            preferred_lineup.append(player)
            break

# If we couldn't find all players, use the first 5 players from the roster
if len(preferred_lineup) < 5:
    print("Warning: Could not find all preferred players. Using first 5 players from roster.")
    preferred_lineup = knicks_roster[:5]

print("\n🟦 Preferred Knicks Lineup:")
for p in preferred_lineup:
    print(f" - {p['PLAYER']}")

# Sort players by PIE (Player Impact Estimate) to identify star and role players
sorted_roster = sorted(knicks_roster, key=lambda x: float(x.get('PIE', 0)), reverse=True)
star_players = sorted_roster[:5]  # Top 5 players by PIE
role_players = sorted_roster[5:]  # Rest are role players

print("\n⭐ Star Players:")
for p in star_players:
    print(f" - {p['PLAYER']} (PIE: {p.get('PIE', 'N/A')})")

print("\n👥 Role Players:")
for p in role_players:
    print(f" - {p['PLAYER']} (PIE: {p.get('PIE', 'N/A')})")

# Load all matchups
matchups = supabase.table("matchups").select("lineup_b").execute()
print(f"Loaded {len(matchups.data)} opponent lineups")

# Initialize counters for tracking distributions
lineup_distribution = Counter()
player_frequency = Counter()
best_lineups = []

# Number of opponent lineups to analyze
num_opponents = min(100, len(matchups.data))
print(f"\nAnalyzing Knicks lineups against {num_opponents} different opponents...")

# Function to evaluate a lineup
def evaluate_lineup(lineup, opponent_tensor):
    input_tensor = encode_lineup(lineup)
    with torch.no_grad():
        own_score = critic(input_tensor).item()
        opp_score = critic(opponent_tensor).item()
        return own_score - opp_score

# Function to get a slightly modified version of the preferred lineup
def get_modified_preferred_lineup(roster, num_changes=1):
    # Start with the preferred lineup
    modified_lineup = preferred_lineup.copy()
    
    # Randomly select players to replace
    indices_to_replace = random.sample(range(5), num_changes)
    
    # Get available players not in the lineup
    available_players = [p for p in roster if p not in modified_lineup]
    
    # Replace selected players with random ones
    for idx in indices_to_replace:
        if available_players:
            new_player = random.choice(available_players)
            modified_lineup[idx] = new_player
            available_players.remove(new_player)
    
    return modified_lineup

# Function to create a lineup with rest management (including some role players)
def get_rest_management_lineup(num_stars=2, num_role_players=3):
    # Select random star players
    selected_stars = random.sample(star_players, num_stars)
    
    # Select random role players
    selected_role_players = random.sample(role_players, num_role_players)
    
    # Combine them into a lineup
    lineup = selected_stars + selected_role_players
    
    # Shuffle the lineup to mix stars and role players
    random.shuffle(lineup)
    
    return lineup

# Loop through different opponent lineups
for i in range(num_opponents):
    opponent_names = matchups.data[i]["lineup_b"]
    
    # Get stats for opponent lineup
    opponent_stats = []
    for name in opponent_names:
        res = supabase.table("AdvancedStats24New").select("*").eq("PLAYER", name).execute()
        player = res.data[0] if res.data else {}
        opponent_stats.append(player)
    
    if len(opponent_stats) != 5:
        print(f"Skipping opponent {i} - incomplete lineup data")
        continue
        
    opponent_tensor = encode_lineup(opponent_stats)
    
    # Evaluate the opponent's strength
    opponent_strength = evaluate_lineup(opponent_stats, opponent_tensor)
    
    # Find best Knicks lineup against this opponent
    best_lineup = None
    best_score = -float("inf")
    
    # Always try the preferred lineup first
    preferred_score = evaluate_lineup(preferred_lineup, opponent_tensor)
    if preferred_score > best_score:
        best_score = preferred_score
        best_lineup = preferred_lineup
    
    # Determine how much randomness to introduce based on opponent strength
    # For strong opponents (high net rating), use less randomness
    if opponent_strength > 5.0:  # Strong opponent
        # Try only a few variations of the preferred lineup
        for _ in range(3):
            modified_lineup = get_modified_preferred_lineup(knicks_roster, num_changes=1)
            score = evaluate_lineup(modified_lineup, opponent_tensor)
            if score > best_score:
                best_score = score
                best_lineup = modified_lineup
    else:  # Weaker opponent
        # Try more variations and some completely random lineups
        # Try variations of the preferred lineup
        for _ in range(3):
            modified_lineup = get_modified_preferred_lineup(knicks_roster, num_changes=random.randint(1, 2))
            score = evaluate_lineup(modified_lineup, opponent_tensor)
            if score > best_score:
                best_score = score
                best_lineup = modified_lineup
        
        # Try some completely random lineups
        for _ in range(2):
            random_lineup = random.sample(knicks_roster, 5)
            score = evaluate_lineup(random_lineup, opponent_tensor)
            if score > best_score:
                best_score = score
                best_lineup = random_lineup
        
        # Try rest management lineups (with some role players)
        for _ in range(5):
            # For very weak opponents, use more role players
            if opponent_strength < 0:
                num_stars = random.randint(1, 2)  # 1-2 star players
                num_role_players = 5 - num_stars  # 3-4 role players
            else:
                num_stars = random.randint(2, 3)  # 2-3 star players
                num_role_players = 5 - num_stars  # 2-3 role players
                
            rest_lineup = get_rest_management_lineup(num_stars, num_role_players)
            score = evaluate_lineup(rest_lineup, opponent_tensor)
            
            # For rest management lineups, we need a minimum score threshold
            # to ensure we're not sacrificing too much performance
            min_acceptable_score = -2.0  # Adjust this threshold as needed
            
            if score > best_score and score > min_acceptable_score:
                best_score = score
                best_lineup = rest_lineup
                print(f"Found rest management lineup with score {score:.2f} against opponent {i}")
    
    # Track the best lineup for this opponent
    if best_lineup:
        lineup_names = tuple(sorted([p["PLAYER"] for p in best_lineup]))
        lineup_distribution[lineup_names] += 1
        
        # Track individual player frequency
        for player in best_lineup:
            player_frequency[player["PLAYER"]] += 1
            
        # Store best lineup and score
        best_lineups.append((best_lineup, best_score))
        
        # Print progress every 10 opponents
        if (i+1) % 10 == 0:
            print(f"Processed {i+1}/{num_opponents} opponents")

# Print overall distribution of best Knicks lineups
print("\n📊 Distribution of Best Knicks Lineups (Top 10 most common):")
for lineup, count in lineup_distribution.most_common(10):
    print(f"Count: {count} - Lineup: {', '.join(lineup)}")

# Print player frequency
print("\n👤 Player Frequency in Best Lineups:")
for player, freq in player_frequency.most_common():
    print(f"{player}: {freq} appearances")

# Print top 5 best lineups overall
print("\n🏆 Top 5 Best Knicks Lineups (Across All Opponents):")
best_lineups.sort(key=lambda x: x[1], reverse=True)
for i, (lineup, score) in enumerate(best_lineups[:5]):
    print(f"\n{i+1}. Net Rating Difference: {score:.2f}")
    for p in lineup:
        print(f"   - {p['PLAYER']}")

# Print rest management lineups (those with 3 or more role players)
print("\n🔄 Rest Management Lineups (3+ Role Players):")
rest_lineups = []
for lineup, score in best_lineups:
    role_player_count = sum(1 for p in lineup if p in role_players)
    if role_player_count >= 3:
        rest_lineups.append((lineup, score, role_player_count))

# Sort by score
rest_lineups.sort(key=lambda x: x[1], reverse=True)

# Display top 5 rest management lineups
for i, (lineup, score, role_count) in enumerate(rest_lineups[:5]):
    print(f"\n{i+1}. Net Rating Difference: {score:.2f} (Role Players: {role_count})")
    for p in lineup:
        player_type = "⭐" if p in star_players else "👥"
        print(f"   {player_type} {p['PLAYER']}")
