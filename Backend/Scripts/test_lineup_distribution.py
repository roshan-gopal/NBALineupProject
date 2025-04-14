import sys
import os
from collections import Counter
from typing import List, Dict
import json
from datetime import datetime

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from Backend.server.api.routes.lineup_routes import lineup_optimizer

def get_sample_opponent_lineups() -> List[List[str]]:
    """Return 10 different sample opponent lineups."""
    return [
        ["Jayson Tatum", "Jaylen Brown", "Kristaps Porzingis", "Derrick White", "Jrue Holiday"],  # Celtics
        ["Joel Embiid", "Tyrese Maxey", "Tobias Harris", "Kelly Oubre Jr.", "De'Anthony Melton"],  # 76ers
        ["Terry Rozier", "Bam Adebayo", "Tyler Herro", "Duncan Robinson", "Kyle Lowry"],  # Heat
        ["Giannis Antetokounmpo", "Damian Lillard", "Khris Middleton", "Brook Lopez", "Malik Beasley"],  # Bucks
        ["Donovan Mitchell", "Darius Garland", "Evan Mobley", "Jarrett Allen", "Max Strus"],  # Cavaliers
        ["Paolo Banchero", "Franz Wagner", "Jalen Suggs", "Wendell Carter Jr.", "Gary Harris"],  # Magic
        ["Trae Young", "Dejounte Murray", "De'Andre Hunter", "Jalen Johnson", "Clint Capela"],  # Hawks
        ["LaMelo Ball", "Brandon Miller", "Miles Bridges", "P.J. Washington", "Mark Williams"],  # Hornets
        ["Pascal Siakam", "Tyrese Haliburton", "Myles Turner", "Buddy Hield", "Bruce Brown"],  # Pacers
        ["Bradley Beal", "Devin Booker", "Kevin Durant", "Jusuf Nurkic", "Grayson Allen"]  # Suns
    ]

def analyze_lineup_distribution():
    """Generate and analyze lineup distributions."""
    opponent_lineups = get_sample_opponent_lineups()
    results = {}
    
    print("\nGenerating lineups...")
    for i, opponent_lineup in enumerate(opponent_lineups, 1):
        print(f"\nOpponent Lineup {i}: {', '.join(opponent_lineup)}")
        
        # Get 10 lineups for this opponent
        warriors_lineups = []
        lineup_counter = Counter()
        
        for j in range(10):
            result = lineup_optimizer.optimize_lineup(opponent_lineup, "GSW")
            # Extract lineup from the result - it's in result["lineups"][0]["lineup"]
            if result.get("lineups") and len(result["lineups"]) > 0:
                lineup = result["lineups"][0]["lineup"]
                score = result["lineups"][0]["score"]
                lineup_tuple = tuple(sorted(lineup))  # Sort to ensure same lineup in different orders counts as same
                lineup_counter[lineup_tuple] += 1
                warriors_lineups.append({
                    "lineup": lineup,
                    "score": score
                })
        
        # Analyze results for this opponent
        unique_lineups = len(lineup_counter)
        most_common = lineup_counter.most_common()
        
        print(f"Generated 10 lineups:")
        print(f"Number of unique lineups: {unique_lineups}")
        print("\nLineup distribution:")
        for lineup, count in most_common:
            print(f"Count: {count} - {', '.join(lineup)}")
        
        # Store results
        results[f"opponent_{i}"] = {
            "opponent_lineup": opponent_lineup,
            "warriors_lineups": warriors_lineups,
            "unique_lineups": unique_lineups,
            "distribution": [{
                "lineup": list(lineup),
                "count": count
            } for lineup, count in most_common]
        }
    
    # Save results to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"lineup_distribution_results_GSW_{timestamp}.json"
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to {filename}")

if __name__ == "__main__":
    print("Starting lineup distribution analysis for Warriors...")
    analyze_lineup_distribution() 