import sys
import os
from collections import Counter
from typing import List

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up one more level
sys.path.append(project_root)

from server.api.routes.lineup_routes import LineupOptimizer

def get_sample_opponent_lineups() -> List[List[str]]:
    """Get a list of sample opponent lineups to test against."""
    return [
        # Lakers
        ["LeBron James", "Anthony Davis", "D'Angelo Russell", "Austin Reaves", "Rui Hachimura"],
        
        # Suns
        ["Devin Booker", "Kevin Durant", "Bradley Beal", "Jusuf Nurkic", "Grayson Allen"],
        
        # Nuggets
        ["Nikola Jokic", "Jamal Murray", "Michael Porter Jr.", "Aaron Gordon", "Kentavious Caldwell-Pope"],
        
        # Clippers
        ["Kawhi Leonard", "Paul George", "James Harden", "Russell Westbrook", "Ivica Zubac"]
    ]

def analyze_lineup_distribution():
    """Analyze lineup distribution against multiple opponent lineups."""
    print("Starting Warriors lineup distribution analysis...")
    
    # Initialize the optimizer
    optimizer = LineupOptimizer()
    
    # Get sample opponent lineups
    opponent_lineups = get_sample_opponent_lineups()
    
    # Track results for each opponent lineup
    all_results = []
    
    # Test against each opponent lineup
    for opponent_lineup in opponent_lineups:
        print(f"\nTesting against: {', '.join(opponent_lineup)}")
        
        # Get multiple optimized lineups
        result = optimizer.optimize_lineup(opponent_lineup, "GSW")
        
        # Store the results
        all_results.append({
            "opponent": opponent_lineup,
            "lineups": result["lineups"]
        })
        
        # Print the results
        for lineup_data in result["lineups"]:
            print(f"\nLineup: {', '.join(lineup_data['lineup'])}")
    
    # Save results to a file
    filename = "warriors_lineup_analysis.txt"
    with open(filename, "w") as f:
        f.write("Warriors Lineup Analysis\n")
        f.write("======================\n\n")
        
        for result in all_results:
            f.write(f"Against: {', '.join(result['opponent'])}\n")
            f.write("-" * 50 + "\n")
            
            for lineup_data in result["lineups"]:
                f.write(f"Lineup: {', '.join(lineup_data['lineup'])}\n")
            
            f.write("\n")
    
    print(f"\nDetailed results saved to {filename}")

if __name__ == "__main__":
    print("Starting lineup distribution analysis...")
    analyze_lineup_distribution() 