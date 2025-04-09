# NBA Lineup Optimizer

A Python-based tool that uses machine learning to optimize NBA team lineups. The system evaluates different player combinations against various opponent lineups to find the most effective lineup configurations.

## Features

- Evaluates lineups using a neural network model
- Considers player statistics like PIE, NETRTG, MIN, USG%, TS%, and AST%
- Supports rest management by balancing star players and role players
- Analyzes performance against multiple opponent lineups
- Provides detailed lineup recommendations with performance metrics

## Setup

1. Clone the repository
2. Install dependencies:
```bash
pip install torch numpy supabase
```
3. Create a `config.py` file with your Supabase credentials:
```python
SUPABASE_URL = "your_supabase_url"
SUPABASE_KEY = "your_supabase_key"
```

## Usage

Run the optimizer:
```bash
python Test_Knicks_Lineup_Optimizer.py
```

The script will:
- Load the Knicks roster
- Analyze different lineup combinations
- Output optimal lineups for various scenarios
- Provide rest management recommendations

## Requirements

- Python 3.x
- PyTorch
- NumPy
- Supabase Python Client 