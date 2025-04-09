import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY

# Supabase config
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Stats to use per player
KEY_STATS = ['PIE', 'NETRTG', 'MIN', 'USG%', 'TS%', 'AST%']

# Fetch player stats from Supabase
def get_player_stats():
    response = supabase.table("AdvancedStats24New").select("*").execute()
    stats = {}
    for row in response.data:
        try:
            stats[row['PLAYER']] = [float(row.get(stat, 0)) for stat in KEY_STATS]
        except:
            continue
    return stats

# Fetch and prepare lineup data
def prepare_data(player_stats):
    response = supabase.table("lineup_stats").select("*").eq("year", "2024").execute()
    X, y = [], []
    for row in response.data:
        if row['minutes'] < 40 or len(row['lineup']) != 5:
            continue
        try:
            features = []
            for p in row['lineup']:
                features.extend(player_stats[p])
            X.append(features)
            weight = np.sqrt(row['minutes'] / 50)
            y.append(row['net_rating'] * weight)
        except:
            continue
    return np.array(X), np.array(y)

# Dataset wrapper
class LineupDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

# Simple feedforward model
class Net(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze()

# Train the model
def train():
    player_stats = get_player_stats()
    X, y = prepare_data(player_stats)
    print(f"Total valid lineups: {len(X)}")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    model = Net(X.shape[1])
    loader = DataLoader(LineupDataset(X_train, y_train), batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()

    for epoch in range(100):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        
        if (epoch+1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                # Training metrics
                train_preds = model(torch.FloatTensor(X_train))
                train_r2 = r2_score(y_train, train_preds.numpy())
                train_rmse = np.sqrt(mean_squared_error(y_train, train_preds.numpy()))
                
                # Test metrics
                test_preds = model(torch.FloatTensor(X_test))
                test_r2 = r2_score(y_test, test_preds.numpy())
                test_rmse = np.sqrt(mean_squared_error(y_test, test_preds.numpy()))
                
                print(f"Epoch {epoch+1}")
                print(f"Training - R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}")
                print(f"Test     - R²: {test_r2:.4f}, RMSE: {test_rmse:.2f}")
                print("-" * 50)
    
    # Save the trained model
    torch.save(model.state_dict(), 'lineup_critic_modelNN.pth')
    print("Model saved to lineup_critic_modelNN.pth")
    
    return model, X.shape[1]  # Return model and input size for later use

if __name__ == "__main__":
    train() 