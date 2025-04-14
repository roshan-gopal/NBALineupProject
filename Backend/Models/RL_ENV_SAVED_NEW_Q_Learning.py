import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from supabase import create_client
import gym
import sys
import os

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from config import SUPABASE_URL, SUPABASE_KEY

# Define database column metadata
PLAYER_COLUMNS = [
    'id', 'PLAYER', 'TEAM',  # Keep identification columns
    'PIE', 'NETRTG', 'MIN', 'USG%', 'TS%', 'AST%'  # Advanced stats we need
]

# Stats used for state representation
STATE_COLUMNS = [
    'PIE', 'NETRTG', 'MIN', 'USG%', 'TS%', 'AST%'
]

# Load environment variables from .env file
def get_supabase_client():
    """
    Create and return a Supabase client instance.
    
    Returns:
        A configured Supabase client
    """
    # Get Supabase URL and API key from environment variables
    supabase_url = SUPABASE_URL
    supabase_key = SUPABASE_KEY
    
    # Validate that the required environment variables are set
    if not supabase_url or not supabase_key:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
    
    # Create and return the client
    return create_client(supabase_url, supabase_key)

def build_target_lineup(season='2024-25'):
    """Build a lineup by randomly selecting a lineup from the matchups table."""
    
    # Get Supabase client
    supabase = get_supabase_client()
    
    # Query the matchups table to get all lineups in lineup_b
    response = supabase.table('matchups').select('lineup_b').execute()
    
    if not response.data:
        # If no lineups found, get random players from the database
        response = supabase.table('AdvancedStats24New').select('PLAYER').execute()
        if response.data:
            all_players = [player['PLAYER'] for player in response.data]
            # Select 5 random players
            return random.sample(all_players, min(5, len(all_players)))
        else:
            return []
    
    # Extract all lineups from lineup_b
    all_lineups_b = [matchup['lineup_b'] for matchup in response.data]
    
    # Randomly select a lineup from lineup_b
    selected_lineup = random.choice(all_lineups_b)
    
    return selected_lineup

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, state):
        return self.net(state)

# Define the Q-Learning Agent
class QLearningAgent:
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99, epsilon=0.3, epsilon_min=0.01, epsilon_decay=0.995):
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Initial exploration rate
        self.epsilon_min = epsilon_min  # Minimum exploration rate
        self.epsilon_decay = epsilon_decay  # Exploration decay rate
        self.action_dim = action_dim
        
        # For tracking metrics
        self.total_reward = 0
        self.episode_rewards = []
        self.best_reward = float('-inf')
        self.patience = 20  # Number of episodes to wait for improvement
        self.patience_counter = 0
        self.early_stop = False
    
    def select_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()
    
    def update(self, state, action, reward, next_state, done):
        # Convert to tensors
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        
        # Get current Q values
        current_q_values = self.q_network(state_tensor)
        
        # Get next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensor)
            max_next_q = next_q_values.max(1)[0]
            target_q_values = current_q_values.clone()
            
            # Q-learning update formula
            if done:
                target_q_values[0, action] = reward
            else:
                target_q_values[0, action] = reward + self.gamma * max_next_q
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        if random.random() < 0.01:  # 1% chance to update target network
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def update_epsilon(self):
        """Decay epsilon over time to reduce exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def check_early_stopping(self, episode_reward):
        """Check if training should be stopped early."""
        # Update best reward if current episode is better
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.patience_counter = 0
            return False
        
        # Increment patience counter
        self.patience_counter += 1
        
        # Check if we've waited long enough without improvement
        if self.patience_counter >= self.patience:
            self.early_stop = True
            return True
        
        return False
    
    def save(self, path):
        """Save the Q-network to the specified path."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'best_reward': self.best_reward,
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load the Q-network from the specified path."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.best_reward = checkpoint.get('best_reward', self.best_reward)
        print(f"Model loaded from {path}")

# Define the Lineup Environment
class LineupEnv(gym.Env):
    def __init__(self, team_name=None):
        super(LineupEnv, self).__init__()
        self.supabase = get_supabase_client()
        self.team_name = team_name  # Store the team name
        self.roster = None
        self.current_lineup = []
        self.opponent_lineup = None  # Will be set in reset()

        # Load the critic model
        self.critic_model = self.load_critic_model()

        # Load an initial roster to define spaces
        self.load_new_roster()

    def load_critic_model(self):
        """Load the trained critic model for lineup evaluation."""
        # Define the critic model architecture
        class CriticModel(nn.Module):
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
        
        # Calculate input size based on state columns
        input_size = len(STATE_COLUMNS) * 5  # 5 players
        
        # Create and load the model
        model = CriticModel(input_size)
        try:
            model.load_state_dict(torch.load('lineup_critic_modelNN.pth'))
            print("Critic model loaded successfully")
        except Exception as e:
            print(f"Error loading critic model: {e}")
            print("Using untrained critic model")
        
        model.eval()  # Set to evaluation mode
        return model

    def load_new_roster(self):
        """Load a new roster from the database."""
        # Get players from AdvancedStats24New table
        if self.team_name:
            # If a team is specified, filter players by that team
            response = self.supabase.table('AdvancedStats24New').select('PLAYER,TEAM,PIE,NETRTG,MIN,"USG%","TS%","AST%"').eq('TEAM', self.team_name).execute()
        else:
            # Otherwise, load all players
            response = self.supabase.table('AdvancedStats24New').select('PLAYER,TEAM,PIE,NETRTG,MIN,"USG%","TS%","AST%"').execute()
        
        if not response.data:
            raise ValueError("No players found in the database")
        
        # Convert to list of dictionaries
        self.roster = response.data
        
        # Update action space to match roster size
        self.action_space = gym.spaces.Discrete(len(self.roster))
        
        # Reset the environment
        self.reset()

    def reset(self):
        """Reset the environment to initial state."""
        # Select 5 random players from the roster
        self.current_lineup = random.sample(self.roster, min(5, len(self.roster)))
        
        # Get a random opponent lineup
        self.opponent_lineup = build_target_lineup()
        
        # Return the initial state
        return self._get_state()

    def _get_state(self):
        """Convert the current lineup to a state vector."""
        state = []
        for player in self.current_lineup:
            # Extract stats from the player dictionary
            player_stats = []
            for stat in STATE_COLUMNS:
                player_stats.append(float(player.get(stat, 0)))
            state.extend(player_stats)
        # Convert list to numpy array before returning
        return np.array(state, dtype=np.float32)

    def step(self, action):
        """Take a step in the environment."""
        # Action is the index of the player to replace
        if action >= len(self.roster):
            print(f"Invalid action: {action} (roster size: {len(self.roster)})")
            return self._get_state(), 0, True, {}  # End the episode or skip the action
        
        # Replace a random player in the current lineup with the selected player
        replace_idx = random.randint(0, 4)
        self.current_lineup[replace_idx] = self.roster[action]
        
        # Calculate reward
        reward = self.calculate_lineup_reward(self.current_lineup, self.opponent_lineup)
        
        # Get new state
        state = self._get_state()
        
        # Episode is done after a certain number of steps
        done = False  # We'll let the training loop control episode length
        
        return state, reward, done, {}

    def calculate_lineup_reward(self, current_lineup, opponent_lineup):
        # Convert lineups to feature vectors for the critic model
        current_features = []
        for player in current_lineup:
            # Extract stats from the player dictionary
            player_stats = []
            for stat in STATE_COLUMNS:
                player_stats.append(float(player.get(stat, 0)))
            current_features.extend(player_stats)
        
        opponent_features = []
        for player_name in opponent_lineup:
            # Get player stats from database
            response = self.supabase.table('AdvancedStats24New').select('PIE,NETRTG,MIN,"USG%","TS%","AST%"').eq('PLAYER', player_name).execute()
            if response.data:
                player_data = response.data[0]
                player_stats = []
                for stat in STATE_COLUMNS:
                    player_stats.append(float(player_data.get(stat, 0)))
            else:
                # If player not found, use zeros
                player_stats = [0.0] * len(STATE_COLUMNS)
            opponent_features.extend(player_stats)
        
        # Convert to tensors for the critic model
        current_tensor = torch.FloatTensor(current_features).unsqueeze(0)
        opponent_tensor = torch.FloatTensor(opponent_features).unsqueeze(0)
        
        # Get predicted net ratings from critic model
        with torch.no_grad():
            current_rating = self.critic_model(current_tensor).item()
            opponent_rating = self.critic_model(opponent_tensor).item()
        
        # Calculate reward
        reward = current_rating - opponent_rating
        
        return reward

# Main Training Loop
# Train exclusively on the New York Knicks
team_name = "NYK"  # New York Knicks
print(f"Training exclusively on the {team_name}")

# Create the Q-learning agent with a fixed state dimension
state_dim = len(STATE_COLUMNS) * 5  # 5 players
action_dim = 30  # Maximum expected roster size

# Initialize the agent with fixed dimensions
agent = QLearningAgent(state_dim=state_dim, action_dim=action_dim)

# Training parameters
num_episodes = 300  # Reduced from 1000 to 300
max_steps_per_episode = 50
save_interval = 50  # Save model every 50 episodes
opponent_change_frequency = 50  # Change opponent every 50 steps (entire episode)

# Create environment for the Knicks
env = LineupEnv(team_name=team_name)

# Update action space to match the Knicks roster size
roster_size = len(env.roster)
print(f"Knicks roster size: {roster_size} players")

# Create an agent with the correct action space for the Knicks
agent = QLearningAgent(state_dim=state_dim, action_dim=roster_size)

# Training loop
for episode in range(num_episodes):
    print(f"Epoch {episode+1}/{num_episodes} - Team: {team_name} - Epsilon: {agent.epsilon:.4f}")
    
    # Select an opponent lineup and stay with it for the entire episode
    opponent_lineup = build_target_lineup()
    print(f"Selected opponent lineup: {opponent_lineup}")
    
    # Reset environment with the same opponent lineup
    state = env.reset()
    env.opponent_lineup = opponent_lineup  # Ensure we use the same opponent
    
    episode_reward = 0
    
    for step in range(max_steps_per_episode):
        # Get action from agent
        action = agent.select_action(state)
        
        # Take action in environment
        next_state, reward, done, _ = env.step(action)
        
        # Update agent after each substitution (single-step update)
        loss = agent.update(state, action, reward, next_state, done)
        
        # Update state and reward
        state = next_state
        episode_reward += reward
        
        if done:
            break
    
    # Decay epsilon after each episode
    agent.update_epsilon()
    
    # Check for early stopping
    if agent.check_early_stopping(episode_reward):
        print(f"Early stopping triggered at episode {episode+1}")
        break
    
    # Print episode results
    print(f"Epoch {episode+1}/{num_episodes}, Team: {team_name}, Reward: {episode_reward:.2f}, Best: {agent.best_reward:.2f}")
    
    # Save model periodically
    if (episode + 1) % save_interval == 0:
        # Save the model
        model_save_path = f"q_learning_lineup_model_knicks_episode_{episode+1}.pth"
        agent.save(model_save_path)
        print(f"Model saved to {model_save_path}")

# Save the final model
final_model_path = "q_learning_lineup_model_knicks_final.pth"
agent.save(final_model_path)
print(f"Final model saved to {final_model_path}") 