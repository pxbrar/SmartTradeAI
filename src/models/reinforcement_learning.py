"""
Reinforcement Learning for Trading

DQN agent that learns to trade by trial and error.
The agent gets rewards for making profitable trades.
"""

import numpy as np
import pandas as pd
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')

# try to import tensorflow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not installed - RL wont work")


class TradingEnvironment:
    """
    The environment where the agent trades.
    Actions: 0=Sell, 1=Hold, 2=Buy
    """
    
    def __init__(self, df, initial_balance=10000, transaction_cost=0.001, window_size=20):
        self.df = df.copy().reset_index(drop=True)
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost  # 0.1% per trade
        self.window_size = window_size
        
        # features the agent sees
        self.feature_cols = ['daily_return', 'rsi_14', 'macd', 'macd_histogram',
                            'bb_percent', 'volatility_20', 'momentum_10']
        self.available_features = [c for c in self.feature_cols if c in df.columns]
        
        # dimensions for the neural network
        self.state_dim = len(self.available_features) * window_size + 3  # +3 for position info
        self.action_dim = 3  # sell, hold, buy
        
        self.reset()
    
    def reset(self):
        """Start a new episode."""
        self.current_step = self.window_size
        self.cash = self.initial_balance
        self.shares = 0
        self.total_trades = 0
        return self._get_state()
    
    def _get_state(self):
        """Get current state (what the agent sees)."""
        # get recent market data
        start = self.current_step - self.window_size
        features = self.df[self.available_features].iloc[start:self.current_step].values.flatten()
        
        # normalize features
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)
        
        # add info about current position
        current_price = self.df['close'].iloc[self.current_step]
        total_value = self.cash + self.shares * current_price
        
        position_info = np.array([
            self.shares / 100,  # how many shares (normalized)
            self.cash / self.initial_balance,  # how much cash (ratio)
            (total_value - self.initial_balance) / self.initial_balance  # profit/loss
        ])
        
        return np.concatenate([features, position_info])
    
    def step(self, action):
        """Take an action and get reward."""
        current_price = self.df['close'].iloc[self.current_step]
        prev_value = self.cash + self.shares * current_price
        
        # do the action
        if action == 2 and self.cash > 0:  # BUY
            shares_to_buy = (self.cash * (1 - self.transaction_cost)) / current_price
            self.cash -= shares_to_buy * current_price * (1 + self.transaction_cost)
            self.shares += shares_to_buy
            self.total_trades += 1
        elif action == 0 and self.shares > 0:  # SELL
            self.cash += self.shares * current_price * (1 - self.transaction_cost)
            self.shares = 0
            self.total_trades += 1
        # action == 1 is HOLD, do nothing
        
        # move to next day
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1
        
        # calculate reward as percent change in portfolio value
        new_price = self.df['close'].iloc[self.current_step] if not done else current_price
        new_value = self.cash + self.shares * new_price
        reward = (new_value - prev_value) / prev_value * 100
        
        # small penalty for trading (to avoid churning)
        if action != 1:
            reward -= 0.1
        
        state = self._get_state() if not done else np.zeros(self.state_dim)
        info = {'portfolio_value': new_value, 'trades': self.total_trades}
        
        return state, reward, done, info
    
    def get_portfolio_value(self):
        price = self.df['close'].iloc[self.current_step]
        return self.cash + self.shares * price


class ReplayBuffer:
    """
    Memory that stores past experiences.
    We sample random batches from here to train.
    """
    
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent.
    Uses a neural network to estimate Q-values (expected future rewards).
    """
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma  # discount factor for future rewards
        self.epsilon = epsilon  # exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        if TF_AVAILABLE:
            self.model = self._build_network(learning_rate)
            self.target_model = self._build_network(learning_rate)
            self.update_target_network()
        
        self.replay_buffer = ReplayBuffer()
    
    def _build_network(self, lr):
        """Build the Q-network."""
        model = Sequential([
            Dense(128, activation='relu', input_dim=self.state_dim),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_dim, activation='linear')  # Q-value for each action
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model
    
    def update_target_network(self):
        """Copy weights to target network (for stability)."""
        self.target_model.set_weights(self.model.get_weights())
    
    def select_action(self, state, training=True):
        """Pick an action using epsilon-greedy."""
        if training and random.random() < self.epsilon:
            # random action (exploration)
            return random.randint(0, self.action_dim - 1)
        # best action according to model (exploitation)
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        return np.argmax(q_values)
    
    def train_step(self, batch_size=32):
        """Train on a batch from replay buffer."""
        if len(self.replay_buffer) < batch_size:
            return 0
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # calculate target Q-values
        next_q = self.target_model.predict(next_states, verbose=0)
        max_next_q = np.max(next_q, axis=1)
        targets = rewards + self.gamma * max_next_q * (1 - dones)
        
        # update Q-values only for the actions we took
        current_q = self.model.predict(states, verbose=0)
        for i, action in enumerate(actions):
            current_q[i, action] = targets[i]
        
        loss = self.model.train_on_batch(states, current_q)
        
        # decay epsilon (less exploration over time)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return loss
    
    def train(self, env, episodes=100, batch_size=32, target_update_freq=10):
        """Train the agent on the environment."""
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not installed'}
        
        episode_rewards = []
        portfolio_values = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = self.select_action(state, training=True)
                next_state, reward, done, info = env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                self.train_step(batch_size)
                state = next_state
                total_reward += reward
            
            # update target network periodically
            if episode % target_update_freq == 0:
                self.update_target_network()
            
            episode_rewards.append(total_reward)
            portfolio_values.append(env.get_portfolio_value())
            
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward={total_reward:.2f}, Portfolio=${env.get_portfolio_value():.2f}")
        
        return {
            'model_type': 'dqn',
            'episodes': episodes,
            'final_portfolio': portfolio_values[-1],
            'total_return': (portfolio_values[-1] - env.initial_balance) / env.initial_balance * 100,
            'episode_rewards': episode_rewards,
            'portfolio_values': portfolio_values
        }
    
    def evaluate(self, env):
        """Test the agent without exploration."""
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = self.select_action(state, training=False)
            state, reward, done, info = env.step(action)
            total_reward += reward
        
        return {
            'total_reward': total_reward,
            'final_portfolio_value': env.get_portfolio_value(),
            'total_return_pct': (env.get_portfolio_value() - env.initial_balance) / env.initial_balance * 100,
            'total_trades': env.total_trades
        }


class PolicyGradientAgent:
    """
    Policy Gradient (REINFORCE) agent.
    Directly learns what action to take instead of Q-values.
    """
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        if TF_AVAILABLE:
            self.model = self._build_network(learning_rate)
        
        # store episode data
        self.states = []
        self.actions = []
        self.rewards = []
    
    def _build_network(self, lr):
        model = Sequential([
            Dense(128, activation='relu', input_dim=self.state_dim),
            Dense(64, activation='relu'),
            Dense(self.action_dim, activation='softmax')  # output is action probabilities
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy')
        return model
    
    def select_action(self, state):
        """Sample action from probability distribution."""
        probs = self.model.predict(state.reshape(1, -1), verbose=0)[0]
        return np.random.choice(self.action_dim, p=probs)
    
    def store_transition(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
    
    def _calculate_returns(self):
        """Calculate discounted returns for each timestep."""
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        # normalize returns
        returns = np.array(returns)
        return (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
    
    def train_episode(self):
        """Train after completing an episode."""
        if len(self.states) == 0:
            return 0
        
        returns = self._calculate_returns()
        states = np.array(self.states)
        actions = np.array(self.actions)
        
        # policy gradient update
        with tf.GradientTape() as tape:
            probs = self.model(states, training=True)
            indices = tf.range(len(actions))
            action_probs = tf.gather_nd(probs, tf.stack([indices, actions], axis=1))
            loss = -tf.reduce_mean(tf.math.log(action_probs + 1e-8) * returns)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        # clear episode data
        self.states, self.actions, self.rewards = [], [], []
        return float(loss)
    
    def train(self, env, episodes=100):
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not installed'}
        
        episode_rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                self.store_transition(state, action, reward)
                state = next_state
                total_reward += reward
            
            self.train_episode()
            episode_rewards.append(total_reward)
            
            if episode % 10 == 0:
                print(f"Episode {episode}: Reward={total_reward:.2f}")
        
        return {
            'model_type': 'policy_gradient',
            'episodes': episodes,
            'avg_reward_last_10': np.mean(episode_rewards[-10:])
        }


def train_rl_agent(df, agent_type='dqn', episodes=50):
    """Quick way to train an RL agent."""
    if not TF_AVAILABLE:
        return None, {'error': 'TensorFlow not installed'}
    
    env = TradingEnvironment(df)
    
    if agent_type == 'dqn':
        agent = DQNAgent(env.state_dim, env.action_dim)
    elif agent_type == 'policy_gradient':
        agent = PolicyGradientAgent(env.state_dim, env.action_dim)
    else:
        raise ValueError(f"unknown agent type: {agent_type}")
    
    metrics = agent.train(env, episodes=episodes)
    
    if hasattr(agent, 'evaluate'):
        eval_metrics = agent.evaluate(env)
        metrics.update(eval_metrics)
    
    return agent, metrics


# test it with data
if __name__ == "__main__":
    print("Testing RL Agent with Real Stock Data...")
    
    if not TF_AVAILABLE:
        print("TensorFlow not installed - skipping")
        exit(1)
    
    # Import data collection and indicators
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    
    from src.data_collection import StockDataCollector
    from src.indicators import calculate_all_indicators
    
    # Fetch AAPL data (use 2 years for faster training)
    collector = StockDataCollector()
    df = collector.fetch_stock_data('AAPL', period='2y')
    
    if df.empty:
        print("Failed to fetch data, check internet connection")
        exit(1)
    
    # Calculate all technical indicators
    df = calculate_all_indicators(df)
    print(f"Loaded {len(df)} rows of real AAPL data\n")
    
    # Use only last 100 rows for faster training
    df_train = df.tail(100).reset_index(drop=True)
    print(f"Using last {len(df_train)} rows for RL training")
    
    # Train DQN agent with limited episodes
    print("Training DQN agent (3 episodes)...")
    agent, metrics = train_rl_agent(df_train, 'dqn', episodes=3)
    
    print(f"\nResults:")
    print(f"  Final Portfolio Value: ${metrics.get('final_portfolio_value', 'N/A'):.2f}")
    print(f"  Total Return: {metrics.get('total_return_pct', 0):.2f}%")
    print(f"  Total Trades: {metrics.get('total_trades', 'N/A')}")
    
    print("\nDone!")

