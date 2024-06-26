import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# state variable definition and parameter initialization
S0 = 100
sigma = 2.0
days = 60 
goal = 100

def simulate_price(S_n, X, sigma):
    prices = [S_n]
    for x in X:
        S_n = S_n + sigma * x
        prices.append(S_n)
    return prices

def reward(total_spent, A_n):
    return 100 * A_n - total_spent

class StockNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(5, 128)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(128, 128)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(128, 128)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(128, 2)  # Outputs: number of stocks to buy, ring the bell or not
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act_output(self.output(x))
        return x

# state normalization
def normalize_state(state):
    t, S_n, A_n, total_stocks, total_spent = state
    return np.array([t / days, S_n / 100, A_n / 100, total_stocks / goal, total_spent / (goal * 100)])

min_action_threshold = 1.0 # minimum threshold for the number of stocks to buy

def simulate_episode(model):
    S_n = S0
    total_stocks = 0
    total_spent = 0
    X = np.random.normal(0, 1, days)
    prices = simulate_price(S_n, X, sigma)
    actions = []
    rewards = []
    states = []
    done = False
    
    for t in range(days): 
        # check if the goal is reached
        if total_stocks >= goal and t >= 19:
            ring_bell = True
            done = True
        elif t >= 19:
            ring_bell = (action[1] >=0.5) 
        else:
            ring_bell = False

    
        # state
        A_n = np.mean(prices[:t+1])
        state = normalize_state((t, S_n, A_n, total_stocks, total_spent))
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        # forward pass
        with torch.no_grad(): # to avoid calculating gradients during inference (forward pass) 
            action = model(state_tensor).numpy()

        # action
        v_n = action[0] * (goal - total_stocks) if not done else 0 # if the episode is done, reset the number of stocks to 0

        if v_n < min_action_threshold and (goal - total_stocks) > min_action_threshold:
            v_n = min_action_threshold
        elif (goal - total_stocks) <= min_action_threshold:
            v_n = goal - total_stocks

        v_n = min(max(v_n, 0), goal - total_stocks) # to avoid buying more stocks than needed vn >= 0 and vn <= goal - total_stocks

        # reward and next state
        total_spent += v_n * prices[t]
        total_stocks += v_n 
        actions.append((v_n, ring_bell))
        rewards.append(reward(total_spent, A_n))
        states.append(state)
        

        if done:
            break
    
    return states, actions, rewards, prices

def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns

def train(num_episodes):
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    gamma = 0.99

    for _ in tqdm(range(num_episodes)):
        states, actions, rewards, _ = simulate_episode(model)
        returns = compute_returns(rewards, gamma)
        
        optimizer.zero_grad() # reset the gradients
        loss = 0 # loss initialization

        for state, action, G in zip(states, actions, returns):
            state_tensor = torch.tensor(state, dtype=torch.float32)
            action_tensor = torch.tensor([int(action[0] >= 0.5), int(action[1] >= 0.5)], dtype=torch.float32)
            G_tensor = torch.tensor(G, dtype=torch.float32)
            
            probs = model(state_tensor)
            dist = torch.distributions.Bernoulli(probs)
            log_prob = dist.log_prob(action_tensor).sum()
    
            loss -= log_prob * G_tensor

        loss.backward()
        optimizer.step()

# initialisation and training of the model
model = StockNetwork()
train(10000)

# Évaluation de la politique sur 1 épisode
states, actions, rewards, prices = simulate_episode(model)
final_day = len(actions)
total_spent = sum([a[0] * prices[t] for t, a in enumerate(actions)])
total_stocks = sum([a[0] for a in actions])
A_n = np.mean(prices[:final_day])
reward_value = reward(total_spent, A_n)

print(f"Gain moyen d'episode: {reward_value}")
print(f"Cout total: {total_spent}")
print(f"Jour de fin: {final_day}")
print(f"Prix action: {prices[:final_day+1]}")
print(f"Prix moyen: {A_n}")
print(f"Nombre d'action acheté: {[a[0] for a in actions[:final_day+1]]}")
print(f"Cloche sonnée: {[a[1] for a in actions[:final_day+1]]}")
print(f"Actions cumulées: {np.cumsum([a[0] for a in actions[:final_day+1]])}")
print("\nProgramme d'achat optimal:")
for t, action in enumerate(actions):
    print(f"Jour {t+1}: Achat de {action[0]} actions, Sonner la cloche: {action[1]}")