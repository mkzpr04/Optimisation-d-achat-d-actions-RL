#%% Avec pytorch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# défintion des variables d'état et initialisation des paramètres

S0 = 100
sigma = 1.0
days = 60 
goal = 100

current_day = 0
S_n = S0
A_n = S0
total_spent = 0
total_stocks = 0


np.random.seed(0)
X = np.random.normal(0, 1, days)

def simulate_price(S_n, X, sigma):
    prices = [S_n]
    for x in X:
        S_n = S_n + sigma * x
        prices.append(S_n)
    return prices

price_history = simulate_price(S0, X, sigma) # len : days + 1

def reward(total_stocks, total_spent, A_n):
    return 100 * A_n - total_spent

class StockNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(5, 64)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(64, 64)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(64, 2)  # Sorties: nombre d'actions à acheter, sonner la cloche ou pas
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act_output(self.output(x))
        return x
    
def simulate_episode(model, sigma, max_days=60, max_stocks=100, bell_day=20):
    S = np.zeros(max_days) # prix action chaque t (jour)
    A = np.zeros(max_days) # moyenne des prix
    V = np.zeros(max_days) # nombre d'actions acheté chaque jour
    C = np.zeros(max_days)  # cloche sonnée (1) non sonnée (0)
    total_stocks = 0 # nombre total d'action acheté
    total_cost = 0 # cout total
    bell = False # cloche sonnée ou pas
    actions = []  # pour après afficher le programme d'achat optimal d'action
    log_probs = []  # pour après calculer le gradient de la politique
    rewards = []  # stocker les recompenses

    # initialisation premier jour
    S[0] = 100
    A[0] = S[0]

    for t in range(1, max_days):
        S[t] = S[t-1] + sigma * np.random.randn()
        A[t] = np.mean(S[1:t+1])

        state = torch.tensor([t, S[t-1], A[t-1], total_stocks, total_cost], dtype=torch.float32) 
        # normalisation par min et max
        min_t, min_S, min_A, min_total_stocks, min_total_cost = 0, 0, 0, 0, 0
        max_t, max_S, max_A, max_total_stocks, max_total_cost = max_days, np.max(S), np.max(A), max_stocks, 100 * np.max(A)

        min_values = torch.tensor([min_t, min_S, min_A, min_total_stocks, min_total_cost], dtype=torch.float32)
        max_values = torch.tensor([max_t, max_S, max_A, max_total_stocks, max_total_cost], dtype=torch.float32)

        state_normalized = (state - min_values) / (max_values - min_values)

        action_probs = model(state_normalized) # actions_prob est un tenseur de taille 2
        num_stocks = action_probs[0].item() * max_stocks 
        sonner_cloche = action_probs[1].item() > 0.5

        log_prob = torch.log(action_probs[0]) + torch.log(action_probs[1])
        log_probs.append(log_prob)

        if num_stocks + total_stocks > max_stocks:  # pour ne pas dépasser 100 stocks
            num_stocks = max_stocks - total_stocks

        V[t] = num_stocks
        total_stocks += num_stocks
        total_cost += num_stocks * S[t]

        actions.append([num_stocks, sonner_cloche])
        rewards.append(0)  # La récompense immédiate est 0 jusqu'à la fin de l'épisode

        if sonner_cloche:
            C[t] = 1
            cloche = True
            break

        if total_stocks >= max_stocks and t >= bell_day:
            C[t] = 1  # Sonne la cloche immédiatement lorsque 100 actions sont achetées après 20 jours
            actions[-1][1] = 1  # Mettre à jour la dernière action pour refléter la cloche
            break

    final_day = t
    final_reward = reward(total_stocks, total_cost, A[final_day])
    rewards[-1] = final_reward  # La récompense finale est calculée à la fin de l'épisode

    return S, A, V, C, total_cost, final_reward, final_day, actions, log_probs, rewards


def train_reinforce(policy, optimizer, num_episodes=1000, gamma=0.99):
    for episode in range(num_episodes):
        S, A, V, C, total_cost, final_reward, final_day, actions, log_probs, rewards = simulate_episode(policy, sigma)

        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        policy_losses = []
        for log_prob, R in zip(log_probs, returns):
            policy_losses.append(-log_prob * R)

        optimizer.zero_grad()
        policy_loss = torch.cat(policy_losses).sum()
        policy_loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}/{num_episodes}, Last Reward: {final_reward}")

    print("Training finished!")

model = StockNetwork()
loss_fn = nn.BCEwithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
train_reinforce(model, optimizer, num_episodes=1000, gamma=0.99)



"""
class Reinforce:
    def __init__(self, model, optimizer, gamma=0.99):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.episode_rewards = []
        self.episode_log_probs = []

class Gradient_Policy:
    def __init__(self, state_dim, action_dim):  # stade_dim : nb de caracteristique décrivant l'état, action_dim : nombre maximum d'actions possibles (y compris sonner la cloche à partir du 20e jour)
        self.poids = np.random.randn(state_dim, action_dim) * 0.01 # petits poids pour éviter des grands gradient en début d'entrainement

    def action_prob(self, state):
        score = np.dot(state, self.poids) # pour chaque action, le score est calculé comme une combinaison linéaire des caractéristiques de l'état et des poids associés à cette action
        exp_score = np.exp(score - np.max(score)) # on soustrait le max pour éviter d'avoir des trop grand nombres à gérer (c'est équivalent)
        return exp_score / np.sum(exp_score) # softmax

    def get_action(self, state): # choisir l'action selon les probabilités calculées
        action_prob = self.action_prob(state)
        return np.random.choice(len(action_prob), p=action_prob)

    def update_policy(self, states, actions, rewards, learning_rate=0.01): # states : np array , actions prise : list,  récompense correspondant à l'action prise dans l'état précédent (S0 A0 R1..), learning_rate de 1% pour ajuster les poids à chaque update.
        G = 0
        policy_gradient = np.zeros_like(self.poids) # pour après pouvoir ajuster les poids de la politique
        y = 0.99  # discount

        for t in reversed(range(len(rewards))): #boucle inversée
            G = rewards[t] + y * G
            state = states[t]
            action = actions[t]
            action_probs = self.action_prob(state)
            dlog = np.zeros_like(self.poids[:, :self.poids.shape[1]]) #  matrice pour stocker les gradients de la log-probabilité des actions

            for a in range(self.poids.shape[1]): # pour chaque action
                dlog[:, a] = state * ((1 - action_probs[a]) if a == action else -action_probs[a])
                # calcul du gradient de la policy
                # ∇ln(π(a∣s,θ)) = ∂ln(π(a∣s,θ))/∂θ = s×(1−π(a∣s,θ)) si a est l’action choisie, sinon −s×π(a∣s,θ)
            policy_gradient += dlog * G

        self.poids += learning_rate * policy_gradient




model = StockNetwork()


"""