

#%% Avec pytorch

# state variables : t, S_t, A_t, total_stocks, total_cost
# reward function : maximize 100 * A_t - total_cost
# policy network : neural network that outputs the probability distribution over actions:  of buying a certain number of stocks and ringing the bell 
# reinforce algorithm : update the policy network by maximizing the expected return using the policy gradient method


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# défintion des variables d'état et initialisation des paramètres

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


def reward(total_stocks, total_spent, A_n):
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
        self.output = nn.Linear(128, 2)  # Sorties: nombre d'actions à acheter, sonner la cloche ou pas
        self.act_output = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.act3(self.hidden3(x))
        x = self.act_output(self.output(x))
        return x

# normalisation des états
def normalize_state(state):
    t, S_n, A_n, total_stocks, total_spent = state
    return np.array([t / days, S_n / 100, A_n / 100, total_stocks / goal, total_spent / (goal * 100)])



def train(num_episodes):
    for episode in tqdm(range(num_episodes)):

        np.random.seed(episode)
        X = np.random.normal(0, 1, days)
        price_history = simulate_price(S0, X, sigma) # len : days + 1


        states = []
        actions = []
        rewards = []

        S_n = S0
        total_stocks = 0
        total_spent = 0

    
        for t in range(1, days + 1):
            S_n  = price_history[t]
            A_n = np.mean(price_history[1:t+1])
            state = np.array([t, S_n, A_n, total_stocks, total_spent])
            state = normalize_state(state)
            
            state_tensor = torch.tensor(state, dtype=torch.float32) 
            action_output = model(state_tensor)
            nb_stocks = int(action_output[0].item() * goal)
            bell_ring = action_output[1].item() >= 0.5

            if nb_stocks + total_stocks > goal:
                nb_stocks = goal - total_stocks
            
            total_stocks += nb_stocks
            total_spent += nb_stocks * S_n

            actions.append((nb_stocks, bell_ring))
            rewards.append(reward(total_stocks, total_spent, A_n))

            if bell_ring and t >= 20 and total_stocks >= goal:
                break # interruption de la boucle for
            else:
                continue
            
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + 0.99 * G
            returns.insert(0, G)  # discount factor 0.99

        # conversion en tenseurs des listes de retour, d'états normalisés et d'actions
        returns = torch.tensor(returns)
        states = torch.stack([torch.tensor(normalize_state([t, price_history[t], np.mean(price_history[:t+1]), sum(a[0] for a in actions[:t]), total_spent]), dtype=torch.float32) for t in range(len(actions))])
        actions = torch.tensor(actions)

        # Maj du modèle
        optimizer.zero_grad() # reset the gradients
        action_output = model(states) # batch processing of states : treat all states at once
        nb_stocks_output = action_output[:, 0] # each line of action_output corresponds to the output of the model for a state
        bell_ring_output = action_output[:, 1] 

        
        loss_nb_stocks = nn.MSELoss()(nb_stocks_output, actions[:, 0].float())
        loss_sonner_cloche = nn.BCEWithLogitsLoss()(bell_ring_output, actions[:, 1].float())

        loss_policy = loss_nb_stocks + loss_sonner_cloche
        loss_policy.backward()
        optimizer.step()


def simulate_episode(model):
    X = np.random.normal(0, 1, days)
    price_history = simulate_price(S0, X, sigma)

    S = [] # prix action chaque t (jour)
    A = [] # moyenne des prix
    V = [] # nombre d'actions acheté chaque jour
    C =[]  # cloche sonnée (1) non sonnée (0)
    total_stocks = 0 # nombre total d'action acheté
    total_cost = 0 # cout total
    actions = []  # pour après afficher le programme d'achat optimal d'action

    for t in range(1, days+1):
        S_n = price_history[t]
        A_n = np.mean(price_history[1:t+1])
        state = np.array([t, S_n, A_n, total_stocks, total_cost])
        state = normalize_state(state)
        state_tensor = torch.tensor(state, dtype=torch.float32) 
        action_output = model(state_tensor)
        nb_stocks = int(action_output[0].item() * goal)
        bell_ring = action_output[1].item() >= 0.5 

        if nb_stocks + total_stocks > goal:
            nb_stocks = goal - total_stocks
        
        S.append(S_n)
        A.append(A_n)
        V.append(nb_stocks)
        total_stocks += nb_stocks
        total_cost += nb_stocks * S_n

        actions.append((nb_stocks, bell_ring))

        if (bell_ring and t >= 20 and total_stocks >= goal):
            C.append(1)
            break
        else:
            C.append(0)

    final_day = t
    reward_value = reward(total_stocks, total_cost, A_n)

    return S, A, V, C, total_cost, reward_value, final_day, actions



def evaluate_policy(model, num_episodes):
    total_rewards = 0
    total_costs = 0
    total_days = 0
    all_S = np.zeros(days + 1)
    all_A = np.zeros(days + 1)
    all_V = np.zeros(days + 1)
    all_C = np.zeros(days + 1)
    for _ in range(num_episodes):
        S, A, V, C, total_cost, reward_value, final_day, _ = simulate_episode(model)
        total_rewards += reward_value
        total_costs += total_cost
        total_days += final_day
        all_S[:final_day+1] += S[:final_day+1]
        all_A[:final_day+1] += A[:final_day+1]
        all_V[:final_day+1] += V[:final_day+1]
        all_C[:final_day+1] += C[:final_day+1]

    avg_reward = total_rewards / num_episodes
    avg_cost = total_costs / num_episodes
    avg_days = total_days / num_episodes
    avg_S = all_S / num_episodes
    avg_A = all_A / num_episodes
    avg_V = all_V / num_episodes
    avg_C = all_C / num_episodes

    return avg_S, avg_A, avg_V, avg_C, avg_cost, avg_reward, avg_days



model = StockNetwork()
optimizer = optim.Adam(model.parameters(), lr=0.05)
train(10000)
S, A, V, C, total_cost, reward_value, final_day, actions = evaluate_policy(model, num_episodes=10000)
print(f"Gain moyen d'episode: {reward_value}")
print(f"Cout total: {total_cost}")
print(f"Jour de fin: {final_day}")
print(f"Prix action: {S[:final_day+1]}")
print(f"Prix moyen: {A[:final_day+1]}")
print(f"Nombre d'action acheté: {V[:final_day+1]}")
print(f"Cloche sonnée: {C[:final_day+1]}")
print(f"Actions cumulées: {np.cumsum(V)[:final_day+1]}")
print("\nProgramme d'achat optimal:")
for t, action in enumerate(actions):
    print(f"Jour {t+1}: Achat de {action[0]} actions, Sonner la cloche: {action[1]}")




#%%
"""

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


"""