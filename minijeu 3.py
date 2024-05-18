#Mini jeu 3
#%%Stratégie d'achat fixe d'action : 1 action pendant 20 jours et 2 action jusqu'au 60ème jour
import numpy as np
sigma = 1.2
nb_jours = 60
objectif_stock = 100
S0 = 100
np.random.seed(1) #reproductibilité

X_n=np.random.normal(0, 1, nb_jours)
X_n=np.insert(X_n, 0, 0)
S_n=np.zeros(nb_jours+1)
S_n[0]=S0
for i in range(1, nb_jours + 1):
    S_n[i]=S_n[i-1]+sigma*X_n[i]

vn=[1]*20+[2]*40
depense_totale=0
stocks_achetees=0
A_n=S0

for jour in range(nb_jours):
    if stocks_achetees >= objectif_stock:
        break
    stocks_achetees+=vn[jour]
    depense_totale+=vn[jour]*S_n[jour+1]
    A_n=((A_n*jour) + S_n[jour + 1])/(jour+1)

gain = 100 * A_n - depense_totale

print(f"Total dépensé : {depense_totale:.2f}")
print(f"Prix moyen cumulé des actions : {A_n:.2f}")
print(f"Gain net : {gain:.2f}")
print(f"nombre de stock acheté:{stocks_achetees}")

#Total dépensé : 9704.93
#Prix moyen cumulé des actions : 97.32
#Gain net : 26.69
#nombre de stock acheté:100
# En somme avec plusieurs essai, on est souvent centré autour de 0 avec parfois des gains négatif ce qui n'assure pas un gain positif.

#%% gradient policy method

#variable d'état : temps t [current_day] / prix d'état du sous jacent S_n / moyenne courant du spot A_n / , nb d'action déjà achetée somme V_n et ce que l'on a dépensé. [somme (v_k)*S_(k+1), k=1...n]
#action : nbr d'action a acheté, sonne la cloche ou pas?

# a chaque etat comparer la fonction dépendant de tout les paramètres avec la sortie

#retourner à chaque etat le programme d'action [nb d'action acheté, sonne la cloche ou pas 0 ou 1]


"""
S0 = 100
S_(n+1) = S_n + sigma*X_(n+1)
(X_n) sont des N(0,1) i.i.d.
A_n = 1/n*sum(S_k) k=1..n  (A_0 = S_0)
but : acheter 100 stocks sur 60 jours.
v_n * S_(n+1)
somme (v_n) = 100
chaque soir à partir du 20e jour, on peut sonner une cloche (lorsque l'on a tout acheté), et le jeu s'arrête
100*A_n - (ce que l'on a dépensé)
"""
# Fraction d'action et condition de 100 actions et de cloche à partir de 20.

#%% Simple sans reinforce

import numpy as np



class LinearPolicy:
    def __init__(self, state_dim, action_dim):  # stade_dim : nb de caracteristique décrivant l'état, action_dim : nombre maximum d'actions possibles (y compris sonner la cloche à partir du 20e jour)
        self.poids = np.random.randn(state_dim, action_dim) * 0.01 # petits poids pour éviter des grands gradient en début d'entrainement

    def action_prob(self, state):
        score = np.dot(state, self.poids) # pour chaque action, le score est calculé comme une combinaison linéaire des caractéristiques de l'état et des poids associés à cette action
        exp_score = np.exp(score - np.max(score)) # on soustrait le max pour éviter d'avoir des trop grand nombres à gérer (c'est équivalent)
        return exp_score / np.sum(exp_score) # softmax

    def get_action(self, state): # choisir l'action selon les probabilités calculées
        action_prob = self.action_prob(state)
        return np.random.choice(len(action_prob), p=action_prob)

def simulate_episode(policy, sigma, max_days=60, max_stocks=100, jour_min_pour_sonner_cloche=20):
    S = np.zeros(max_days) # prix action chaque t (jour)
    A = np.zeros(max_days) # moyenne des prix
    V = np.zeros(max_days) # nombre d'actions acheté chaque jour
    C = np.zeros(max_days)  # cloche sonnée (1) non sonnée (0)
    total_stocks = 0 # nombre total d'action acheté
    total_cost = 0 # cout total
    cloche = False # cloche sonnée ou pas
    actions = []  # pour après afficher le programme d'achat optimal d'action

    # initialisation premier jour
    S[0] = 100
    A[0] = S[0]

    for t in range(1, max_days):
        S[t] = S[t-1] + sigma * np.random.randn()
        A[t] = np.mean(S[:t+1])

        state = np.array([t, S[t], A[t], total_stocks, total_cost])
        state = (state - np.mean(state)) / 100 # Normalisation ATTENTION np.std pas de sens
        # Normalisation pour chaque dimensions

        action = policy.get_action(state)
        num_stocks = action if action < 100 else 0
        sonner_cloche = 1 if action >= 100 and total_stocks >= max_stocks and t >= jour_min_pour_sonner_cloche else 0

        if num_stocks + total_stocks > max_stocks: # pour ne pas dépasser 100 stocks
            num_stocks = max_stocks - total_stocks

        V[t] = num_stocks
        total_stocks += num_stocks
        total_cost += num_stocks * S[t]

        actions.append([num_stocks, sonner_cloche])

        if sonner_cloche:
            C[t] = 1
            cloche = True
            break

        if total_stocks >= max_stocks and t >= jour_min_pour_sonner_cloche:
            C[t] = 1  # Sonne la cloche immediatement lorsque 100 actions sont achetées après 20 jours
            actions[-1][1] = 1  # Update the last action to reflect the bell ringing
            break

    final_day = t
    reward = 100 * A[final_day] - total_cost
    return S, A, V, C, total_cost, reward, final_day, actions

def evaluate_policy(policy, sigma, num_episodes=1000):
    best_episode = None
    best_reward = -np.inf

    for _ in range(num_episodes):
        S, A, V, C, total_cost, reward, final_day, actions = simulate_episode(policy, sigma)

        if reward > best_reward:
            best_reward = reward
            best_episode = (S, A, V, C, total_cost, reward, final_day, actions)

    return best_episode

sigma = 1.0
state_dim = 5
action_dim = 101
# action dim 2 : nb action a acheter , probabilité de sonner la cloche (continue pour explorer) [+ bruit gaussien pour nb action a acheter.]


policy = LinearPolicy(state_dim, action_dim)


best_episode = evaluate_policy(policy, sigma)


S, A, V, C, total_cost, reward, final_day, actions = best_episode
print(f"Meilleur gain d'episode: {reward}")
print(f"Cout total: {total_cost}")
print(f"Jour de fin: {final_day}")
print(f"Prix action: {S[:final_day+1]}")
print(f"Prix moyen: {A[:final_day+1]}")
print(f"Nombre d'action acheté: {V[:final_day+1]}")
print(f"Cloche sonnée: {C[:final_day+1]}")
print(f"Actions cumulées: {V.cumsum()[:final_day+1]}")
print("\nProgramme d'achat optimal':")
for t, action in enumerate(actions):
    print(f"Jour {t+1}: Achat de {action[0]} actions, Sonner la cloche: {action[1]}")



#%% Avec reinforce

import numpy as np



class LinearPolicy:
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

def simulate_episode(policy, sigma, max_days=60, max_stocks=100, jour_min_pour_sonner_cloche=20):
    S = np.zeros(max_days) # prix action chaque t (jour)
    A = np.zeros(max_days) # moyenne des prix
    V = np.zeros(max_days) # nombre d'actions acheté chaque jour
    C = np.zeros(max_days)  # cloche sonnée (1) non sonnée (0)
    total_stocks = 0 # nombre total d'action acheté
    total_cost = 0 # cout total
    cloche = False # cloche sonnée ou pas
    actions = []  # pour après afficher le programme d'achat optimal d'action

    # initialisation premier jour
    S[0] = 100
    A[0] = S[0]

    for t in range(1, max_days):
        S[t] = S[t-1] + sigma * np.random.randn()
        A[t] = np.mean(S[:t+1])

        state = np.array([t, S[t], A[t], total_stocks, total_cost])
        state = (state - np.mean(state)) / (np.std(state) + 1e-5) # Normalisation

        action = policy.get_action(state)
        num_stocks = action if action < 10 else 0
        sonner_cloche = 1 if action >= 10 and total_stocks >= max_stocks and t >= jour_min_pour_sonner_cloche else 0

        if num_stocks + total_stocks > max_stocks: # pour ne pas dépasser 100 stocks
            num_stocks = max_stocks - total_stocks

        V[t] = num_stocks
        total_stocks += num_stocks
        total_cost += num_stocks * S[t]

        actions.append([num_stocks, sonner_cloche])

        if sonner_cloche:
            C[t] = 1
            cloche = True
            break

        if total_stocks >= max_stocks and t >= jour_min_pour_sonner_cloche:
            C[t] = 1  # Sonne la cloche immediatement lorsque 100 actions sont achetées après 20 jours
            actions[-1][1] = 1  # Met à jour la dernière action pour la cloche sonnée
            break

    final_day = t
    reward = 100 * A[final_day] - total_cost
    return S, A, V, C, total_cost, reward, final_day, actions

def evaluate_policy(policy, sigma, num_episodes=100, learning_rate=0.01):
    best_episode = None
    best_reward = -np.inf

    for _ in range(num_episodes):
        states = []
        actions = []
        rewards = []

        S, A, V, R, total_cost, reward, final_day, episode_actions = simulate_episode(policy, sigma)

        for t in range(final_day+1):
            assert len(episode_actions) == final_day
            if t >= len(episode_actions):
                break
            state = np.array([t, S[t], A[t], np.sum(V[:t+1]), total_cost])
            state = (state - np.mean(state)) / (np.std(state) + 1e-5)
            states.append(state)
            actions.append(episode_actions[t][0])
            rewards.append(reward)

        policy.update_policy(states, actions, rewards, learning_rate)

        if reward > best_reward:
            best_reward = reward
            best_episode = (S, A, V, R, total_cost, reward, final_day, episode_actions)

    return best_episode #prendre la moyenne de l'épisode (/ VaR)


sigma = 1.0
state_dim = 5
action_dim = 11

policy = LinearPolicy(state_dim, action_dim)


best_episode = evaluate_policy(policy, sigma, num_episodes=100)

S, A, V, C, total_cost, reward, final_day, actions = best_episode
print(f"Meilleur gain d'episode: {reward}")
print(f"Cout total: {total_cost}")
print(f"Jour de fin: {final_day}")
print(f"Prix action: {S[:final_day+1]}")
print(f"Prix moyen: {A[:final_day+1]}")
print(f"Nombre d'action acheté: {V[:final_day+1]}")
print(f"Cloche sonnée: {C[:final_day+1]}")
print(f"Actions cumulées: {V.cumsum()[:final_day+1]}")
print("\nProgramme d'achat optimal:")
for t, action in enumerate(actions):
    print(f"Jour {t+1}: Achat de {action[0]} actions, Sonner la cloche: {action[1]}")


#- lire des docs sur pytorch et réseau que je vais envoyer
#- simulation pour évaluer une stratégie donnée
#- rappelle : stratégie = fonction : {états] -> {actions]
#"- actions : espace continue de dim 2 : Rx[0,1]
#- normaliser les grandeurs par des valeurs homogènes du problème
