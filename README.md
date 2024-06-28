# Optimisation-d-achat-de-100-actions-sur-60-jours

S0 = 100 le prix initial de l'action 
S_(n+1) = S_n + sigma*X_(n+1) les prix des jours suivants
où (X_n) sont des loi normales N(0,1) i.i.d.
A_n = 1/n*sum(S_k) k=1..n  (A_0 = S_0) est la moyenne du prix 
but : nous voulons acheter 100 stocks sur 60 jours.
v_n * S_(n+1) désigne le prix qu'on a dépensé pour acheter v_n action
sachant que à la fin on doit avoir somme (v_n) = 100 
chaque soir à partir du 20e jour, on peut sonner une cloche uniquement lorsque 100 actions ont été achetés, et le jeu s'arrête alors.
Nous souhaitons maximiser ceci :100*A_n - (ce que l'on a dépensé)

# 1ère version
Initialisation :

LinearPolicy initialise une politique avec des poids aléatoires.
simulate_episode simule un épisode complet, retournant les états, actions et récompenses pour chaque jour.
evaluate_policy évalue et met à jour la politique sur plusieurs épisodes, cherchant le meilleur épisode.
Simulation d'un épisode :

Chaque jour, le prix de l'action est mis à jour et l'état est normalisé.
Une action est choisie en fonction de l'état normalisé.
La cloche est sonnée si les conditions sont remplies.
L'épisode s'arrête si la cloche est sonnée ou si le nombre maximal d'actions est atteint.
Évaluation et mise à jour de la politique :

Pour chaque épisode, les états, actions et récompenses sont enregistrés.
La politique est mise à jour en utilisant le gradient de politique basé sur les retours.

# 2ème version nn_continuous_(final) :
Ce programme implémente une stratégie d'achat d'actions utilisant des réseaux de neurones et des méthodes de reinforcement learning. Le modèle apprend à maximiser les gains en achetant des actions de manière optimale sur une période de 60 jours.

1. Initialisation des paramètres
Définition des paramètres initiaux du problème, y compris le prix initial de l'action, la volatilité (sigma), le nombre de jours et l'objectif de nombre d'actions à acheter.
2. Simulation des prix
Fonction pour simuler les prix des actions sur les days jours suivants, à partir d'un prix initial et d'une série de variations quotidiennes.
3. Définition de la récompense
Fonction pour calculer la récompense en tenant compte du coût total et d'une pénalité pour les grands achats en une seule journée.
4. Définition du modèle de réseau de neurones
Un réseau de neurones avec trois couches cachées, produisant des sorties pour la moyenne et l'écart-type des distributions de nombre d'actions à acheter et de décision de sonner la cloche.
5. Normalisation de l'état
Fonction pour normaliser les variables d'état avant de les passer au modèle.
6. Simulation d'un épisode
Fonction pour simuler un épisode complet, où le modèle prend des décisions d'achat d'actions sur une période de 60 jours.
7. Calcul des retours
Fonction pour calculer les retours cumulés pour chaque étape de l'épisode.
8. Entraînement du modèle
Fonction pour entraîner le modèle sur plusieurs épisodes, en ajustant les poids du réseau de neurones pour maximiser la récompense.
9. Évaluation de la politique
Fonction pour évaluer la politique du modèle sur plusieurs épisodes, en calculant les statistiques moyennes des performances.
10. Initialisation et entraînement du modèle
Initialisation du modèle, application de l'initialisation des poids, entraînement du modèle et affichage des résultats moyens sur plusieurs épisodes.


# Modification

retourner la moyenne des episodes et non plus le meilleur, ou le dernier épisode ce qui est plus robuste pour évaluer la performance de la politique
Normalisation changée (min max)
distribution gaussienne avec torch.distributions.Normal


problème ca achete tjr 100 actions dès le début, je comprend pas pq.
