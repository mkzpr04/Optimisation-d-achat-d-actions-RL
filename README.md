# Optimisation-d-achat-de-100-actions-sur-60-jours

S0 = 100 le prix initial de l'action 
S_(n+1) = S_n + sigma*X_(n+1) 
(X_n) sont des loi normales N(0,1) i.i.d.
A_n = 1/n*sum(S_k) k=1..n  (A_0 = S_0) est la moyenne du prix 
but : nous voulons acheter 100 stocks sur 60 jours.
v_n * S_(n+1) désigne le prix qu'on a dépensé pour acheter v_n action
somme (v_n) = 100 
chaque soir à partir du 20e jour, on peut sonner une cloche uniquement lorsque 100 actions ont été achetés, et le jeu s'arrête alors.
Nous souhaitons maximiser ceci :100*A_n - (ce que l'on a dépensé)

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

# Achat de fraction d'action


# Modification

retourner la moyenne des episodes et non plus le meilleur épisode
Normalisation changée (min max)
