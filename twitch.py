import pulp
import numpy as np
import random as rd
from itertools import product

all_bids = [] # all_bids va contenir toutes les offres du marché

q_size = 10 # taille du vecteur q <=> nombre de types de données différents


# pas besoin de classes pour les org
all_q = [] # all_q contient tous les vecteurs q possible (avec 0 et 1) => 2^10
for combo in product([0, 1], repeat= q_size):
    all_q.append(list(combo))


# on commence par remplir all_bids avec les bids des org, chaque org a un q différent => 2^10 orgs
for q in all_q :
    b = 0
    for i in range(len(q)) : # bid = (b, q) où b est la somme des 1 dans q
        b += q[i]
    bid = (-b, q)
    all_bids.append(bid)



class Brands :
    def __init__(self) :
        self.utility =[rd.random() for _ in range (10)] # chaque marque attribue une utilité ∈ [0,1] pour chaque type de donnée
        self.bid = ()
        
    # Chaque marque propose une bid: si son utilité pour un type de données dépasse un seuil, elle le demande => -1 dans q
    def generate_bid(self, threshold):
        b = 0
        q = [0 for _ in range(10)] # on initialise q à [0(x10)]
        for i, data_utility in enumerate(self.utility):
            if data_utility > threshold :
                q[i] = -1 # si l'utilité du type i est >threshold, on le demande
                b += 2*data_utility # le b est la (somme des 1 dans q)*2 => ça assure que 
        self.bid = (b, q)

nb_brands = 600

brands_list = [Brands() for _ in range(nb_brands)]

threshold = 0.5 # seuil d'utilité pour lequel les marques vondront les données, à voir si on le fait varier en fonction des marques
# Générer une offre pour chaque marque dans la liste et l'ajouter au marché (all_bids)
for brand in brands_list:
    brand.generate_bid(threshold) #
    all_bids.append(brand.bid)

# !!! des bid = (b=0, q=[0(x10)]) apparaissent ducoup...



rd.shuffle(all_bids)

def make_var(i):
    return pulp.LpVariable(f"w{i}", lowBound=0, cat="Integer")

def optimization(new_bids) :
    A = len(new_bids)
    b = []
    q = []
    for i in range (A) :
        b.append(new_bids[i][0])
        q.append(new_bids[i][1])
    prob = pulp.LpProblem("Winner", pulp.LpMaximize)
    wvars = [make_var(i) for i in range(A)]
    prob += pulp.lpSum(b[i]*wvars[i] for i in range(A)) # fonction objectif à maximiser
    for j in range (10) : # pour chaque type de donnée ->
        prob += pulp.lpSum(q[i][j]*wvars[i] for i in range(A)) >= 0 # -> on met une contrainte
    for i in range(A):
        prob += wvars[i] <= 1
    pulp.LpStatus[prob.solve()]

    # on peut afficher les variables de décisions, si wi = 1 => l'offre de l'agent i choisie
    print("WD = ", pulp.value(prob.objective), "\nVariables de décisions : ", [pulp.value(wvars[i]) for i in range(A)])
    cnt = 0 # comptele nb de wi à 1
    for i in range (A) :
        cnt += pulp.value(wvars[i])
    print("cnt = ", cnt, " sur ", len(new_bids))


#print(all_bids)
optimization(all_bids)


