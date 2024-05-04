import pulp
import numpy as np

# cas où chaque agent a plusieurs offre sur différents lieux (10 lieux par ex)

# matrice qui contient les offres : chaque sous-liste contient les offres d'un agent
all_bids = []

S = 100 # nombre d'agents
L = 10 # nombre de lieux
locations = list(range(1, L + 1))  # liste des lieux disponibles

# création des offres de façon aléatoire : bid = (b,q) avec q = (lieu, 1 ou -1)
for i in range (S) :
    agent_bids = [] # sous-liste qui contient les offres de l'agent i
    is_seller = np.random.choice([True, False]) # est un vendeur ou pas
    nb_bids = np.random.randint(1,L + 1) # nombres d'offres qu'aura l'agent i, max = L (car 1 offre par lieu au max ?)
    chosen_locations = np.random.choice(locations, nb_bids, replace=False)  # choisi nb_bids lieux sans remplacement => chaque offre aura un lieu différent
    for location in chosen_locations :
        if is_seller :
            price = np.random.random() - 1
            quantity = 1 
        else : 
            price = np.random.random()
            quantity = -1
        q = (price, (location, quantity))
        agent_bids.append(q)
    all_bids.append(agent_bids)
        
# pour voir toutes les offres du marchés : 
#print(all_bids, "\n")



# fonction qui retourne une liste avec les bids concernant le même lieu
def filtered_bids (all_bids, n_location) :
    A = len(all_bids)
    new_bids = [[] for i in range (A)] 
    for i in range (A) :
        for bid in all_bids[i] :
            if bid[1][0] == n_location :
                new_bids[i].append(bid)
    return new_bids
# dans cette liste, l'élément i est l'offre de l'agent i => si l'agent i n'a pas d'offre pour le lieux n_location, l'élément i est vide


# par exemple si on veut réunir les offres du lieu 3 :
#loc = 3 
#print("\n Offres du lieu ", loc, " : \n", filtered_bids(all_bids, loc), "\n")


def make_var(i):
    return pulp.LpVariable(f"w{i}", lowBound=0, cat="Integer")

# fonction d'optimisation sur les bids du même lieu
def optimization(new_bids) :
    A = len(new_bids)
    b = []
    q = []
    for i in range (A) :
        if new_bids[i] : # si il y a une offre pour l'agent i on ajoute son b et sa quantité (1 ou -1)
            b.append(new_bids[i][0][0])
            q.append(new_bids[i][0][1][1])
        else :
            b.append(0) # si l'agent i n'a pas d'offre, on met 0 => pas d'impact, wi = 0
            q.append(0) # pareil
    prob = pulp.LpProblem("Winner", pulp.LpMaximize)
    wvars = [make_var(i) for i in range(A)]
    prob += pulp.lpSum(b[i]*wvars[i] for i in range(A))
    prob += pulp.lpSum(q[i]*wvars[i] for i in range(A)) >= 0
    for i in range(A):
        prob += wvars[i] <= 1
    pulp.LpStatus[prob.solve()]
    # on peut afficher les variables de décisions, si wi = 1 => l'offre de l'agent i choisie
    print("WD = ", pulp.value(prob.objective), "\nVariables de décisions : ", [pulp.value(wvars[i]) for i in range(A)])






# exemple sur les offres du lieu n°2
optimization(filtered_bids(all_bids, 2))