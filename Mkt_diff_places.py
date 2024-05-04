import pulp
import numpy as np

# cas où chaque agent a une seule offre de température sur un lieu donné (5 lieux par exemple)

# liste qui contient les offres
all_bids = []

# création des offres de façon aléatoire : bid = (b,q) avec q = (lieu, 1 ou -1)
S = 40 # nombre d'agents
for i in range (S) :
    price = np.random.random() - 0.5
    location = np.random.randint(1,6) # => de 1 à 5
    quantity = 0
    if price < 0 : 
        quantity = 1
    else :
        quantity = -1
    q = (price, (location, quantity))
    all_bids.append(q)

print(all_bids)

# fonction qui retourne une liste avec les bids concernant le même lieu
def filtered_bids (all_bids, n_location) :
    A = len(all_bids)
    new_bids = []
    for i in range (A) :
        if all_bids[i][1][0] == n_location :
            new_bids.append(all_bids[i])
    return new_bids

def make_var(i):
    return pulp.LpVariable(f"w{i}", lowBound=0, cat="Integer")


# optimisation faites sur les bids du même lieu
def optimization(new_bids) :
    A = len(new_bids)
    prob = pulp.LpProblem("Winner", pulp.LpMaximize)
    wvars = [make_var(i) for i in range(A)]
    prob += pulp.lpSum(new_bids[i][0]*wvars[i] for i in range(A))
    prob += pulp.lpSum(new_bids[i][1][1]*wvars[i] for i in range(A)) >= 0
    for i in range(A):
        prob += wvars[i] <= 1
    pulp.LpStatus[prob.solve()]
    print(pulp.value(prob.objective), [pulp.value(wvars[i]) for i in range(A)])
    #la fin peut changer en fonction de ce qu'on veut faire ensuite


# exemple sur les offres du lieu n°2
optimization(filtered_bids(all_bids, 2))
