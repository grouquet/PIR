import pulp # exécuter "pip install pulp"


# représente les vendeurs
class Seller:
    def __init__ (self, agent_id):
        self.agent_id = agent_id # identifie le vendeur de manière unique
        self.bids = [] # contient ses offres 
    
    #exemple de méthode qui ajoute une offre
    def add_bid(self, amount, commodities_quantities):
            new_bid = Bid(self.agent_id, amount, commodities_quantities)
            self.bids.append(new_bid)

# représente les acheteurs
class Buyer :
    def __init__ (self, agent_id):
        self.agent_id = agent_id # identifie l'acheteur de manière unique
        self.bids = [] # contient les offres du vendeur
    
    #exemple de méthode qui ajoute une offre
    def add_bid(self, amount, commodities_quantities):
            new_bid = Bid(self, amount, commodities_quantities)
            self.bids.append(new_bid)


# représente les biens (par exemple une instance peut être : vol paris->londres)
class Commodity :
    def __init__(self, commodity_id, name) :
        self.commodity_id = commodity_id # identifie la commodity de manière unique
        self.name = name

# représente les offres 
class Bid :
    def __init__(self, agent_id, amount, commodities_quantities) :
        self.agent_id = agent_id # id de l'Agent qui fait l'offre (pas obligatoire ?)
        self.amount = amount # montant de l'offre (+ pour un achat, - pour une vente)
        self.commodities_quantities = commodities_quantities # dictionnaire où clés sont ids des biens et valeurs sont quantités




#création des instances d'agents et des biens 
N_sellers = 2
sellers = [] # Liste pour stocker les instances d'Agent
for i in range(1, N_sellers + 1) :
    sellers.append(Seller(i))

N_buyers = 2
buyers = []
for i in range(1, N_buyers + 1) :
    buyers.append(Buyer(i))

commodity_details = [
    (1, "Temperature"),
    (2, "Humidity"),
    (3, "Lieu")
    # en rajouter après
]
commodities = []
for commodity_id, name in commodity_details:
    commodities.append(Commodity(commodity_id, name))


# exemples de dictionnaires qui répertorient les biens et leurs quantités
# et assignation des classes
cq_seller1 = {commodities[0].commodity_id : -1, commodities[1].commodity_id : -1}
sellers[0].add_bid(180, cq_seller1) # vendeur qui vend 1 vols P->L et 1 L->P pour 180
cq_seller2 = {commodities[0].commodity_id : 0, commodities[1].commodity_id : -1}
sellers[1].add_bid(160, cq_seller2) # vendeur qui vend 0 vols P->L et 1 L->P pour 160


cq_buyer1 = {commodities[0].commodity_id : 1, commodities[1].commodity_id : 1}
buyers[0].add_bid(140, cq_buyer1) # acheteur qui veut 1 vols P->L et 1 L->P pour 140


#Affichage des offres soumises
#   for bid in agent1.bids:
#        print(f"Agent {bid.agent.agent_id} offre {bid.amount} pour {bid.commodities_quantities}")
#
#   for bid in agent2.bids:
#       print(f"Agent {bid.agent.agent_id} offre {bid.amount} pour {bid.commodities_quantities}")





#----------------------------------------------------


# matrices qui stockent les bids : chaque ligne est un agent (n° ligne = agent_id -1) et chaque colonne est une offre
matrix_sellers = [[] for _ in range(N_sellers)] 
matrix_buyers = [[] for _ in range(N_buyers)]

for seller in sellers:
    matrix_sellers[seller.agent_id - 1].append(seller.bids)

for buyer in buyers:
    matrix_buyers[buyer.agent_id - 1].append(buyer.bids)

all_bids = [] #stocke toutes les offres (pratique pour la suite)
for row in matrix_sellers:
    for column in row:
        all_bids.extend(column)
for row in matrix_buyers:
    for column in row:
        all_bids.extend(column)



# Optimisation

prob = pulp.LpProblem("MaximizeTotalValue", pulp.LpMaximize) # crée un objet LpProblem pour optimisation linéaire    

bid_vars = {bid: pulp.LpVariable(f"bid_{bid.agent_id}", cat='Binary') for bid in all_bids}

prob += pulp.lpSum([bid.amount * bid_vars[bid] for bid in all_bids])

# Contrainte vérifiant que pour chaque produit, la quantité demandée n'excède pas la quantité fournie :  ∑∑qikjwik ≥ 0 
for commodity in commodities:
    commodity_id = commodity.commodity_id

    quantity_sum = pulp.lpSum([bid_vars[bid] * bid.amount * bid.commodities_quantities.get(commodity_id, 0) for bid in all_bids])
    
    prob += quantity_sum >= 0, f"Constraint_Commodity_{commodity_id}"

# contrainte ∑wik <= 1 pour tout agent i
for seller in sellers:
    prob += pulp.lpSum([bid_vars[bid] for bid in seller.bids]) <= 1, f"Constraint_Seller_{seller.agent_id}"

for buyer in buyers:
    prob += pulp.lpSum([bid_vars[bid] for bid in buyer.bids]) <= 1, f"Constraint_Buyer_{buyer.agent_id}"

prob.solve()




# Vérification de la solution
if pulp.LpStatus[prob.status] == "Optimal":
    print("Solution optimale trouvée!")
    
    # Impression des valeurs des variables de décision
    for bid, var in bid_vars.items():
        if var.varValue == 1:
            print(f"Offre sélectionnée : {bid.agent_id} - Montant : {bid.amount} - Quantités : {bid.commodities_quantities}")

    # Vérification des contraintes
    for constraint in prob.constraints.values():
        slack = constraint.slack
        if slack >= 0:
            print(f"Contrainte {constraint.name} respectée avec un écart de {slack}")
        else:
            print(f"Contrainte {constraint.name} violée de {-slack}")

else:
    print("Pas de solution optimale trouvée.")

