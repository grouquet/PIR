import pulp
import numpy as np
import random as rd
from itertools import product
import csv
from collections import defaultdict
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import random

nom_fichier = './Twitch_game_data.csv' # Nom du fichier CSV contenant les données

# Initialisation d'un dictionnaire vide
donnees = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))


with open(nom_fichier, mode='r', newline='', encoding='cp1252') as fichier_csv:
    lecteur_csv = csv.DictReader(fichier_csv)
    
    # Parcours des lignes du fichier CSV
    for ligne in lecteur_csv:
        game_name = ligne['Game']
        year = ligne['Year']
        month = ligne['Month']
        hours_watched = ligne['Hours_watched']
        hours_streamed = ligne['Hours_streamed']
        peak_viewers = ligne['Peak_viewers']
        peak_channels = ligne['Peak_channels']
        streamers = ligne['Streamers']
        avg_viewers = ligne['Avg_viewers']
        
        # Ajout de chaque valeur à la liste appropriée dans le dictionnaire
        donnees[game_name][year][month] = {'hours_watched': hours_watched, 'hours_streamed': hours_streamed, 'peak_viewers': peak_viewers, 'peak_channels': peak_channels, 'streamers': streamers, 'avg_viewers': avg_viewers}

# On a un dictionnaire où chaque clé est un nom de jeu, chaque sous-clé est une année, chaque sous-sous-clé est un mois, et la valeur est un autre dictionnaire avec 'hours_watched' et 'hours_streamed'

# On décide d'enlever les jeux du dictionnaire qui n'ont qu'une seule entrée de données
# On crée une liste de jeux à enlever
jeux_a_enlever = [game for game, data in donnees.items() if len(data) == 1]

# On enlève les jeux de la liste des jeux à enlever
for game in jeux_a_enlever:
    del donnees[game]

# Maintenant le dictionnaire ne contient que des jeux avec plus d'une entrée de données

# On converti le dictionnaire en un dictionnaire régulier
def defaultdict_to_dict(d):
    if isinstance(d, defaultdict):
        d = {k: defaultdict_to_dict(v) for k, v in d.items()}
    return d

donnees_dict = defaultdict_to_dict(donnees)

'''
# Affichage de tous les éléments du dictionnaire
for game_name, game_data in donnees_dict.items():
    print(f"{game_name}: {game_data}\n")
'''

'''
# Ou afficher seulement les premiers éléments de chaque jeu
for game_name, game_data in list(donnees_dict.items())[:5]:
    print(f"{game_name}: {game_data}\n")
'''

'''
# Affichage du premier élément du dictionnaire (League of Legends) afin de faire des tests avec les q retournés par remplissage_q()
first_key = next(iter(donnees_dict))
first_value = donnees_dict[first_key]

print(f"First key: {first_key}")
print(f"First value: {first_value}")
'''


###############################################################################

# Côté vendeur
all_bids = [] # all_bids va contenir toutes les offres du marché

q_size = 96 # taille du vecteur q <=> nombre de types de données différents


# pas besoin de classes pour les org

# remplissage de all_q avec les vecteurs q possibles pour chaque jeu en fonction des données présentées dans la dataset
def remplissage_q():

    all_q = [] # all_q contient tous les vecteurs q possible (avec 0 et 1) => 817, un pour chaque jeu et vendeur
    for game_name, game_data in donnees_dict.items():
        q = [0]*q_size
        for year, year_data in game_data.items():
            for month, value in year_data.items():
                # on remplit q en fonction des données présentes par année et mois
                if year == "2016":
                    k = 0
                elif year == "2017":
                    k = 12
                elif year == "2018":
                    k = 24
                elif year == "2019":
                    k = 36
                elif year == "2020":
                    k = 48
                elif year == "2021":
                    k = 60
                elif year == "2022":
                    k = 72
                elif year == "2023":
                    k = 84
                
                j = 0
                if month == "01":
                    j = 0
                elif month == "02":
                    j = 1
                elif month == "03":
                    j = 2
                elif month == "04":
                    j = 3
                elif month == "05":
                    j = 4
                elif month == "06":
                    j = 5
                elif month == "07":
                    j = 6
                elif month == "08":
                    j = 7
                elif month == "09":
                    j = 8
                elif month == "10":
                    j = 9
                elif month == "11":
                    j = 10
                elif month == "12":
                    j = 11
                
                q[j+k]= 1
        all_q.append(q)
    return all_q


all_q = remplissage_q()

'''
# On vérifie que tous les vecteurs q sont dans all_q (il y a bine 817 vecteurs q)
print(len(all_q))
'''

'''
#On vérifie que all_q est bien rempli
print(all_q[0])
'''

#on crée les bmin des vendeurs en fonction de chaque q

for q in all_q: # pour associer un bmin à chaque q, on additionne le nombre de 1 dans q pouis on divise ce nombre par 96 (sorte de moyenne)
    nombre_de_uns = sum(q)
    bmin = nombre_de_uns / len(q)
    bid = (-bmin, q)
    all_bids.append(bid)
    
'''
#On teste 
print(all_bids[0])
print(len(all_bids))
'''

###############################################################################

# Côté acheteur

#Une fois le dictionnaire créé, on va l'utiliser pour le Logistic Classifier

def classifier():

    # On choisi un jeu 
    jeu = random.choice(list(donnees_dict.keys())) # on choisit un jeu aléatoire dans le dictionnaire
    game_data = donnees_dict[jeu]
        
    # Initialisation d'une liste vide pour stocker les données hours_watched
    hours_watched_data = []
    
    # Parcours des données du jeu
    for year, year_data in game_data.items():
        for month, month_data in year_data.items():
            hours_watched_data.append(month_data['hours_watched'])
    
    # Conversion de la liste en un tableau numpy
    X = np.array(hours_watched_data, dtype=float).reshape(-1, 1)
        

    print(X)

    # On choisi un seuil
    while True:
        seuil = np.random.randint(np.percentile(X, 25), np.percentile(X, 75)) # on choisit un seuil aléatoire entre le min et le max de X

        # On peut maintenant utiliser X pour entraîner un modèle de régression logistique

        # y est construit à partir d'une condition. Cependant, celle-ci est des fois toujours vraie (ou toujours fausse), ce qui ne permet pas de tester le modèle. Il faut trouver une bonne définition de y
        y = np.where(X > seuil, 1, 0).flatten()
        unique_classes = np.unique(y)
        if len(unique_classes) > 1 and np.min(np.bincount(y)) > 1:
            break

    print(y)
    # Preprocessing (scale the data set)
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # build train/test datasets
    # build train/test datasets
    trainX, testX, trainy, testy = train_test_split(X_scaled, y.flatten(), test_size=0.5, random_state=None, stratify=y.flatten()) # Quelle test_size mettre?
    '''
    trainX, testX, trainy, testy = train_test_split(X_scaled, y, test_size=0.5, random_state=None) # Quelle test_size mettre?
    '''

    # fit a model
    model = LogisticRegression(solver='liblinear',max_iter=500) # Logistic model
    model.fit(trainX, trainy)

    # predict probabilities
    lr_probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    lr_probs = lr_probs[:, 1]

    error = 0
    for j in range(len(lr_probs)):
        temp = np.round(lr_probs[j])
        if temp != testy[j]:
            error = error + 1/len(lr_probs)


    # Clacul de b

    bmax = 1 - error

    return bmax

''' Exemple de jeu et seuil
Call of Duty: Modern Warfare II
10000000
'''


class Brands :
    def __init__(self) :
        self.bid = ()
        
    # Chaque marque propose une bid: si son utilité pour un type de données dépasse un seuil, elle le demande => -1 dans q
    def generate_bid(self):
        bmax = classifier() # on récupère bmax
        q_acheteur = [0 for _ in range(96)] # on initialise q à [0(x96)]
        self.bid = (bmax, q_acheteur)

nb_brands = 817 # nombre de jeux dans le dictionnaire

brands_list = [Brands() for _ in range(nb_brands)]


# Générer une offre pour chaque marque dans la liste et l'ajouter au marché (all_bids)
for brand in brands_list:
    brand.generate_bid() 
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
    #print("WD = ", pulp.value(prob.objective), "\nVariables de décisions : ", [pulp.value(wvars[i]) for i in range(A)])
    cnt = 0 # comptele nb de wi à 1
    for i in range (A) :
        cnt += pulp.value(wvars[i])
    print("cnt = ", cnt, " sur ", len(new_bids))
    
    winning_bids = [new_bids[i] for i in range(A) if pulp.value(wvars[i]) == 1]
    print("Winning bids:")
    for i, bid in enumerate(winning_bids):
        print(f"  Bid {i+1}: {bid}")
    print("\n")


#print(all_bids)
optimization(all_bids)


