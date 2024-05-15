import csv
from collections import defaultdict
import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot


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

# Remove the games from the dictionary
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

# Ou afficher seulement les premiers éléments de chaque jeu
for game_name, game_data in list(donnees_dict.items())[:5]:
    print(f"{game_name}: {game_data}\n")

########################################################

#Une fois le dictionnaire créé, on va l'utiliser pour le Logistic Classifier

def classifier():

    # On choisi un jeu 
    while True:
        jeu = input("Choisissez un jeu :")

        # On vérifie si le jeu existe dans le dictionnaire
        if jeu in donnees_dict:
            game_data = donnees_dict[jeu]
            
            # Initialisation d'une liste vide pour stocker les données hours_watched
            hours_watched_data = []
            
            # Parcours des données du jeu
            for year, year_data in game_data.items():
                for month, month_data in year_data.items():
                    hours_watched_data.append(month_data['hours_watched'])
            
            # Conversion de la liste en un tableau numpy
            X = np.array(hours_watched_data, dtype=float).reshape(-1, 1)
            break
        else:
            print(f"Le jeu {jeu} n'apparaît pas dans le dictionnaire.")
            continue
    
    print(X)

    # On choisi un seuil
    while True:
        seuil = int(input("Choisissez un seuil :"))

        # On peut maintenant utiliser X pour entraîner un modèle de régression logistique

        # y est construit à partir d'une condition. Cependant, celle-ci est des fois toujours vraie (ou toujours fausse), ce qui ne permet pas de tester le modèle. Il faut trouver une bonne définition de y
        y = np.where(X > seuil, 1, 0)
        if np.all(y == 0):
            print("Attention : la condition est toujours fausse. Veuillez choisir un autre seuil ou le modèle peut être défectueux")
        elif np.all(y == 1):
            print("Attention : la condition est toujours vraie. Veuillez choisir un autre seuil ou le modèle peut être défectueux")
        else:
            break

    print(y)
    # Preprocessing (scale the data set)
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    # build train/test datasets
    trainX, testX, trainy, testy = train_test_split(X_scaled, y, test_size=0.5, random_state=None) # Quelle test_size mettre?

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
# On utilise la fonction classifier pour obtenir b
bmax = classifier()
print(f"bmax = {bmax}")
