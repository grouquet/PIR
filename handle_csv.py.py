import pulp
import numpy as np
import csv

nom_fichier = './twitchdata-update.csv'

donnees = []

# Ouverture du fichier CSV en mode lecture
with open(nom_fichier, mode='r', newline='', encoding='utf-8') as fichier_csv:
    # Création d'un lecteur CSV
    lecteur_csv = csv.reader(fichier_csv)
    
    # Parcours des lignes du fichier CSV
    for ligne in lecteur_csv:
        # Ajout de la ligne à la liste des données
        donnees.append(ligne)

# format des lignes : 
# Channel,Watch time(Minutes),Stream time(minutes),Peak viewers,Average viewers,Followers,Followers gained,Views gained,Partnered,Mature,Language



S = 100 # nombre d'agents

# matrice qui contient les offres : chaque sous-liste contient les offres d'un agent
all_bids = [[] for i in range (S)]


# création des offres
cnt = 0 # permet de remplir all_bids
for i in range (len(donnees)) :
    q = []
    for k in range (11) : 
        valeur = donnees[i][k]
        # on convertit en int si possible
        try : 
            val_convertie = int(valeur)
        except ValueError:
            if valeur == 'True' :
                val_convertie = True
            elif valeur == 'False' :
                val_convertie = False
            else :
                val_convertie = valeur
        q.append(val_convertie)
    #q.append(indice) # indice de tendance à ajouter
    q.append(int(donnees[i][1])/int(donnees[i][2])) # taux de rétention
    q.append(int(donnees[i][1])/int(donnees[i][5])) # engagement audience
    b = np.random.random() - 0.5
    all_bids.append((b, q))
    cnt += 1
    if cnt >= S :
        cnt = 0

print(all_bids)

def filtered_bids(all_bids, donnée_ciblée, critère) :
    S = len(all_bids)
    filt_bids = [[] for i in range (S)]
    for bids in all_bids : 
        for bid in bids :
                
    if type(all_bidsdonnée_ciblée) == 




