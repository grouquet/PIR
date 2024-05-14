import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
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




# generate data
iter = 1000 # number of data samples
#dim = 7 # dimension of each sample
X = [[] for i in range(iter)]
for i in range (iter) :
    X[i].append(donnees[i][1])
    X[i].append(donnees[i][2])
    X[i].append(donnees[i][3])
    X[i].append(donnees[i][4])
    X[i].append(donnees[i][5])
y = [[] for i in range(iter)]
for i in range (iter) :
    y[i].append(donnees[i][6])

# Preprocessing (scale the data set)
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

# build train/test datasets
trainX, testX, trainy, testy = train_test_split(X_scaled, y, test_size=0.5, random_state=None)

# fit a model
model = LogisticRegression(solver='liblinear',max_iter=500) # Logistic model
model.fit(trainX, trainy)

# predict probabilities
lr_probs = model.predict_proba(testX)
print(lr_probs)
# keep probabilities for the positive outcome only
lr_probs = lr_probs[:, 1]

error = 0
for j in range(len(lr_probs)):
    temp = np.round(lr_probs[j])
    if temp != testy[j]:
        error = error + 1/len(lr_probs)
print(error)