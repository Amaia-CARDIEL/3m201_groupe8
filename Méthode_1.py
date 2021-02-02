
#Les instructions suivantes permettent de charger les données de chiffres manuscrits: (Code proposé par M.Nataf lors de la Séance 1)
import scipy.io as spi
import numpy as np
import matplotlib.pyplot as plt
mat=spi.loadmat("mnist-original.mat")
data=np.transpose(mat['data'])
label=np.array(mat['label']) #label: chiffre numérisé
label=label.astype(int) #Les labels sont stockés en flottants, on les convertit en entiers

#Notre premier programme de classification est basé sur l'algorithme suivant:
#Etape 1: On partage notre base de données en deux parties: la première partie servira de base d'apprentissage avec
#80% des données et la deuxième partie servira de base de tests avec 20% des données.
#Etape 2: Dans la base d'apprentissage, on calcule les centroïdes des classes de 0 à 9 en utilisant une distance donnée.
#Etape 3: Pour chaque vecteur de la base de tests, on lui attribue le chiffre dont le centroïde est le plus proche.
#Etape 4: Finalement, on déduit une estimation du pourcentage de prédictions correctes sur la base de tests.
#Plusieurs distances peuvent être utilisées. Dans cette partie, on choisit de travailler avec la distance euclidienne et la mesure de similarité cosinus
#On comparera leur précision respective. 

###########################################
#           Distance euclidienne          #
###########################################

#Etape 1 : Définir la base d'apprentissage et la base de tests

Y,y=data,label[0]
#On change l'ordre des données et des labels avec la même permutation pour que data_test et data_app soient hétérogènes.
m=np.random.permutation((len(y))) #permutation arbitraire
Y_m=Y[m] #data après permutation
y_m=y[m] #labels après permutation
n=len(Y) #nombre d'images dans la base donnée
n_80=80*n/100 #nombre d'images dans la base d'apprentissage
n_80=int(n_80)
data_app=Y_m[:n_80] #base d'apprentissage
label_app=y_m[:n_80]
data_test=Y_m[n_80:]
label_test=y_m[n_80:]

#Etape 2 : Dans la base d'apprentissage, calcul de la valeur moyenne des classes de 0 à 9 et affichage de l'image moyenne de chaque chiffre

X,x=data_app,label_app
moy_chiff=[]
plt.figure(figsize=(15,2))
for i in range(10):
  moy=np.mean(X[x==i],axis=0)
  moy_chiff+=[moy]
  plt.subplot(1,10,i+1)
  plt.imshow(moy.reshape(28,28),cmap='gray')

#Etape 3 : Pour chaque vecteur de la base de tests, on lui attribue le chiffre dont le centroïde est le plus proche par rapport à la norme euclidienne

#Définition de la fonction qui estime le chiffre d'un vecteur de la base de tests:
def estim_chiffre_1(v):
  distances=np.array([np.linalg.norm(v-u) for u in moy_chiff]) #distances entre v et les "chiffres moyens"
  return np.argmin(distances)

#Labels estimés pour les vecteurs de la base de tests:
k=len(data_test)
label_estim_1=np.zeros(k)
for i in range(k):
  label_estim_1[i]=estim_chiffre_1(data_test[i])

# Etape 4 : Finalement, on déduit une estimation du pourcentage de prédictions correctes sur la base de tests.

np.mean(label_estim_1==label_test)

# Conclusion: Cet algorithme donne alors une estimation exacte d'un chiffre manuscrit dans environ 80% des cas.

#############################################
#       Mesure de similitude Cosinus        #
#############################################

#Les deux premières étapes sont les mêmes que la partie 1.

#Etape 3 : Pour chaque vecteur de la base de tests, on lui attribue le chiffre dont le centroïde est le plus proche par rapport à la "distance" cosinus.

#On définit la "distance" cosine:
def cosine(u,v):
  return np.inner(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))

#Définition de la fonction qui estime le chiffre d'un vecteur de la base de tests:
def estim_chiffre_2(v):
  distances=np.array([cosine(u,v) for u in moy_chiff]) #distances cosine entre v et les "chiffres moyens"
  return np.argmax(distances) #On prend le max car plus deux vecteurs sont proches, plus la "distance" cos est grande (proche de 1)

#Labels estimés pour les vecteurs de la base de tests:
k=len(data_test)
label_estim_2=np.zeros(k)
for i in range(k):
  label_estim_2[i]=estim_chiffre_2(data_test[i])

#Etape 4 : Finalement, on déduit une estimation du pourcentage de prédictions correctes sur la base de tests.

np.mean(label_estim_2==label_test)

#Conclusion: Cet algorithme donne alors une estimation exacte d'un chiffre manuscrit dans 81% des cas. 

####################################
#      Premières conclusions       #
####################################

#La similarité cosinus semble donner une précision plus grande que la distance euclidienne sur la base de notre base de données.
