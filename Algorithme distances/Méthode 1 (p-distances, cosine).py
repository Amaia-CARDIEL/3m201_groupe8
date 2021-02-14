#!/usr/bin/env python
# coding: utf-8

# #  LU3MA201 : Projet / Travail d’étude et de recherche

# <!-- dom:AUTHOR: Aya Bouzidi at [Sorbonne Université](http://www.sorbonne-universite.fr/), -->
# <!-- Author: -->  
# **Aya Bouzidi**, **Amaia Cardiel**, **Camille Grimal**, **Elysé Miadantsoa Rasoloarivony** ( Etudiants en L3 de Mathématiques à [Sorbonne Université](http://www.sorbonne-universite.fr/) ).
# 
# Sous la direction de : **Fréderic Nataf** ( Directeur de recherche au [Laboratoire J.L. Lions](https://www.ljll.math.upmc.fr/), [Sorbonne Université](http://www.sorbonne-universite.fr/) )
# 
# Licence <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/">CC BY-NC-ND</a>

# # 1 Algorithme simple de reconnaissance de chiﬀres manuscrits

# <div id="ch:method_1"></div>
# 
# Ce travail est une introduction à l'apprentissage automatique, il s'agit d'écrire un programme simple de reconnaissance de chiffres manuscrits. Pour cela, on utilise la base de données **MNIST** très utilisée en machine learning. Cette base est constituée de **70 000** images de chiffres écrits à la main, chaque image est constituée de **28x28** pixels. Voici un exemple de chiffres de la base MNIST ([source](https://fr.wikipedia.org/wiki/Base_de_donn%C3%A9es_MNIST#/media/Fichier:MnistExamples.png)).
# 
# 
# 
# Les instructions suivantes permettent de charger les données de chiffres manuscrits disponibles dans le package mnist-original.mat :
# 

# In[14]:


#Code proposé par M.Nataf:Séance 1
import scipy.io as spi
import numpy as np
import matplotlib.pyplot as plt
mat=spi.loadmat("mnist-original.mat")
data=np.transpose(mat['data'])
label=np.array(mat['label']) #label: chiffre numérisé
label=label.astype(int) #Les labels sont stockés en flottants, on les convertit en entiers


# <div id="ch:method_1"></div>
# Regardons un exemple:

# In[15]:


#Lecture du 2021-ième chiffre de la base de données:
print('Le chiffre est',label[0][12021])
#Affichage du chiffre:
plt.imshow(data[12021].reshape(28,28),cmap='gray')
plt.show()


# <div id="ch:method_1"></div>
# Le chiffre codé est donc bien un 1.

# <div id="ch:method_1"></div>
# 
# **Notre premier programme de classification est basé sur l'algorithme suivant:**
# 
# * **Etape 1:** On partage notre base de données en deux parties: la première partie servira de base d'apprentissage avec 80% de données et la deuxième partie servira de base de tests avec 20% de données.
# 
# * **Etape 2:** Dans la base d'apprentissage, on calcule les centroïdes des classes de 0 à 9 en utilisant une distance donnée.
# 
# * **Etape 3:** Pour chaque vecteur de la base de tests, on lui attribue le chiffre dont le centroïde est le plus proche.
# 
# * **Etape 4:** On déduit une estimation du pourcentage de prédictions correctes sur la base de tests.
# 
# * **Etape 5:** Pour chaque chiffre, on déduit une estimation du pourcentage de prédictions correctes.
# 
# Plusieurs distances peuvent être utilisées. Dans cette partie, on choisit de travailler avec la distance euclidienne et la distance cosine et de comparer leurs précisions.
# 
# **Distances utilisées**:
# 
# * Distance euclidienne
# 
# * Similarité cosinus (cosine)
# 

# # 1.1 Distance euclidienne

# <div id="method_1"></div>
# 
# **Etape 1 :** Définir la base d'apprentissage et la base de tests.
# 

# In[16]:


Y,y=data,label[0]
#On change l'ordre des données et des labels avec la même permutation pour que data__test et data_app soient hétérogènes...
m=np.random.permutation((len(y))) #permutation arbitraire
Y_m=Y[m] #data après permutation
y_m=y[m] #labels après permutation
n=len(Y) #nombre d'images dans la base donnée
n_80=80*n/100 #nombre d'images dans la base d'apprentissage
n_80=int(n_80)
data_app=Y_m[:n_80] #base d'apprentissage
label_app=y_m[:n_80]
data_test=Y_m[n_80:] #base de tests
label_test=y_m[n_80:]


# <div id="method_1"></div>
# 
# **Etape 2 :** Dans la base d'apprentissage, on calcule la valeur moyenne des classes de 0 à 9 et on affiche l'image moyenne associée à chaque chiffre.
# 

# In[17]:


X,x=data_app,label_app
moy_chiff=[]
plt.figure(figsize=(15,2))
for i in range(10):
    moy=np.mean(X[x==i],axis=0)
    moy_chiff+=[moy]
    plt.subplot(1,10,i+1)
    plt.imshow(moy.reshape(28,28),cmap='gray')


# <div id="method_1"></div>
# 
# **Etape 3 :** Pour chaque vecteur de la base de tests, on lui attribue le chiffre dont le centroïde est le plus proche par rapport à la distance euclidienne.

# In[18]:


#Définition de la fonction qui estime le chiffre d'un vecteur de la base de tests:
def estim_chiffre_1(v):
    distances=np.array([np.linalg.norm(v-u) for u in moy_chiff]) #distances entre v et les "chiffres moyens"
    return np.argmin(distances)


# In[19]:


#Labels estimés pour les vecteurs de la base de tests:
k=len(data_test)
label_estim_1=np.zeros(k)
for i in range(k):
    label_estim_1[i]=estim_chiffre_1(data_test[i])


# <div id="method_1"></div>
# 
# **Etape 4 :** On déduit une estimation du pourcentage de prédictions correctes sur la base de tests.

# In[20]:


np.mean(label_estim_1==label_test)


# <div id="method_1"></div>
# 
# **Conclusion**: Cet algorithme donne alors une estimation exacte d'un chiffre manuscrit dans 80% des cas.

# <div id="method_1"></div>
# 
# **Etape 3 :** Pour chaque chiffre, on déduit une estimation du pourcentage de prédictions correctes.

# In[21]:


#On définit la fonction qui donne la liste des chiffres i dans la base de tests: 
long=len(label_test)
def estim_1(i):
    lae=[] #list des labels estimés pour les chiffres i
    for j in range(long):
        if label_test[j]==i:
            lae+=[estim_chiffre_1(data_test[j])]
    long1=len(lae)
    laee=np.array(lae)
    la=np.array([i for j in range(long1)]) 
    return np.mean(la==laee)
#Estimation du pourcentage de prédictions correctes pour chaque chiffre: 
for i in range(10):
    print("précision pour le chiffre", i ,"est de" , estim_1(i))


# In[22]:


plt.xlabel(r'Chiffres')
plt.ylabel(r'Précisions')
x=[i for i in range(10)]
y=[estim_1(i) for i in range(10)]
plt.plot(x, y, marker='o')
plt.savefig("test.png", dpi=100) # exporte la figure en PNG


# <div id="method_1"></div>
# 
# **Remarque**: Le chiffre 1 a la plus grande précision, le chiffre 5 a la plus petite précision.

# # 1.1 Distance de Minkowski: p-distance (généralisation)

# <div id="method_1"></div>
# Les deux premières étapes sont les mêmes que la partie 1.

# <div id="method_1"></div>
# 
# **Etape 3 :** Pour chaque vecteur de la base de tests, on lui attribue le chiffre dont le centroïde est le plus proche par rapport à la distance p.

# In[23]:


#Définition de la fonction qui estime le chiffre d'un vecteur de la base de tests en p-distance:
def estim_chiffre(v,p):
    distances=np.array([np.linalg.norm(v-u,p) for u in moy_chiff]) #distances entre v et les "chiffres moyens"
    return np.argmin(distances)


# <div id="method_1"></div>
# 
# **Etape 4 :** On déduit une estimation du pourcentage de prédictions correctes sur la base de tests pour chaque p-distance.

# In[24]:


#Labels estimés pour les vecteurs de la base de tests:
k=len(data_test)
for p in range(3,11):
    label_estim=np.zeros(k)
    for i in range(k):
        label_estim[i]=estim_chiffre(data_test[i],p)
    print("précision pour la distance", p ,"est de" ,np.mean(label_estim==label_test))


# <div id="method_1"></div>
# 
# **Etape 3 :** Pour chaque chiffre et chaque p-distance, on déduit une estimation du pourcentage de prédictions correctes.

# In[25]:


#On définit la fonction qui donne la liste des chiffres i dans la base de tests: 
long=len(label_test)
def estim(i,p):
    lae=[] #list des labels estimés pour les chiffres i
    for j in range(long):
        if label_test[j]==i:
            lae+=[estim_chiffre(data_test[j],p)]
    long1=len(lae)
    laee=np.array(lae)
    la=np.array([i for j in range(long1)]) 
    return np.mean(la==laee)
#Estimation du pourcentage de prédictions correctes pour chaque chiffre et chaque distance: 
for p in range(3,11):
    for i in range(10):
        print("précision pour le chiffre", i ,"pour la distance", p, "est de" , estim(i,p)) 
    print("")


# In[26]:


plt.xlabel(r'Chiffres')
plt.ylabel(r'Précisions')
for p in range(3,11):
    x=[i for i in range(10)]
    y=[estim(i,p) for i in range(10)]
    plt.plot(x, y, marker='o', label=(r"distance",p))
plt.legend()
plt.show()
plt.savefig("test.png", dpi=100) # exporte la figure en PNG


# <div id="method_1"></div>
# 
# **Remarque**: Pour toutes ces distances, le chiffre 0 a la plus grande précision. On remarque que plus on augmente p, plus les précisions de 1, 4 et 7 diminuent. Les autres précisions sont presque constantes. 

# # 1.2 Similarité cosinus

# In[27]:


#On définit la fonction cosine: 
def cosine(u,v):
    return np.inner(u,v)/(np.linalg.norm(u)*np.linalg.norm(v))


# <div id="method_1"></div>
# Les deux premières étapes sont les mêmes que la partie 1.

# <div id="method_1"></div>
# 
# **Etape 3 :** Pour chaque vecteur de la base de tests, on lui attribue le chiffre dont le centroïde est le plus proche par rapport à la distance cosine.

# In[28]:


#Définition de la fonction qui estime le chiffre d'un vecteur de la base de tests:
def estim_chiffre_2(v):
    distances=np.array([cosine(u,v) for u in moy_chiff]) #distances cosine entre v et les "chiffres moyens"
    return np.argmax(distances) #On prend le max car plus l'angle est petit, plus le cos est grand


# In[29]:


#Labels estimés pour les vecteurs de la base de tests:
k=len(data_test)
label_estim_2=np.zeros(k)
for i in range(k):
    label_estim_2[i]=estim_chiffre_2(data_test[i])


# In[30]:


np.mean(label_estim_2==label_test)


# <div id="method_1"></div>
# 
# **Conclusion**: Cet algorithme donne alors une estimation exacte d'un chiffre manuscrit dans 81% des cas.

# <div id="method_1"></div>
# 
# **Etape 3 :** Pour chaque chiffre, on déduit une estimation du pourcentage de prédictions correctes.

# In[31]:


#On définit la fonction qui donne la liste des chiffres i dans la base de tests: 
long=len(label_test)
def estim_2(i):
    lae=[] #list des labels estimés pour les chiffres i
    for j in range(long):
        if label_test[j]==i:
            lae+=[estim_chiffre_2(data_test[j])]
    long1=len(lae)
    laee=np.array(lae)
    la=np.array([i for j in range(long1)]) 
    return np.mean(la==laee)
#Estimation du pourcentage de prédictions correctes pour chaque chiffre: 
for i in range(10):
    print("précision pour le chiffre", i ,"est de" , estim_2(i))


# In[32]:


plt.xlabel(r'Chiffres')
plt.ylabel(r'Précisions')
x=[i for i in range(10)]
y=[estim_2(i) for i in range(10)]
plt.plot(x, y, marker='o')
plt.savefig("test.png", dpi=100) # exporte la figure en PNG


# <div id="method_1"></div>
# 
# **Remarque**: Le chiffre 1 a la plus grande précision, le chiffre 5 a la plus petite précision.

# In[ ]:




