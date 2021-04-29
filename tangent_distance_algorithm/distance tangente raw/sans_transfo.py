import scipy.io as spi
import numpy as np
import matplotlib.pyplot as plt
import time

mat=spi.loadmat("base_apprentissage.mat")
data_train=np.transpose(mat['data'])
label_train=np.array(mat['label'])[0] #label: chiffre numérisé
label_train=label_train.astype(int) #Les labels sont stockés en flottants, on les convertit en entiers

mat = spi.loadmat("base_test.mat")
data_test = np.transpose(mat['data'])
label_test = np.array(mat['label'])[0]
label_test =label_test.astype(int)

filehandle = open('sans_transfo.txt', 'w')

tuples=[]
for i in range(28):
    for j in range(28):
        tuples+=[(i,j)]
        
tuples_x=np.array([tuples[i][0] for i in range(784)])
tuples_y=np.array([tuples[i][1] for i in range(784)])

def smooth_fcl(v):
    P = np.reshape(v, (28,28))
    x, y = tuples_x, tuples_y
    return sum([P[i,j]*np.e**(-((x-i)**2+(y-j)**2)/(2*0.9**2)) for i,j in tuples])

smooth_train=[smooth_fcl(data_train[i]) for i in range(8000)]
smooth_test=[smooth_fcl(data_test[i]) for i in range(2000)]

def estim_sans_transfo(j): #estime l'image data_test[j]
    résidus=[np.linalg.norm(smooth_test[j]-smooth_train[i]) for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

def precis_sans_transfo(j):
    digits_j=[i for i in range(2000) if label_test[i]==j]
    a=len(digits_j)
    estimations=[estim_sans_transfo(i) for i in digits_j]
    return list(estimations).count(j)/a

précision_totale=[]
for j in range(10):
    précision_totale+=[precis_sans_transfo(j)*100]
    filehandle.write(f'\n Précision pour le chiffre {j} est : {précision_totale[j]}' )
filehandle.write(f'\n Précision totale est : {sum(précision_totale)/10}' )
