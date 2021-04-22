import scipy.io as spi
import numpy as np
import matplotlib.pyplot as plt

mat=spi.loadmat("base_apprentissage.mat")
data_train=np.transpose(mat['data'])
label_train=np.array(mat['label'])[0] #label: chiffre numérisé
label_train=label_train.astype(int) #Les labels sont stockés en flottants, on les convertit en entiers

mat = spi.loadmat("base_test.mat")
data_test = np.transpose(mat['data'])
label_test = np.array(mat['label'])[0]
label_test =label_test.astype(int)

tuples=[]
for i in range(28):
    for j in range(28):
        tuples+=[(i,j)]
        
tuples_x=np.array([tuples[i][0] for i in range(784)])
tuples_y=np.array([tuples[i][1] for i in range(784)])

def smooth(v):
    P = np.reshape(v, (28,28))
    x, y = tuples_x, tuples_y
    return sum([P[i,j]*np.e**(-((x-i)**2+(y-j)**2)/(2*0.9**2)) for i,j in tuples])

smooth_train=[smooth(data_train[i]) for i in range(8000)]
smooth_test=[smooth(data_test[i]) for i in range(2000)]

def T_X(v):
    P=np.reshape(v,(28,28))
    P_=np.gradient(P)[0]
    return np.reshape(P_,(1,-1))[0]

p_x_train=[T_X(smooth_train[i]) for i in range(8000)]
p_x_test=[T_X(smooth_test[i]) for i in range(2000)]

def T_Y(v):
    P=np.reshape(v,(28,28))
    P_=np.gradient(P)[1]
    return np.reshape(P_,(1,-1))[0]

p_y_train=[T_Y(smooth_train[i]) for i in range(8000)]
p_y_test=[T_Y(smooth_test[i]) for i in range(2000)]

p_r_train=tuples_y*np.array(p_x_train)-tuples_x*np.array(p_y_train)
p_r_test=tuples_y*np.array(p_x_test)-tuples_x*np.array(p_y_test)

p_s_train=tuples_x*np.array(p_x_train)+tuples_y*np.array(p_y_train)
p_s_test=tuples_x*np.array(p_x_test)+tuples_y*np.array(p_y_test)

p_TPH_train=tuples_x*np.array(p_x_train)-tuples_y*np.array(p_y_train)
p_TPH_test=tuples_x*np.array(p_x_test)-tuples_y*np.array(p_y_test)

p_TDH_train=tuples_y*np.array(p_x_train)+tuples_x*np.array(p_y_train)
p_TDH_test=tuples_y*np.array(p_x_test)+tuples_x*np.array(p_y_test)

p_T_train=np.array(p_x_train)**2+np.array(p_y_train)**2
p_T_test=np.array(p_x_test)**2+np.array(p_y_test)**2

matrix_T_train=[np.hstack([p_x_train[i].reshape(-1,1)]+[p_y_train[i].reshape(-1,1)]+[p_r_train[i].reshape(-1,1)]+[p_s_train[i].reshape(-1,1)]+[p_TPH_train[i].reshape(-1,1)]+[p_TDH_train[i].reshape(-1,1)]+[p_T_train[i].reshape(-1,1)]) for i in range(8000)]
matrix_T_test=[np.hstack([p_x_test[i].reshape(-1,1)]+[p_y_test[i].reshape(-1,1)]+[p_r_test[i].reshape(-1,1)]+[p_s_test[i].reshape(-1,1)]+[p_TPH_test[i].reshape(-1,1)]+[p_TDH_test[i].reshape(-1,1)]+[p_T_test[i].reshape(-1,1)]) for i in range(2000)]

def estim(j): #estime l'image data_test[j]
    A=[np.hstack((-matrix_T_train[i],matrix_T_test[j])) for i in range(8000)]
    b=[np.reshape(smooth_train[i]-smooth_test[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

def precis(j):
    digits_j=[i for i in range(2000) if label_test[i]==j]
    a=len(digits_j)
    estimations=[estim(i) for i in digits_j]
    return list(estimations).count(j)/a

with open('all_mat_pres.txt', 'w') as filehandle:
    
    for j in range(10):
        p_temp = precis(j)*100
        filehandle.write(f'\n Précision pour le chiffre {j} est :  {p_temp}')
