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

filehandle = open('sans_lissage.txt', 'w')

tuples=[]
for i in range(28):
    for j in range(28):
        tuples+=[(i,j)]
        
tuples_x=np.array([tuples[i][0] for i in range(784)])
tuples_y=np.array([tuples[i][1] for i in range(784)])

def T_X(v):
    P=np.reshape(v,(28,28))
    P_=np.gradient(P)[0]
    return np.reshape(P_,(1,-1))[0]

p_x_train=[T_X(data_train[i]) for i in range(8000)]
p_x_test=[T_X(data_test[i]) for i in range(2000)]

plt.imshow(np.reshape(p_x_train[0],(28,28)), cmap='gray_r')

def T_Y(v):
    P=np.reshape(v,(28,28))
    P_=np.gradient(P)[1]
    return np.reshape(P_,(1,-1))[0]

p_y_train=[T_Y(data_train[i]) for i in range(8000)]
p_y_test=[T_Y(data_test[i]) for i in range(2000)]

plt.imshow(np.reshape(p_y_train[0],(28,28)), cmap='gray_r')

def estim_X(j): #estime l'image data_test[j]
    A=[np.hstack((np.reshape(-p_x_train[i],(-1,1)),np.reshape(p_x_test[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(data_train[i]-data_test[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

def precis_X(j):
    digits_j=[i for i in range(2000) if label_test[i]==j]
    a=len(digits_j)
    estimations=[estim_X(i) for i in digits_j]
    return list(estimations).count(j)/a

précision_totale=[]
filehandle.write(f'\n ToX')
for j in range(10):
    précision_totale+=[precis_X(j)*100]
    filehandle.write(f'\n Précision pour le chiffre {j} est : {précision_totale[j]}' )
filehandle.write(f'\n Précision totale est : {sum(précision_totale)/10}' )

def precis_Y(j):
    digits_j=[i for i in range(2000) if label_test[i]==j]
    a=len(digits_j)
    estimations=[estim_Y(i) for i in digits_j]
    return list(estimations).count(j)/a


précision_totale=[]
filehandle.write(f'\n ToY')
for j in range(10):
    précision_totale+=[precis_Y(j)*100]
    filehandle.write(f'\n Précision pour le chiffre {j} est : {précision_totale[j]}' )
filehandle.write(f'\n Précision totale est : {sum(précision_totale)/10}' )

p_r_train=tuples_y*np.array(p_x_train)-tuples_x*np.array(p_y_train)
p_r_test=tuples_y*np.array(p_x_test)-tuples_x*np.array(p_y_test)

plt.imshow(np.reshape(p_r_train[0],(28,28)), cmap='gray_r')

def estim_R(j): #estime l'image data_test[j]
    A=[np.hstack((np.reshape(-p_r_train[i],(-1,1)),np.reshape(p_r_test[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(data_train[i]-data_test[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

def precis_R(j):
    digits_j=[i for i in range(2000) if label_test[i]==j]
    a=len(digits_j)
    estimations=[estim_R(i) for i in digits_j]
    return list(estimations).count(j)/a

précision_totale=[]
filehandle.write(f'\n Rot')
for j in range(10):
    précision_totale+=[precis_R(j)*100]
    filehandle.write(f'\n Précision pour le chiffre {j} est : {précision_totale[j]}' )
filehandle.write(f'\n Précision totale est : {sum(précision_totale)/10}' )

p_s_train=tuples_x*np.array(p_x_train)+tuples_y*np.array(p_y_train)
p_s_test=tuples_x*np.array(p_x_test)+tuples_y*np.array(p_y_test)

plt.imshow(np.reshape(p_s_train[0],(28,28)), cmap='gray_r')

def estim_S(j): #estime l'image data_test[j]
    A=[np.hstack((np.reshape(-p_s_train[i],(-1,1)),np.reshape(p_s_test[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(data_train[i]-data_test[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)]
    return label_train[résidus.index(min(résidus))]

def precis_S(j):
    digits_j=[i for i in range(2000) if label_test[i]==j]
    a=len(digits_j)
    estimations=[estim_S(i) for i in digits_j]
    return list(estimations).count(j)/a

précision_totale=[]
filehandle.write(f'\n Scaling')
for j in range(10):
    précision_totale+=[precis_S(j)*100]
    filehandle.write(f'\n Précision pour le chiffre {j} est : {précision_totale[j]}' )
filehandle.write(f'\n Précision totale est : {sum(précision_totale)/10}' )

p_TPH_train=tuples_x*np.array(p_x_train)-tuples_y*np.array(p_y_train)
p_TPH_test=tuples_x*np.array(p_x_test)-tuples_y*np.array(p_y_test)

plt.imshow(np.reshape(p_TPH_train[0],(28,28)), cmap='gray_r')

def estim_TPH(j): #estime l'image data_test[j]
    A=[np.hstack((np.reshape(-p_TPH_train[i],(-1,1)),np.reshape(p_TPH_test[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(data_train[i]-data_test[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)]
    return label_train[résidus.index(min(résidus))]

def precis_TPH(j):
    digits_j=[i for i in range(2000) if label_test[i]==j]
    a=len(digits_j)
    estimations=[estim_TPH(i) for i in digits_j]
    return list(estimations).count(j)/a

précision_totale=[]
filehandle.write(f'\n TPH')
for j in range(10):
    précision_totale+=[precis_TPH(j)*100]
    filehandle.write(f'\n Précision pour le chiffre {j} est : {précision_totale[j]}' )
filehandle.write(f'\n Précision totale est : {sum(précision_totale)/10}' )

p_TDH_train=tuples_y*np.array(p_x_train)+tuples_x*np.array(p_y_train)
p_TDH_test=tuples_y*np.array(p_x_test)+tuples_x*np.array(p_y_test)

plt.imshow(np.reshape(p_TDH_train[0],(28,28)), cmap='gray_r')

def estim_TDH(j): #estime l'image data_test[j]
    A=[np.hstack((np.reshape(-p_TDH_train[i],(-1,1)),np.reshape(p_TDH_test[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(data_train[i]-data_test[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)]
    return label_train[résidus.index(min(résidus))]

def precis_TDH(j):
    digits_j=[i for i in range(2000) if label_test[i]==j]
    a=len(digits_j)
    estimations=[estim_TDH(i) for i in digits_j]
    return list(estimations).count(j)/a

précision_totale=[]
filehandle.write(f'\n TDH')
for j in range(10):
    précision_totale+=[precis_TDH(j)*100]
    filehandle.write(f'\n Précision pour le chiffre {j} est : {précision_totale[j]}' )
filehandle.write(f'\n Précision totale est : {sum(précision_totale)/10}' )

p_T_train=np.array(p_x_train)**2+np.array(p_y_train)**2
p_T_test=np.array(p_x_test)**2+np.array(p_y_test)**2

def estim_T(j): #estime l'image data_test[j]
    A=[np.hstack((np.reshape(-p_T_train[i],(-1,1)),np.reshape(p_T_test[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(data_train[i]-data_test[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)]
    return label_train[résidus.index(min(résidus))]

def precis_T(j):
    digits_j=[i for i in range(2000) if label_test[i]==j]
    a=len(digits_j)
    estimations=[estim_T(i) for i in digits_j]
    return list(estimations).count(j)/a

précision_totale=[]
filehandle.write(f'\n Translation')
for j in range(10):
    précision_totale+=[precis_T(j)*100]
    filehandle.write(f'\n Précision pour le chiffre {j} est : {précision_totale[j]}' )
filehandle.write(f'\n Précision totale est : {sum(précision_totale)/10}' )
