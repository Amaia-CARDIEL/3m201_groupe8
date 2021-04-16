from baseline import *

def estim_X(j): #estime l'image data_test[j]
    A=[np.hstack((np.reshape(-p_x_train[i],(-1,1)),np.reshape(p_x_test[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(smooth_train[i]-smooth_test[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

# ajout overfit

def estim_X_overfit(j): #estime l'image data_test_overfit[j]
    A=[np.hstack((np.reshape(-p_x_train[i],(-1,1)),np.reshape(p_x_train[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(smooth_train[i]-smooth_train[j],(-1,1)) for i in range(8000)]
    x=[np.linalg.lstsq(np.transpose(A[i])@A[i], np.transpose(A[i])@b[i],rcond=None)[0] for i in range(8000)]
    résidus=[np.linalg.norm(A[i]@x[i]-b[i]) for i in range(8000)]
    return label_train[résidus.index(min(résidus))]# LISTES A STOCKER

#estim_x_trans=[estim_X(i) for i in range(2000)] # pour matrices de confusion
estim_x_trans_overfit=[estim_X_overfit(i) for i in range(2000)] # pour overfit
"""
with open('estim_x_trans.txt', 'w') as filehandle:
    for listitem in estim_x_trans:
        filehandle.write('%s\n' % listitem)

"""
with open('estim_x_trans_overfit.txt', 'w') as filehandle:
    for listitem in estim_x_trans_overfit:
        filehandle.write('%s\n' % listitem)

