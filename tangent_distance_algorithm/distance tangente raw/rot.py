from baseline import *

p_r_train = tuples_y*np.array(p_x_train)-tuples_x*np.array(p_y_train)
p_r_test = tuples_y*np.array(p_x_test)-tuples_x*np.array(p_y_test)

# ajout overfit
p_r_test_overfit=tuples_y*np.array(p_x_test_overfit)-tuples_x*np.array(p_y_test_overfit)

def estim_R(j): #estime l'image data_test[j]
    A=[np.hstack((np.reshape(-p_r_train[i],(-1,1)),np.reshape(p_r_test[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(smooth_train[i]-smooth_test[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

# ajout overfit

def estim_R_overfit(j): #estime l'image data_test_overfit[j]
    A=[np.hstack((np.reshape(-p_r_train[i],(-1,1)),np.reshape(p_r_test_overfit[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(smooth_train[i]-smooth_test_overfit[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

# LISTES A STOCKER

estim_rotation=[estim_R(i) for i in range(2000)]
estim_rotation_overfit=[estim_R_overfit(i) for i in range(2000)]

np.savez("estim_rotation", estim_rotation)
np.savez("estim_rotation_overfit", estim_rotation_overfit)

