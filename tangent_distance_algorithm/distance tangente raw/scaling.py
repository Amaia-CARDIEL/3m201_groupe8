from baseline import *

p_s_train=tuples_x*np.array(p_x_train)+tuples_y*np.array(p_y_train)
p_s_test=tuples_x*np.array(p_x_test)+tuples_y*np.array(p_y_test)

# ajout overfit
p_s_test_overfit=tuples_x*np.array(p_x_test_overfit)+tuples_y*np.array(p_y_test_overfit)

def estim_S(j): #estime l'image data_test[j]
    A=[np.hstack((np.reshape(-p_s_train[i],(-1,1)),np.reshape(p_s_test[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(smooth_train[i]-smooth_test[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

# ajout overfit

def estim_S_overfit(j): #estime l'image data_test_overfit[j]
    A=[np.hstack((np.reshape(-p_s_train[i],(-1,1)),np.reshape(p_s_test_overfit[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(smooth_train[i]-smooth_test_overfit[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

# LISTES A STOCKER

estim_scaling=[estim_S(i) for i in range(2000)]
estim_scaling_overfit=[estim_S_overfit(i) for i in range(2000)]

np.savez("estim_scaling", estim_scaling)
np.savez("estim_scaling_overfit", estim_scaling_overfit)

