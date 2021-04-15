from baseline import *

def estim_Y(j): #estime l'image data_test[j]
    A=[np.hstack((np.reshape(-p_y_train[i],(-1,1)),np.reshape(p_y_test[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(smooth_train[i]-smooth_test[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

# ajout overfit

def estim_Y_overfit(j): #estime l'image data_test_overfit[j]
    A=[np.hstack((np.reshape(-p_y_train[i],(-1,1)),np.reshape(p_y_test_overfit[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(smooth_train[i]-smooth_test_overfit[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

# LISTES A STOCKER

estim_y_trans=[estim_Y(i) for i in range(2000)]
#estim_y_trans_overfit=[estim_Y_overfit(i) for i in range(2000)]

with open('estim_y_trans.txt', 'w') as filehandle:
    for listitem in estim_y_trans:
        filehandle.write('%s\n' % listitem)
"""
with open('estim_y_trans_overfit.txt', 'w') as filehandle:
    for listitem in estim_y_trans_overfit:
        filehandle.write('%s\n' % listitem)
"""
