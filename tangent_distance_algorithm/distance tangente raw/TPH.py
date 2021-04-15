from baseline import *

p_TPH_train=tuples_x*np.array(p_x_train)-tuples_y*np.array(p_y_train)
p_TPH_test=tuples_x*np.array(p_x_test)-tuples_y*np.array(p_y_test)

# ajout overfit
p_TPH_test_overfit=tuples_x*np.array(p_x_test_overfit)-tuples_y*np.array(p_y_test_overfit)

def estim_TPH(j): #estime l'image data_test[j]
    A=[np.hstack((np.reshape(-p_TPH_train[i],(-1,1)),np.reshape(p_TPH_test[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(smooth_train[i]-smooth_test[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

# ajout overfit

def estim_TPH_overfit(j): #estime l'image data_test_overfit[j]
    A=[np.hstack((np.reshape(-p_TPH_train[i],(-1,1)),np.reshape(p_TPH_test_overfit[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(smooth_train[i]-smooth_test_overfit[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

# LISTES A STOCKER

estim_TPH=[estim_TPH(i) for i in range(2000)]
#estim_TPH_overfit=[estim_TPH_overfit(i) for i in range(2000)]

with open('estim_TPH.txt', 'w') as filehandle:
    for listitem in estim_TPH:
        filehandle.write('%s\n' % listitem)
"""
with open('estim_TPH_overfit.txt', 'w') as filehandle:
    for listitem in estim_TPH_overfit:
        filehandle.write('%s\n' % listitem)

"""
