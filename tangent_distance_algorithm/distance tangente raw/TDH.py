from baseline import *

p_TDH_train=tuples_y*np.array(p_x_train)+tuples_x*np.array(p_y_train)
p_TDH_test=tuples_y*np.array(p_x_test)+tuples_x*np.array(p_y_test)

# ajout overfit
p_TDH_test_overfit=tuples_y*np.array(p_x_test_overfit)+tuples_x*np.array(p_y_test_overfit)

def estim_TDH(j): #estime l'image data_test[j]
    A=[np.hstack((np.reshape(-p_TDH_train[i],(-1,1)),np.reshape(p_TDH_test[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(smooth_train[i]-smooth_test[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

# ajout overfit

def estim_TDH_overfit(j): #estime l'image data_test_overfit[j]
    A=[np.hstack((np.reshape(-p_TDH_train[i],(-1,1)),np.reshape(p_TDH_test_overfit[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(smooth_train[i]-smooth_test_overfit[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

# LISTES A STOCKER

estim_TDH=[estim_TDH(i) for i in range(2000)]
#estim_TDH_overfit=[estim_TDH_overfit(i) for i in range(2000)]

with open('estim_TDH.txt', 'w') as filehandle:
    for listitem in estim_TDH:
        filehandle.write('%s\n' % listitem)
"""

with open('estim_TDH_overfit.txt', 'w') as filehandle:
    for listitem in estim_TDH_overfit:
        filehandle.write('%s\n' % listitem)
"""
