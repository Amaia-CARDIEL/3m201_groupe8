from baseline import *

p_T_train=np.array(p_x_train)**2+np.array(p_y_train)**2
p_T_test=np.array(p_x_test)**2+np.array(p_y_test)**2

# ajout overfit
p_T_test_overfit=np.array(p_x_test_overfit)**2+np.array(p_y_test_overfit)**2

def estim_T(j): #estime l'image data_test[j]
    A=[np.hstack((np.reshape(-p_T_train[i],(-1,1)),np.reshape(p_T_test[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(smooth_train[i]-smooth_test[j],(-1,1)) for i in range(8000)]
    résidus=[np.linalg.lstsq(A[i], b[i], rcond=None)[1][0] for i in range(8000)] 
    return label_train[résidus.index(min(résidus))]

# ajout overfit

def estim_T_overfit(j): #estime l'image data_test_overfit[j]
    A=[np.hstack((np.reshape(-p_T_train[i],(-1,1)),np.reshape(p_T_train[j],(-1,1)))) for i in range(8000)]
    b=[np.reshape(smooth_train[i]-smooth_train[j],(-1,1)) for i in range(8000)]
    x=[np.linalg.lstsq(np.transpose(A[i])@A[i], np.transpose(A[i])@b[i],rcond=None)[0] for i in range(8000)]
    résidus=[np.linalg.norm(A[i]@x[i]-b[i]) for i in range(8000)]
    return label_train[résidus.index(min(résidus))]

# LISTES A STOCKER

#estim_thick=[estim_T(i) for i in range(2000)]
estim_thick_overfit=[estim_T_overfit(i) for i in range(2000)]
"""
with open('estim_thick.txt', 'w') as filehandle:
    for listitem in estim_thick:
        filehandle.write('%s\n' % listitem)

"""
with open('estim_thick_overfit.txt', 'w') as filehandle:
    for listitem in estim_thick_overfit:
        filehandle.write('%s\n' % listitem)
        

