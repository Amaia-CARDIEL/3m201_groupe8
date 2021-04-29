from sans_lissage import *

filehandle = open('sl_scal.txt', 'w')

précision_totale=[]
filehandle.write(f'\n Scaling')
for j in range(10):
    précision_totale+=[precis_S(j)*100]
    filehandle.write(f'\n Précision pour le chiffre {j} est : {précision_totale[j]}' )
filehandle.write(f'\n Précision totale est : {sum(précision_totale)/10}' )
