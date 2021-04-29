from sans_lissage import *

filehandle = open('sl_toy.txt', 'w')

précision_totale=[]
filehandle.write(f'\n ToY')
for j in range(10):
    précision_totale+=[precis_Y(j)*100]
    filehandle.write(f'\n Précision pour le chiffre {j} est : {précision_totale[j]}' )
filehandle.write(f'\n Précision totale est : {sum(précision_totale)/10}' )
