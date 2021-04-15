"""
Attention ! Ce fichier n'a pas pour but d'être exécuté:
il contient simplement le code à copier (et adapter en conséquence)
pour lire un fichier contenant une des listes sauvegardées
"""

NOM_DE_LA_LISTE = []

with open('NOM_DU_FICHIER.txt', 'r') as filehandle:
    for line in filehandle:
        currentPlace = line[:-1]
        NOM_DE_LA_LISTE.append(currentPlace)



"""
#pour les fichiers npz, qu'on utilise finalement pas, mais je laisse ça au cas où

zfile = np.load("NOM_DU_FICHIER.npz")

NOM_DE_LA_LISTE = [zfile[label] for label in zfile.files]
"""
