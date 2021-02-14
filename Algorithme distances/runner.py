import scipy.io
import numpy as np
import matplotlib.pyplot
from exceptions import*
from base import*


def moyennes(dataset, bD = 'A'):
    """
    renvoie une nouvelle base de données contenant uniquement les moyennes des images pour chaque chiffre;
    dans un jeu spécifié (par défaut, A)
    """

    if bD != 'A' and bD != 'B':
        raise MauvaiseBD
    
    images = np.zeros((10, 784))
    labels = np.zeros((1, 10))

    for i in range(0,10):
        imTemp = dataset.sortir_chiffre(i, bD)
        images[i] = np.mean(imTemp, 0)
        labels[0][i] = i

    return Donnees(images, labels, portion = 1)

        

