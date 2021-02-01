import scipy.io
import numpy as np
import matplotlib.pyplot
from exceptions import*

#On ouvre le fichier de données (qui doit être dans le dossier de l'exécutable !)
def lecture(chemin):
    """
    méthode qui ouvre le fichier de données au nom spécifié en argument
    renvoie les données, indexées par ordre;
    et les affectations de chaque chiffre, indexées aussi par ordre
    """
    mat = scipy.io.loadmat(chemin)
    data = np.transpose(mat['data'])
    label = np.array(mat['label'])
    label = label.astype(int)
    return data, label


class Donnees():
    """
    objet contenant les images de la base de données sous forme de matrice numpy,
    et le chiffre correspondant à chaque image.
    Tout est indexé selon le "numéro" de chaque image.
    Le paramètre optionnel "portion" permet de diviser la base de données aléatoirement selon un pourcentage.
    Par défaut, la division est de 80%.
    """

    def __init__ (self, chemin = "mnist-original.mat", portion = 0.8):
        """
        Un set de données contient huit propriétés:
        deux jeux d'images qu'il contient, divisés selon un pourcentage en paramètre;
        un set de classification par jeu d'images;
        pour chaque jeu d'image, un tableau d'index permettant une stabilité dans la numérotation;
        et le cardinal de chaque jeu.
        Si le pourcentage est paramétré à 100 ou 0, le jeu A sera le seul rempli.
        Par défaut, le chemin correspond au dossier de l'executable
        """
        
        images, labels = lecture(chemin)
        card_defaut = len(labels[0])
        
        if portion > 1 or portion < 0:
            raise PourcentageIncorrect
        
        if portion == 1 or portion == 0:
            self.imagesA = images
            self.categA = labels[0]
            self.cardinalA = card_defaut
            self.imagesB = np.zeroes(1)
            self.categB = np.zeroes(1)
            self.cardinalB = 1

        if portion > 0 and portion < 1:
            indices = np.random.permutation(images.shape[0])
            setA_indx, setB_indx = indices[:int(portion*card_defaut)], indices[int(portion*card_defaut):]
            self.indexA = setA_indx
            self.indexB = setB_indx
            self.imagesA = images[setA_indx,:]
            self.imagesB = images[setB_indx,:]
            self.categA = labels[:,setA_indx]
            self.categB = labels[:,setB_indx]
            self.cardinalA = int(card_defaut*portion)
            self.cardinalB = int(card_defaut*(1-portion))
            
        """
        principe de l'index:
        je veux accéder à l'image numéro i dans la base de données originale.
        à supposer que je sais dans quelle base de données elle est (disons A),
        iA = int(np.where(self.indexA == i)[0]) est l'index de l'image i dans la base A
        self.imagesA[iA] me renvoie cette image.
        inversement, je veux savoir quel est l'index original de la i-ème image de ma base A:
        i0 = indexA[i] est exactement cet index.

        Je mets ça en commentaire pour plus de clarté, mais le processus est implémenté entièrement dans les
        méthodes respectives index_dans_bd et index_originel
        """


    def index_dans_bd(self, i):
        """
        renvoie l'index d'une image i dans la nouvelle répartition, et la base de donnée dans laquelle elle est
        """
        if len(np.where(self.indexA == i)[0]) != 0:
            return int(np.where(self.indexA == i)[0]), 'A'
        if len(np.where(self.indexB == i)[0]) != 0:
            return int(np.where(self.indexB == i)[0]), 'B'
        else:
            raise IndexIncorrect

    def index_originel(self, i, bD):
        """
        renvoie l'index originel d'une image i dans une BD spécifiée
        """
        if bD == 'A':
            return indexA[i]
        if bD == 'B':
            return indexB[i]
        else:
            raise MauvaiseBD

    def affichage(self, i):
        """
        affiche l'image numérotée i (dans la base de données originale !)
        """
        index_converti, bD = self.index_dans_bd(i)
        if bD == 'A':
            V = self.imagesA[index_converti].reshape((28,28))
            matplotlib.pyplot.imshow(V, cmap = 'gray',vmin = 0 ,vmax = 255)
            matplotlib.pyplot.show()
        if bD == 'B':
            V = self.imagesB[index_converti].reshape((28,28))
            matplotlib.pyplot.imshow(V, cmap = 'gray',vmin = 0 ,vmax = 255)
            matplotlib.pyplot.show()
        if bD != 'A' and bD != 'B':
            raise MauvaiseBD

    def sortir_image(self, i):
        """
        renvoie l'image numérotée i dans la base originelle
        """
        iC, bD = self.index_dans_bd(i)
        if bD == 'A':
            return self.imagesA[iC]
        if bD == 'B':
            return self.imagesB[iC]

