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

    def __init__ (self, images, labels, portion = 0.8):
        """
        Un set de données contient huit propriétés:
        deux jeux d'images qu'il contient, divisés selon un pourcentage en paramètre;
        un set de classification par jeu d'images;
        pour chaque jeu d'image, un tableau d'index permettant une stabilité dans la numérotation;
        et le cardinal de chaque jeu.
        Si le pourcentage est paramétré à 100 ou 0, le jeu A sera le seul rempli.
        Par défaut, le chemin correspond au dossier de l'executable
        """
        card_defaut = len(labels[0])
        
        if portion > 1 or portion < 0:
            raise PourcentageIncorrect
        
        if portion == 1 or portion == 0:
            self.imagesA = images
            self.categA = labels[0]
            self.cardinalA = card_defaut
            self.imagesB = np.zeros(1)
            self.categB = np.zeros(1)
            self.cardinalB = 1
            self.indexA = labels[0]
            self.indexB = np.zeros(1)

        if portion > 0 and portion < 1: #En commentaires, des notes sur la forme de chaque variable
            indices = np.random.permutation(images.shape[0])
            setA_indx, setB_indx = indices[:int(portion*card_defaut)], indices[int(portion*card_defaut):]
            self.indexA = setA_indx #matrice numpy de forme (portion*card_defaut,) (la virgule n'est pas accidentelle!)
            self.indexB = setB_indx #contiennent les "anciens" indices des images, dans l'ordre de leur nouvelle base
            self.imagesA = images[setA_indx,:] #matrice numpy de forme (portion*card_defaut, 784) 
            self.imagesB = images[setB_indx,:] #contiennent les images, dans le nouvel ordre, sous forme de vecteurs de R784
            self.categA = labels[:,setA_indx] #matrice numpy de forme (1, portion*card_defaut)
            self.categB = labels[:,setB_indx] #ignorez le premier index, l'important est que le second vecteur fait correspondre les images à leur valeur, au même index
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
            return self.indexA[i]
        if bD == 'B':
            return self.indexB[i]
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

    def affichage_index_bD(self, i, bD):
        """
        affiche l'image numérotée i dans la base de données spécifiée
        """
        if bD != 'A' and bD != 'B':
            raise MauvaiseBD
        self.affichage(i, bD)
        

    def sortir_image(self, i):
        """
        renvoie l'image numérotée i dans la base originelle
        """
        iC, bD = self.index_dans_bd(i)
        if bD == 'A':
            return self.imagesA[iC]
        if bD == 'B':
            return self.imagesB[iC]

    def sortir_chiffre(self, n, bD = 'A'):
        """
        Renvoie toutes les images correspondant au chiffre indiqué, dans une des bases
        Par défaut, la base A est utilisée
        Les images renvoyées sont toujours sous le même format numpy
        """
        if n < 0 or n > 9:
            raise MauvaisChiffre
        if bD == 'A':
            index_temp = np.where(self.categA[0] == n)[0]
            images_temp = np.zeros((len(index_temp), 784))
            j = 0
            for i in index_temp:
                images_temp[j] = self.imagesA[i]
                j+=1
            return images_temp

        if bD == 'B':
            index_temp = np.where(self.categB[0] == n)[0]
            images_temp = np.zeros((len(index_temp), 784))
            j = 0
            for i in index_temp:
                images_temp[j] = self.imagesB[i]
                j+=1
            return images_temp

        else:
            raise MauvaiseBD

    def exporter_mat(self):
        """
        enregistre la base de données sous la même forme standart que l'originale;
        un fichier pour la base de test, un pour la base d'apprentissage
        """
        mdic_apprentissage = {"data": np.transpose(self.imagesA), "label": self.categA}
        scipy.io.savemat("base_apprentissage.mat", mdic_apprentissage)
        mdic_test = {"data": np.transpose(self.imagesB), "label":self.categB}
        scipy.io.savemat("base_test.mat", mdic_test)
        








        

