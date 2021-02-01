

class PourcentageIncorrect(Exception):
    """
    Valeur de pourcentage incorrecte
    """
    def __init__ (self, message = "La valeur d'un pourcentage doit être entre 0 et 1"):
        self.message = message
    def __str__(self):
        return self.message

class MauvaiseBD(Exception):
    """
    Base de donnée incorrecte
    """
    def __init__ (self, message = "La base de données requise n'existe pas"):
        self.message = message
    def __str__(self):
        return self.message

class IndexIncorrect(Exception):
    """
    Index incorrect
    """
    def __init__ (self, message = "Il n'existe pas d'image a cet index"):
        self.message = message
    def __str__(self):
        return self.message
