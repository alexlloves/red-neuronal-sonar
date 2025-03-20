import torch.nn.functional as F
import torch.nn as nn


class Model(nn.Module):
    """
    Modelo de red neuronal para clasificación con tres capas densas (fully connected).

    La red tiene la siguiente arquitectura:
    - Capa lineal 1: Convierte la entrada en un vector de 50 dimensiones.
    - Capa lineal 2: Mantiene la dimensionalidad en 50.
    - Capa lineal 3: Reduce la dimensionalidad a 2 (salida con dos clases).
    
    Funciones de activación:
    - ReLU en las dos primeras capas ocultas para introducir no linealidad.
    - Softmax en la capa de salida para convertir los valores en probabilidades.

    Args:
        input_dim (int): Dimensión de la entrada (cantidad de características del dataset).
    """
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(in_features=50, out_features=2)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x