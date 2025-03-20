# Importación de módulos necesarios
from SonarDataset import SonarDataset  # Importa el dataset personalizado de Sonar
from torch.utils.data import random_split  # Para dividir el dataset
import torch  # Framework PyTorch
from Model import *  # Importa el modelo definido externamente
import torch.nn as nn  # Módulos de redes neuronales de PyTorch

def split_dataset(dataset):
    # Función para dividir el dataset en conjuntos de entrenamiento y validación
    lonxitudeDataset = len(dataset)  # Obtiene el tamaño total del dataset
    tamTrain = int(lonxitudeDataset * 0.8)  # Calcula el 80% para entrenamiento
    tamVal = lonxitudeDataset - tamTrain  # El resto para validación
    print(f"Tam dataset: {lonxitudeDataset} train: {tamTrain} tamVal: {tamVal}")
    
    # Divide el dataset aleatoriamente según las proporciones calculadas
    train_set, val_set = random_split(dataset, [tamTrain, tamVal])
    
    # Crea los DataLoader con configuraciones específicas
    # DataLoader de entrenamiento con batch_size=2 y mezcla los datos
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=2,
                                             shuffle=True, drop_last=False)
    # DataLoader de validación con batch_size=4 y usa múltiples workers para carga paralela
    validation_loader = torch.utils.data.DataLoader(val_set, batch_size=4,
                                                  shuffle=False, num_workers=2)
    return train_loader, validation_loader

# Carga el dataset desde los archivos especificados
dataset = SonarDataset("data/sonar.all-data",".")
print(dataset[0])  # Muestra el primer elemento del dataset

# Divide el dataset en conjuntos de entrenamiento y validación
train_loader, validation_loader = split_dataset(dataset)

# Inicializa el modelo con 60 características de entrada
model = Model(60)  # La dimensión 60 corresponde al número de características en los datos

# Configura el optimizador Adam con learning rate de 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define la función de pérdida como Entropía Cruzada (adecuada para clasificación)
loss_fn = nn.CrossEntropyLoss()

print(model)  # Muestra la arquitectura del modelo

# Obtiene un lote (batch) de prueba del DataLoader de entrenamiento
entradaProba, dest = next(iter(train_loader))
print("Entrada:")
print(entradaProba)  # Muestra los datos de entrada
print("Desexada:")
print(dest)  # Muestra las etiquetas deseadas (ground truth)

# Realiza una pasada hacia adelante (forward pass) con el modelo
saida = model(entradaProba)
print("Saída:")
print(saida)  # Muestra la salida del modelo

# Calcula la pérdida para este batch de prueba
loss_fn(saida, dest)  # No se guarda el resultado, solo se ejecuta como prueba

def train_one_epoch(epoch_index):
    # Función para entrenar el modelo durante una época
    running_loss = 0.
    last_loss = 0.
    
    # Itera sobre todos los batches en el DataLoader de entrenamiento
    for i, data in enumerate(train_loader):
        # Obtiene entradas y etiquetas del batch actual
        inputs, labels = data
        
        # Paso 1: Pone a cero los gradientes acumulados
        optimizer.zero_grad()
        
        # Paso 2: Realiza la pasada hacia adelante
        outputs = model(inputs)
        
        # Paso 3: Calcula la pérdida
        loss = loss_fn(outputs, labels)
        
        # Paso 4: Retropropagación - calcula gradientes
        loss.backward()
        
        # Paso 5: Actualiza los pesos del modelo usando el optimizador
        optimizer.step()
        
        # Acumula la pérdida para el seguimiento
        running_loss += loss.item()
        
        # Imprime la pérdida promedio cada 10 batches
        if i % 10 == 9:
            last_loss = running_loss / 10
            print(' batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0.
    
    return last_loss

# Configuración del entrenamiento
EPOCHS = 100  # Número total de épocas para entrenar

# Tensores para almacenar métricas durante el entrenamiento
loss_list = torch.zeros((EPOCHS,))  # Almacena la pérdida por época
accuracy_list = torch.zeros((EPOCHS,))  # Almacena la precisión por época

lonxitudeDataset = len(dataset)  # Tamaño total del dataset

# Bucle principal de entrenamiento
for epoch in range(EPOCHS):
    print('EPOCH {}:'.format(epoch + 1))
    
    # Modo entrenamiento
    model.train(True)  # Activa el modo de entrenamiento (afecta a capas como BatchNorm, Dropout)
    avg_loss = train_one_epoch(epoch)  # Entrena una época y obtiene la pérdida promedio
    loss_list[epoch] = avg_loss  # Guarda la pérdida de esta época
    
    # Modo evaluación/validación
    model.train(False)  # Desactiva el modo de entrenamiento para evaluación
    running_vloss = 0.0
    
    # Evalúa el modelo en el conjunto de validación
    for i, vdata in enumerate(validation_loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)  # Pasada hacia adelante sin cálculo de gradientes
        
        # Calcula la pérdida de validación
        vloss = loss_fn(voutputs, vlabels)
        
        # Calcula la precisión - compara predicciones con etiquetas reales
        correct = (torch.argmax(voutputs, dim=0) == vlabels).type(torch.FloatTensor)
        accuracy_list[epoch] += correct.sum()  # Suma el número de predicciones correctas
        
        running_vloss += vloss  # Acumula la pérdida de validación
    
    # Calcula el promedio de pérdida de validación
    avg_vloss = running_vloss / (i + 1)
    
    # Imprime métricas de la época: pérdida de entrenamiento, pérdida de validación y precisión
    print('LOSS train {} valid {} {}/{}'.format(
        avg_loss, avg_vloss, accuracy_list[epoch], int(lonxitudeDataset * 0.2)))